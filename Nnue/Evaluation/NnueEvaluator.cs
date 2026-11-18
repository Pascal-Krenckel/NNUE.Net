using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Nnue.Backends;
using Nnue.Configuration;
using Nnue.Kernels;
using Nnue.Memory;
using Nnue.Models;

namespace Nnue.Evaluation;

/// <summary>
/// Updates accumulators and evaluates them. Stateless and thread safe: share one evaluator between threads, give every thread its own
/// accumulators. The library does not know what the accumulators mean; the caller decides which accumulators form the slots of an evaluation
/// and in which order (slot order = the order of the arguments).
///
/// Every operation exists twice: the plain method validates its arguments and throws; the <c>*Unchecked</c> method does not and is
/// meant for tight loops with trusted input (measured with BenchmarkDotNet: the checks cost 0-10 % of an evaluation and up to about a
/// quarter of a small incremental update, where the feature indices are validated). Checked methods also throw
/// <see cref="ObjectDisposedException"/> if the model or the buffer of an accumulator was disposed. Unchecked methods with invalid arguments
/// (wrong accumulator size, feature or bucket out of range, wrong number of slots, disposed memory) read or write out of bounds.
///
/// <code>
/// var model = NnueModel.Load("net.nnue");
/// var eval = new NnueEvaluator(model);
/// var accumulator = model.CreateAccumulators(256);                 // e.g. search stack: ply * slots
/// eval.RefreshAccumulator(acc[0], features);                          // from scratch
/// eval.UpdateAccumulator(acc[0], acc[1], removed, added);             // child = parent - removed + added; going back to the parent is free
/// int score = eval.Evaluate(acc[1], acc[2], bucket);       // two slots
/// </code>
/// </summary>
public sealed unsafe class NnueEvaluator
{
    private readonly EvaluationContext _context;
    private readonly FnTable _functions;
    private readonly int _accumulatorSize, _slotCount, _bucketCount, _featureCount;
    private readonly NativeBlock _modelMemory;          // checked calls refuse to run on a disposed model

    /// <summary>The model this evaluator reads.</summary>
    public NnueModel Model { get; }
    /// <summary>The instruction set the kernels of this evaluator use.</summary>
    public SimdBackend Backend { get; }
    /// <summary>How the first dense layer of an Int8 model treats zero input chunks.</summary>
    public SparseInputMode SparseInput { get; }

    /// <summary>Creates an evaluator for <paramref name="model"/>.</summary>
    /// <param name="model">The network. It must stay undisposed while the evaluator is used.</param>
    /// <param name="backend">The instruction set; null picks <see cref="SimdSupport.Best"/>.</param>
    /// <param name="sparseInput">How the first dense layer of an Int8 model treats zero input chunks.</param>
    /// <exception cref="PlatformNotSupportedException"><paramref name="backend"/> is not supported on this CPU.</exception>
    public NnueEvaluator(NnueModel model, SimdBackend? backend = null, SparseInputMode sparseInput = SparseInputMode.Auto)
    {
        ArgumentNullException.ThrowIfNull(model);
        model.ThrowIfDisposed();
        if (backend is { } requested && !Enum.IsDefined(requested))
            throw new ArgumentOutOfRangeException(nameof(backend), $"Unknown backend {(int)requested}.");
        if (!Enum.IsDefined(sparseInput))
            throw new ArgumentOutOfRangeException(nameof(sparseInput), $"Unknown sparse input mode {(int)sparseInput}.");
        SimdBackend b = backend ?? SimdSupport.Best;
        if (!SimdSupport.IsSupported(b))
            throw new PlatformNotSupportedException($"{b} is not supported on this CPU.");
        Model = model;
        Backend = b;
        SparseInput = sparseInput;
        _modelMemory = model.Memory;
        NetworkSpec spec = model.Spec;
        NetworkLayout layout = model.Layout;
        _accumulatorSize = layout.AccumulatorSize;
        _slotCount = layout.Slots;
        _bucketCount = spec.OutputBuckets;
        _featureCount = layout.FeatureCount;
        _context = new EvaluationContext
        {
            Model = model,
            FeatureWeights = model.FeatureWeightsPointer,
            FeatureBias = model.FeatureBiasPointer,
            AccumulatorSize = _accumulatorSize,
            SlotCount = _slotCount,
            FeatureCount = _featureCount,
            TransformerOutputSize = layout.FeatureTransformerOutputSize,
            ChunkCount = layout.ChunkCount,
            TotalActivationSize = layout.TotalActivationSize,
            MaxPaddedOutputLength = layout.MaxPaddedOutputLength,
            LayerCount = layout.DenseLayers.Count,
            BucketCount = _bucketCount,
            ClipMax = spec.ActivationQuantization,
            IsInt8 = spec.Precision == NetworkPrecision.Int8,
            HiddenActivation = spec.HiddenActivation,
            HasSkipConnection = spec.SkipConnection,
            Activation = spec.Activation,
            Steps = model.Steps,
            OutputWeights = model.OutputWeightsPointer,
            OutputBiases = model.OutputBiasPointer,
            OutputWeightsStride = (int)(layout.OutputWeightsBucketStride / sizeof(short)),
            SparseMode = sparseInput,
            Scale = spec.Precision == NetworkPrecision.Int8 ? spec.OutputScale / ((double)NetworkSpec.Int8One * NetworkSpec.Int8WeightScale)
                  : spec.Activation == FeatureTransformerActivation.SquaredClippedReLU ? spec.OutputScale / ((double)spec.ActivationQuantization * spec.ActivationQuantization * spec.OutputWeightQuantization)
                  : spec.OutputScale / ((double)spec.ActivationQuantization * spec.OutputWeightQuantization),
        };
        _functions = FnTable.For(b);
    }

    // ================================================================== argument checks
    // The checks are combined with non-short-circuit operators so that the valid path costs one predictable branch,
    // and every wrapper is inlined into the caller: passing the 24-byte Accumulator handles through a second call level
    // cost more than the checks themselves in the first version (+29 % on a tiny int16 evaluation).
    [DoesNotReturn, MethodImpl(MethodImplOptions.NoInlining)]
    private static void Fail(string message) => throw new ArgumentException(message);
    [DoesNotReturn, MethodImpl(MethodImplOptions.NoInlining)]
    private static void FailRange(string name, string message) => throw new ArgumentOutOfRangeException(name, message);

    [DoesNotReturn, MethodImpl(MethodImplOptions.NoInlining)]
    private void FailEval(int slots, int bucket, ReadOnlySpan<Accumulator> given)
    {
        if (slots != _slotCount)
            Fail($"The model consumes {_slotCount} accumulator(s) per evaluation, got {slots}.");
        foreach (Accumulator a in given)
            if (ModelGone || Bad(a))
                CheckAccSlow(a);
        FailRange(nameof(bucket), $"Bucket {bucket} out of range [0, {_bucketCount}).");
    }

    [DoesNotReturn, MethodImpl(MethodImplOptions.NoInlining)]
    private void CheckAccSlow(Accumulator accumulator)
    {
        if (_modelMemory.IsDisposed)
            throw new ObjectDisposedException(nameof(NnueModel), "The model of this evaluator was disposed.");
        if (accumulator.Owner is { IsDisposed: true })
            throw new ObjectDisposedException(nameof(AccumulatorBuffer), "The buffer of this accumulator was disposed.");
        if (accumulator.IsEmpty)
            Fail("Default (uninitialised) accumulator.");
        Fail($"Accumulator size {accumulator.Length} does not match the model ({_accumulatorSize}).");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool Bad(Accumulator accumulator)
        => accumulator.Length != _accumulatorSize || accumulator.Owner!.IsDisposed;     // a default handle has Length 0, so Owner is only read when set

    /// <summary>True if the model was disposed (checked once per call).</summary>
    private bool ModelGone
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _modelMemory.IsDisposed;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CheckAcc(Accumulator accumulator) { if (ModelGone | Bad(accumulator)) CheckAccSlow(accumulator); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CheckBucket(int bucket) { if ((uint)bucket >= (uint)_bucketCount) FailRange(nameof(bucket), $"Bucket {bucket} out of range [0, {_bucketCount})."); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CheckSlotCount(int n) { if (n != _slotCount) Fail($"The model consumes {_slotCount} accumulator(s) per evaluation, got {n}."); }

    private void CheckFeatures(ReadOnlySpan<int> features)
    {
        foreach (int f in features)
            if ((uint)f >= (uint)_featureCount)
                FailRange(nameof(features), $"Feature index {f} out of range [0, {_featureCount}).");
    }

    // ================================================================== feature transformer
    /// <summary>Sets the accumulator to the bias, i.e. "no features active".</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ResetAccumulator(Accumulator accumulator) { CheckAcc(accumulator); ResetAccumulatorUnchecked(accumulator); }
    /// <summary><see cref="ResetAccumulator"/> without argument checks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ResetAccumulatorUnchecked(Accumulator accumulator) { Unsafe.CopyBlock(accumulator.Pointer, _context.FeatureBias, (uint)(_accumulatorSize * sizeof(short))); GC.KeepAlive(_context); GC.KeepAlive(accumulator.Owner); }

    /// <summary>Recomputes the accumulator from scratch: bias + the weights of all active features (kept in registers, written once).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RefreshAccumulator(Accumulator accumulator, ReadOnlySpan<int> features) { CheckAcc(accumulator); CheckFeatures(features); RefreshAccumulatorUnchecked(accumulator, features); }
    /// <summary><see cref="RefreshAccumulator"/> without argument checks: feature indices are not validated.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RefreshAccumulatorUnchecked(Accumulator accumulator, ReadOnlySpan<int> features) { _functions.RefreshAccumulator(_context, accumulator.Pointer, features); GC.KeepAlive(_context); GC.KeepAlive(accumulator.Owner); }

    /// <summary>destination = source - removed + added in one pass (source may be the destination). With one accumulator per search ply,
    /// "unmake move" is free.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UpdateAccumulator(Accumulator source, Accumulator destination, ReadOnlySpan<int> removed, ReadOnlySpan<int> added)
    {
        if (ModelGone | Bad(source) | Bad(destination))
            CheckAccSlow(ModelGone || Bad(source) ? source : destination);
        CheckFeatures(removed);
        CheckFeatures(added);
        _functions.UpdateAccumulator(_context, source.Pointer, destination.Pointer, removed, added);
        GC.KeepAlive(_context);
        GC.KeepAlive(source.Owner);
        GC.KeepAlive(destination.Owner);
    }
    /// <summary><see cref="UpdateAccumulator"/> without argument checks: feature indices are not validated.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UpdateAccumulatorUnchecked(Accumulator source, Accumulator destination, ReadOnlySpan<int> removed, ReadOnlySpan<int> added)
    {
        _functions.UpdateAccumulator(_context, source.Pointer, destination.Pointer, removed, added);
        GC.KeepAlive(_context);
        GC.KeepAlive(source.Owner);
        GC.KeepAlive(destination.Owner);
    }

    /// <summary>Adds the weights of one feature to the accumulator (the feature became active).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddFeature(Accumulator accumulator, int feature)
    {
        if (ModelGone | Bad(accumulator))
            CheckAccSlow(accumulator);
        if ((uint)feature >= (uint)_featureCount)
            FailRange(nameof(feature), $"Feature index {feature} out of range [0, {_featureCount}).");
        _functions.AddFeature(_context, accumulator.Pointer, feature);
        GC.KeepAlive(_context);
        GC.KeepAlive(accumulator.Owner);
    }
    /// <summary><see cref="AddFeature"/> without argument checks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddFeatureUnchecked(Accumulator accumulator, int feature) { _functions.AddFeature(_context, accumulator.Pointer, feature); GC.KeepAlive(_context); GC.KeepAlive(accumulator.Owner); }

    /// <summary>Subtracts the weights of one feature from the accumulator (the feature became inactive).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RemoveFeature(Accumulator accumulator, int feature)
    {
        if (ModelGone | Bad(accumulator))
            CheckAccSlow(accumulator);
        if ((uint)feature >= (uint)_featureCount)
            FailRange(nameof(feature), $"Feature index {feature} out of range [0, {_featureCount}).");
        _functions.RemoveFeature(_context, accumulator.Pointer, feature);
        GC.KeepAlive(_context);
        GC.KeepAlive(accumulator.Owner);
    }
    /// <summary><see cref="RemoveFeature"/> without argument checks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RemoveFeatureUnchecked(Accumulator accumulator, int feature) { _functions.RemoveFeature(_context, accumulator.Pointer, feature); GC.KeepAlive(_context); GC.KeepAlive(accumulator.Owner); }

    // ================================================================== head
    private struct Ptr1 { public short* A; }
    private struct Ptr2 { public short* A, B; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Score(int raw) => (int)(raw * _context.Scale);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int EvaluateSingleSlot(short* first, int bucket) { Ptr1 p; p.A = first; return Score(_functions.Evaluate(_context, &p.A, bucket)); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int EvaluatePair(short* first, short* second, int bucket) { Ptr2 p; p.A = first; p.B = second; return Score(_functions.Evaluate(_context, &p.A, bucket)); }

    /// <summary>One slot (the model must have Slots = 1). Returns the score in score units (network output * OutputScale).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Evaluate(Accumulator first, int bucket = 0)
    {
        if (_slotCount != 1 | ModelGone | Bad(first) | (uint)bucket >= (uint)_bucketCount)
            FailEval(1, bucket, new[] { first });
        int score = EvaluateSingleSlot(first.Pointer, bucket);
        GC.KeepAlive(_context);
        GC.KeepAlive(first.Owner);
        return score;
    }
    /// <summary><see cref="Evaluate(Accumulator, int)"/> without argument checks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int EvaluateUnchecked(Accumulator first, int bucket = 0)
    {
        int score = EvaluateSingleSlot(first.Pointer, bucket);
        GC.KeepAlive(_context);
        GC.KeepAlive(first.Owner);
        return score;
    }

    /// <summary>Two slots, in this order (the model must have Slots = 2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Evaluate(Accumulator first, Accumulator second, int bucket = 0)
    {
        if (_slotCount != 2 | ModelGone | Bad(first) | Bad(second) | (uint)bucket >= (uint)_bucketCount)
            FailEval(2, bucket, new[] { first, second });
        int score = EvaluatePair(first.Pointer, second.Pointer, bucket);
        GC.KeepAlive(_context);
        GC.KeepAlive(first.Owner);
        GC.KeepAlive(second.Owner);
        return score;
    }
    /// <summary><see cref="Evaluate(Accumulator, Accumulator, int)"/> without argument checks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int EvaluateUnchecked(Accumulator first, Accumulator second, int bucket = 0)
    {
        int score = EvaluatePair(first.Pointer, second.Pointer, bucket);
        GC.KeepAlive(_context);
        GC.KeepAlive(first.Owner);
        GC.KeepAlive(second.Owner);
        return score;
    }

    /// <summary>Any number of slots, in the given order.</summary>
    public int Evaluate(ReadOnlySpan<Accumulator> slots, int bucket = 0)
    {
        CheckSlotCount(slots.Length);
        foreach (Accumulator a in slots)
            CheckAcc(a);
        CheckBucket(bucket);
        return EvaluateUnchecked(slots, bucket);
    }
    /// <summary><see cref="Evaluate(ReadOnlySpan{Accumulator}, int)"/> without argument checks.</summary>
    public int EvaluateUnchecked(ReadOnlySpan<Accumulator> slots, int bucket = 0) => Score(EvaluateRawUnchecked(slots, bucket));

    /// <summary>The integer network output before the final scaling (scale: 127*64 for Int8, ActivationQuantization*OutputWeightQuantization or ActivationQuantization^2*OutputWeightQuantization for Int16). Used by tests.</summary>
    internal int EvaluateRaw(ReadOnlySpan<Accumulator> slots, int bucket = 0)
    {
        CheckSlotCount(slots.Length);
        foreach (Accumulator a in slots)
            CheckAcc(a);
        CheckBucket(bucket);
        return EvaluateRawUnchecked(slots, bucket);
    }

    [SkipLocalsInit]
    internal int EvaluateRawUnchecked(ReadOnlySpan<Accumulator> slots, int bucket = 0)
    {
        short** p = stackalloc short*[slots.Length];
        for (int i = 0; i < slots.Length; i++)
            p[i] = slots[i].Pointer;
        int raw = _functions.Evaluate(_context, p, bucket);
        for (int i = 0; i < slots.Length; i++)
            GC.KeepAlive(slots[i].Owner);
        GC.KeepAlive(_context);
        return raw;
    }

    // ================================================================== convenience
    /// <summary>Evaluates from feature lists, one list per slot. Allocates; for tools and tests, not for search.</summary>
    public int EvaluateFeatures(IReadOnlyList<int[]> featuresPerSlot, int bucket = 0)
    {
        CheckSlotCount(featuresPerSlot.Count);
        using AccumulatorBuffer buffer = Model.CreateAccumulators(_slotCount);
        Accumulator[] slots = new Accumulator[_slotCount];
        for (int i = 0; i < _slotCount; i++)
        { slots[i] = buffer[i]; RefreshAccumulator(slots[i], featuresPerSlot[i]); }
        return Evaluate(slots, bucket);
    }

    /// <summary>Int8 models: fraction of non-zero 4-byte chunks in the input of the first dense layer (diagnostic, allocates).
    /// The sparse kernel does work proportional to this number.</summary>
    public double GetNonZeroInputFraction(ReadOnlySpan<Accumulator> slots)
    {
        if (!_context.IsInt8)
            throw new InvalidOperationException("Int8 models only.");
        CheckSlotCount(slots.Length);
        foreach (Accumulator a in slots)
            CheckAcc(a);
        byte[] ft = new byte[_context.TransformerOutputSize];
        int h = _accumulatorSize, half = h >> 1;
        fixed (byte* f = ft)
        {
            for (int s = 0; s < _slotCount; s++)
            {
                short* accumulator = slots[s].Pointer;
                switch (_context.Activation)
                {
                    case FeatureTransformerActivation.ClippedReLU:
                        ScalarKernels.ApplyCReLU(accumulator, f + (s * h), h);
                        break;
                    case FeatureTransformerActivation.SquaredClippedReLU:
                        ScalarKernels.ApplySCReLU(accumulator, f + (s * h), h);
                        break;
                    case FeatureTransformerActivation.MaxPooling:
                        ScalarKernels.ApplyMaxPooling(accumulator, f + (s * half), half);
                        break;
                    case FeatureTransformerActivation.AveragePooling:
                        ScalarKernels.ApplyAveragePooling(accumulator, f + (s * half), half);
                        break;
                    default:
                        ScalarKernels.ApplyPairwise(accumulator, f + (s * half), half);
                        break;
                }
            }
        }
        int nonZero = 0;
        for (int i = 0; i < ft.Length; i += 4)
            if ((ft[i] | ft[i + 1] | ft[i + 2] | ft[i + 3]) != 0)
                nonZero++;
        foreach (Accumulator a in slots)
            GC.KeepAlive(a.Owner);
        return nonZero / (double)(ft.Length / 4);
    }
}
