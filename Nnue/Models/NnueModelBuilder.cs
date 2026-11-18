using System.Runtime.CompilerServices;
using Nnue.Configuration;
using Nnue.Memory;

namespace Nnue.Models;

/// <summary>
/// Fills the weights of a new <see cref="NnueModel"/>. All spans are views into the final, 64-byte aligned native blob (no copy on <see cref="Build"/>);
/// they must not be used after <see cref="Build"/>.
/// </summary>
public sealed unsafe class NnueModelBuilder
{
    private readonly NativeBlock _blob;
    private readonly byte* _base;
    private bool _built;

    /// <summary>The architecture of the model being built.</summary>
    public NetworkSpec Spec { get; }
    internal NetworkLayout Layout { get; }

    /// <summary>Allocates zeroed weights for a model of <paramref name="spec"/>.</summary>
    /// <exception cref="ArgumentException">The spec is invalid or the model would be larger than 2 GB.</exception>
    public NnueModelBuilder(NetworkSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        Layout = NetworkLayout.Create(spec);
        Spec = spec;
        if (Layout.BlobSize > int.MaxValue - 64) throw new ArgumentException("Model larger than 2 GB is not supported.");
        _blob = new NativeBlock(Layout.BlobSize);
        _base = _blob.Pointer;
    }

    private void Check() { if (_built) throw new InvalidOperationException("The builder has already produced its model."); }

    /// <summary>The whole blob (used by the loader).</summary>
    internal Span<byte> RawBlob { get { Check(); return new Span<byte>(_base, (int)Layout.BlobSize); } }

    /// <summary>The whole blob as <see cref="Memory{T}"/> (used by the asynchronous loader).</summary>
    internal Memory<byte> RawBlobMemory { get { Check(); return new NativeMemoryManager(_base, (int)Layout.BlobSize).Memory; } }

    /// <summary>Frees the blob of a builder whose model will never be built (failed or cancelled load).</summary>
    internal void Abandon() { _built = true; _blob.Dispose(); }

    /// <summary>[FeatureCount * AccumulatorSize]; row f (= feature f) holds the AccumulatorSize weights of that feature.</summary>
    public Span<short> FeatureWeights { get { Check(); return new((short*)(_base + Layout.FeatureWeightsOffset), Layout.FeatureCount * Layout.AccumulatorSize); } }
    /// <summary>The feature transformer biases, one per accumulator value.</summary>
    public Span<short> FeatureBiases { get { Check(); return new((short*)(_base + Layout.FeatureBiasOffset), Layout.AccumulatorSize); } }

    /// <summary>Int16: output layer weights of one bucket, slot-major: [Slots * AccumulatorSize].</summary>
    public Span<short> GetOutputWeights(int bucket)
    {
        Check();
        if (Spec.Precision != NetworkPrecision.Int16) throw new InvalidOperationException("Int16 models only.");
        return new((short*)(_base + Layout.OutputWeightsOffset + bucket * Layout.OutputWeightsBucketStride), Layout.FeatureTransformerOutputSize);
    }

    /// <summary>Int16: output bias of every bucket (scale ActivationQuantization * OutputWeightQuantization, or ActivationQuantization^2 * OutputWeightQuantization for SCReLU).</summary>
    public Span<int> OutputBiases { get { Check(); return new((int*)(_base + Layout.OutputBiasOffset), Spec.OutputBuckets); } }

    /// <summary>Int8: the weights of dense layer <paramref name="layer"/> in storage order (blocked for SIMD, the last layer a contiguous vector).
    /// Filling them in order is only meaningful for random or copied data; use <see cref="SetDenseWeight"/> to set the weight of one input / output pair.</summary>
    public Span<sbyte> GetDenseWeights(int bucket, int layer)
    {
        Check();
        if (Spec.Precision != NetworkPrecision.Int8) throw new InvalidOperationException("Int8 models only.");
        var d = Layout.DenseLayers[layer];
        return new((sbyte*)(_base + Layout.GetDenseWeightsOffset(bucket, layer)), Layout.GetDenseWeightCount(layer));
    }

    /// <summary>Int8: sets the weight from input <paramref name="input"/> to output <paramref name="output"/> of dense layer <paramref name="layer"/> (scale 64 = 1.0).</summary>
    public void SetDenseWeight(int bucket, int layer, int input, int output, sbyte value)
    {
        NnueModel.CheckDenseIndex(Spec, Layout, bucket, layer, input, output);
        GetDenseWeights(bucket, layer)[Layout.GetDenseWeightIndex(layer, input, output)] = value;
    }

    /// <summary>Int8: biases (scale 127*64) of dense layer <paramref name="layer"/>, [PaddedOut] values.</summary>
    public Span<int> GetDenseBiases(int bucket, int layer)
    {
        Check();
        if (Spec.Precision != NetworkPrecision.Int8) throw new InvalidOperationException("Int8 models only.");
        return new((int*)(_base + Layout.GetDenseBiasOffset(bucket, layer)), Layout.DenseLayers[layer].PaddedOutputLength);
    }

    /// <summary>Creates the model from the filled weights (no copy). The builder cannot be used afterwards.</summary>
    public NnueModel Build()
    {
        Check();
        _built = true;
        return new NnueModel(Spec, Layout, _blob);
    }
}
