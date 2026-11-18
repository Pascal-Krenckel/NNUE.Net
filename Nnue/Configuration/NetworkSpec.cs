using System.Collections.Immutable;

namespace Nnue.Configuration;
/// <summary>
/// The declared architecture of a network. The library knows nothing about what the features mean: it has a shared
/// feature transformer that produces accumulators of <see cref="AccumulatorSize"/> values from sparse binary features, and a dense
/// head that consumes <see cref="Slots"/> accumulators in the order the caller passes them.
///
///   features --(shared weights, int16)--> Slots accumulators --FeatureTransformerActivation--> concat --> [dense layers, per output bucket] --> 1 value
///
/// What the slots are (one per side, per colour, per time step, ...) is decided by the abstraction on top of the library.
/// The dense part exists <see cref="OutputBuckets"/> times; one copy is selected per evaluation.
/// Float domain: hidden activations are in [0, 1]; the output o is in units of <see cref="OutputScale"/> score units.
/// Int8 domain (Stockfish): activations are uint8 with 1.0 = 127, weights int8 with 1.0 = 64, biases int32 with 1.0 = 127*64.
/// </summary>
public sealed record NetworkSpec
{
    /// <summary>Int8 activations: the uint8 value that represents 1.0.</summary>
    public const int Int8One = 127;
    /// <summary>Int8 weights: right shift that turns a dense layer sum back into activation units (log2 of <see cref="Int8WeightScale"/>).</summary>
    public const int Int8WeightShift = 6;
    /// <summary>Int8 weights: the int8 value that represents 1.0.</summary>
    public const int Int8WeightScale = 1 << Int8WeightShift;   // 64
    /// <summary>Largest |int8 weight| the trainer and quantizer produce; keeps the 16-bit multiply-add from saturating.</summary>
    public const int Int8MaxWeight = 127;                       // |w_int| <= 127 keeps maddubs from saturating
    /// <summary>Largest hidden layer (outputs). The evaluation keeps its scratch memory on the stack, so the sizes are bounded.</summary>
    public const int MaxHiddenLayerSize = 4096;
    /// <summary>Largest total number of hidden activations over all layers (twice the outputs with <see cref="HiddenActivation.SqrClippedReLU"/>).</summary>
    public const int MaxTotalHiddenActivations = 16384;
    /// <summary>The square in <see cref="HiddenActivation.SqrClippedReLU"/> is scaled by 127/128 (Stockfish: shift by 2*6+7 instead of dividing by 127).</summary>
    public const float SqrScale = 127f / 128f;

    /// <summary>Number of binary input features. Features are plain indices in [0, FeatureCount).</summary>
    public int FeatureCount { get; init; } = 768;

    /// <summary>Width of one accumulator. Multiple of 32 (multiple of 64 for <see cref="FeatureTransformerActivation.Pairwise"/> and the pooling activations).</summary>
    public int AccumulatorSize { get; init; } = 128;

    /// <summary>Number of accumulators the head consumes (1..64). They all come from the same transformer weights.</summary>
    public int Slots { get; init; } = 1;

    /// <summary>Arithmetic after the feature transformer: one int16 output layer, or Stockfish-style int8 dense layers.</summary>
    public NetworkPrecision Precision { get; init; } = NetworkPrecision.Int16;
    /// <summary>How accumulators are turned into the input of the first dense / output layer.</summary>
    public FeatureTransformerActivation Activation { get; init; } = FeatureTransformerActivation.ClippedReLU;

    /// <summary>Sizes of the hidden dense layers before the 1-output layer. Int8 only; multiples of 16. Empty = activations feed the output layer directly.
    /// Immutable, so a validated spec cannot change under a model built from it.</summary>
    public ImmutableArray<int> HiddenLayerSizes { get; init => field = value.IsDefault ? [] : value; } = [];

    /// <summary>Number of dense layers including the 1-output layer (Int8: hidden layers + 1; Int16: 0, its single output layer is not a dense Int8 layer).</summary>
    public int DenseLayerCount => Precision == NetworkPrecision.Int8 ? HiddenLayerSizes.Length + 1 : 0;

    /// <summary>Activation after every hidden dense layer (Int8 only).</summary>
    public HiddenActivation HiddenActivation { get; init; } = HiddenActivation.ClippedReLU;

    /// <summary>Every dense layer sees the concatenated activations of ALL earlier hidden layers (Stockfish's fc_2). Int8 only.</summary>
    public bool DenseConnections { get; init; }

    /// <summary>Stockfish skip connection: output += z0[n-2] - z0[n-1], the last two raw outputs of the first hidden layer. Needs Int8 and HiddenLayerSizes[0] >= 2.</summary>
    public bool SkipConnection { get; init; }

    /// <summary>Number of independent dense stacks (1..256) selected per evaluation ("layer stacks" in Stockfish).</summary>
    public int OutputBuckets { get; init; } = 1;

    private readonly int? _qa;
    /// <summary>Accumulator scale. Defaults: 255 (Int16), 127 (Int8 + ClippedReLU/MaxPooling/AveragePooling), 255 (Int8 + Pairwise/SquaredClippedReLU), the only values the uint8 conversion supports.</summary>
    public int ActivationQuantization { get => _qa ?? DefaultActivationQuantization; init => _qa = value; }

    private readonly int? _qb;
    /// <summary>Output layer weight scale (Int16 only; Int8 uses 64). Default 256, or 64 with SquaredClippedReLU (the int16 product c*w must fit).</summary>
    public int OutputWeightQuantization { get => _qb ?? DefaultOutputWeightQuantization; init => _qb = value; }

    /// <summary>Score units per unit of network output (the integer result of an evaluation is in score units).</summary>
    public int OutputScale { get; init; } = 400;

    /// <summary>The <see cref="ActivationQuantization"/> used when none is set: 127 for Int8 with clipped ReLU, 255 otherwise.</summary>
    public int DefaultActivationQuantization => Precision == NetworkPrecision.Int8 && Activation is FeatureTransformerActivation.ClippedReLU or FeatureTransformerActivation.MaxPooling or FeatureTransformerActivation.AveragePooling ? 127 : 255;

    /// <summary>True when the activation combines the two halves of an accumulator (pairwise product or pooling), so a slot contributes <c>AccumulatorSize / 2</c> inputs.</summary>
    internal bool HalvesAccumulator => Activation is FeatureTransformerActivation.Pairwise or FeatureTransformerActivation.MaxPooling or FeatureTransformerActivation.AveragePooling;
    /// <summary>The <see cref="OutputWeightQuantization"/> used when none is set: 64 for squared clipped ReLU, 256 otherwise.</summary>
    public int DefaultOutputWeightQuantization => Activation == FeatureTransformerActivation.SquaredClippedReLU ? 64 : 256;

    /// <summary>Output dimension rounded up to a multiple of 16 (storage / SIMD block size of the Int8 layout).</summary>
    public static int GetPaddedOutputLength(int outputLength) => (outputLength + 15) & ~15;

    /// <summary>
    /// The topology of current Stockfish: pairwise FT, [sqr|clip] concat after every hidden layer, the last layer sees both layers,
    /// skip connection, layer stacks. Stockfish evaluates two perspectives, hence Slots = 2 (the library itself does not care).
    /// </summary>
    public static NetworkSpec CreateStockfishLike(int featureCount, int accumulatorSize = 1024, int secondLayerSize = 32, int thirdLayerSize = 32, int outputBuckets = 32) => new()
    {
        FeatureCount = featureCount,
        AccumulatorSize = accumulatorSize,
        Slots = 2,
        Precision = NetworkPrecision.Int8,
        Activation = FeatureTransformerActivation.Pairwise,
        HiddenLayerSizes = [secondLayerSize, thirdLayerSize],
        HiddenActivation = HiddenActivation.SqrClippedReLU,
        DenseConnections = true,
        SkipConnection = true,
        OutputBuckets = outputBuckets,
    };

    /// <summary>
    /// Preset: the classic "perspective" network with one int16 output layer, <c>(features -> accumulator) x slots -> output</c>.
    /// Most accurate and simplest; use <paramref name="activation"/> = SquaredClippedReLU for the stronger SCReLU variant.
    /// </summary>
    public static NetworkSpec CreateSimple(int featureCount, int accumulatorSize, int slots = 2, int outputBuckets = 1,
        FeatureTransformerActivation activation = FeatureTransformerActivation.ClippedReLU)
        => CreateBuilder(featureCount).WithAccumulatorSize(accumulatorSize).WithSlots(slots).WithOutputBuckets(outputBuckets)
            .WithInt16(activation).Build();

    /// <summary>
    /// Preset: Stockfish arithmetic (uint8 activations, int8 weights) with the given hidden dense layers, e.g. <c>[16, 32]</c>,
    /// clipped ReLU everywhere. Hidden sizes must be multiples of 16, <c>slots * accumulatorSize</c> a multiple of 64.
    /// </summary>
    public static NetworkSpec CreateInt8(int featureCount, int accumulatorSize, ReadOnlySpan<int> hiddenLayerSizes, int slots = 2, int outputBuckets = 1,
        FeatureTransformerActivation activation = FeatureTransformerActivation.ClippedReLU)
        => CreateBuilder(featureCount).WithAccumulatorSize(accumulatorSize).WithSlots(slots).WithOutputBuckets(outputBuckets)
            .WithInt8(activation, hiddenLayerSizes).Build();

    /// <summary>Starts a fluent <see cref="NetworkSpecBuilder"/>; its <see cref="NetworkSpecBuilder.Build"/> validates the result.</summary>
    public static NetworkSpecBuilder CreateBuilder(int featureCount) => new(featureCount);

    /// <summary>Like <see cref="Validate"/>, but reports the problem instead of throwing.</summary>
    public bool TryValidate([System.Diagnostics.CodeAnalysis.NotNullWhen(false)] out string? error)
    {
        try
        { Validate(); error = null; return true; }
        catch (ArgumentException e) { error = e.Message; return false; }
    }

    /// <summary>Throws <see cref="ArgumentException"/> if the combination is not supported.</summary>
    public void Validate()
    {
        // enum values come from files too: an undefined value would silently take a default branch somewhere
        if (!Enum.IsDefined(Precision))
            throw new ArgumentException($"Unknown Precision {(int)Precision}.");
        if (!Enum.IsDefined(Activation))
            throw new ArgumentException($"Unknown Activation {(int)Activation}.");
        if (!Enum.IsDefined(HiddenActivation))
            throw new ArgumentException($"Unknown HiddenActivation {(int)HiddenActivation}.");
        if (FeatureCount <= 0)
            throw new ArgumentException("FeatureCount must be > 0.");
        if (AccumulatorSize <= 0 || AccumulatorSize % 32 != 0)
            throw new ArgumentException("AccumulatorSize must be a positive multiple of 32.");
        if (Slots is < 1 or > 64)
            throw new ArgumentException("Slots must be in [1, 64].");
        if (HalvesAccumulator && AccumulatorSize % 64 != 0)
            throw new ArgumentException($"{Activation} needs AccumulatorSize to be a multiple of 64.");
        if (OutputScale <= 0)
            throw new ArgumentException("OutputScale must be > 0.");
        if ((long)FeatureCount * AccumulatorSize > int.MaxValue / 2)
            throw new ArgumentException("Feature transformer too large.");
        if (OutputBuckets is < 1 or > 256)
            throw new ArgumentException("OutputBuckets must be in [1, 256].");

        int slotOut = HalvesAccumulator ? AccumulatorSize / 2 : AccumulatorSize;
        int ftOut = Slots * slotOut;
        if (ftOut > 16384)
            throw new ArgumentException("Slots * (accumulator output size) must not exceed 16384.");

        if (Precision == NetworkPrecision.Int16)
        {
            if (HiddenLayerSizes.Length != 0)
                throw new ArgumentException("Hidden dense layers need Precision = Int8.");
            if (HalvesAccumulator)
                throw new ArgumentException($"{Activation} needs Precision = Int8.");
            if (HiddenActivation != HiddenActivation.ClippedReLU || DenseConnections || SkipConnection)
                throw new ArgumentException("Hidden activation, dense connections and skip connection need Precision = Int8.");
            if (ActivationQuantization is < 1 or > 255)
                throw new ArgumentException("ActivationQuantization must be in [1, 255].");
            if (OutputWeightQuantization is < 1 or > 1024)
                throw new ArgumentException("OutputWeightQuantization must be in [1, 1024].");
            if (Activation == FeatureTransformerActivation.SquaredClippedReLU && 32767 / ActivationQuantization / (double)OutputWeightQuantization < 0.25)
                throw new ArgumentException("SquaredClippedReLU needs a smaller OutputWeightQuantization (default 64): the int16 product c*w limits |w_int| to 32767/ActivationQuantization.");
        }
        else
        {
            if (ActivationQuantization != DefaultActivationQuantization)
                throw new ArgumentException($"Int8 + {Activation} requires ActivationQuantization = {DefaultActivationQuantization}.");
            foreach (int n in HiddenLayerSizes)
                if (n <= 0 || n % 16 != 0 || n > MaxHiddenLayerSize)
                    throw new ArgumentException($"Hidden layer sizes must be positive multiples of 16, at most {MaxHiddenLayerSize}.");
            if (HiddenLayerSizes.Length > 64)
                throw new ArgumentException("At most 64 hidden layers.");
            long activationBytes = 0;
            foreach (int n in HiddenLayerSizes)
                activationBytes += HiddenActivation == HiddenActivation.SqrClippedReLU ? 2L * n : n;
            if (activationBytes > MaxTotalHiddenActivations)
                throw new ArgumentException($"The hidden layers produce {activationBytes} activations, at most {MaxTotalHiddenActivations} are supported (an evaluation keeps them on the stack).");
            if (ftOut % 64 != 0)
                throw new ArgumentException("Int8: Slots * (accumulator output size) must be a multiple of 64 (sparse input kernels).");
            if (SkipConnection && (HiddenLayerSizes.Length < 1 || HiddenLayerSizes[0] < 2))
                throw new ArgumentException("SkipConnection needs a first hidden layer with at least 2 outputs.");
        }
    }

    // ---- serialization (used by the model and checkpoint formats) ----
    /// <summary>Writes the spec in the binary form used inside model and checkpoint files.</summary>
    public void Write(BinaryWriter w)
    {
        w.Write(FeatureCount);
        w.Write(AccumulatorSize);
        w.Write(Slots);
        w.Write((int)Precision);
        w.Write((int)Activation);
        w.Write((int)HiddenActivation);
        w.Write(DenseConnections);
        w.Write(SkipConnection);
        w.Write(OutputBuckets);
        w.Write(ActivationQuantization);
        w.Write(OutputWeightQuantization);
        w.Write(OutputScale);
        w.Write(HiddenLayerSizes.Length);
        foreach (int n in HiddenLayerSizes)
            w.Write(n);
    }

    /// <summary>Reads a spec written by <see cref="Write"/> and validates it.</summary>
    public static NetworkSpec Read(BinaryReader r)
    {
        int features = r.ReadInt32(), h = r.ReadInt32(), slots = r.ReadInt32();
        NetworkPrecision precision = (NetworkPrecision)r.ReadInt32();
        FeatureTransformerActivation act = (FeatureTransformerActivation)r.ReadInt32();
        HiddenActivation hidden = (HiddenActivation)r.ReadInt32();
        bool dense = r.ReadBoolean(), skip = r.ReadBoolean();
        int buckets = r.ReadInt32(), qa = r.ReadInt32(), qb = r.ReadInt32(), scale = r.ReadInt32();
        int count = r.ReadInt32();
        if (count is < 0 or > 64)
            throw new InvalidDataException("Bad layer count.");
        int[] layers = new int[count];
        for (int i = 0; i < count; i++)
            layers[i] = r.ReadInt32();
        NetworkSpec spec = new()
        {
            FeatureCount = features,
            AccumulatorSize = h,
            Slots = slots,
            Precision = precision,
            Activation = act,
            HiddenActivation = hidden,
            DenseConnections = dense,
            SkipConnection = skip,
            OutputBuckets = buckets,
            ActivationQuantization = qa,
            OutputWeightQuantization = qb,
            OutputScale = scale,
            HiddenLayerSizes = [.. layers],
        };
        spec.Validate();
        return spec;
    }

    // ---- value semantics: two specs are equal if they describe the same network (effective quantization values, layer sizes by content)
    /// <summary>True if both specs describe the same network: same options, same layer sizes, same effective quantization.</summary>
    public bool Equals(NetworkSpec? other) =>
        other is not null && FeatureCount == other.FeatureCount && AccumulatorSize == other.AccumulatorSize && Slots == other.Slots
        && Precision == other.Precision && Activation == other.Activation && HiddenActivation == other.HiddenActivation
        && HiddenLayerSizes.AsSpan().SequenceEqual(other.HiddenLayerSizes.AsSpan())
        && DenseConnections == other.DenseConnections && SkipConnection == other.SkipConnection && OutputBuckets == other.OutputBuckets
        && ActivationQuantization == other.ActivationQuantization && OutputWeightQuantization == other.OutputWeightQuantization && OutputScale == other.OutputScale;

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        HashCode hash = new();
        hash.Add(FeatureCount);
        hash.Add(AccumulatorSize);
        hash.Add(Slots);
        hash.Add(Precision);
        hash.Add(Activation);
        hash.Add(HiddenActivation);
        foreach (int size in HiddenLayerSizes)
            hash.Add(size);
        hash.Add(DenseConnections);
        hash.Add(SkipConnection);
        hash.Add(OutputBuckets);
        hash.Add(ActivationQuantization);
        hash.Add(OutputWeightQuantization);
        hash.Add(OutputScale);
        return hash.ToHashCode();
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"NetworkSpec {{ FeatureCount = {FeatureCount}, AccumulatorSize = {AccumulatorSize}, Slots = {Slots}, Precision = {Precision}, Activation = {Activation}, " +
        $"HiddenLayerSizes = [{string.Join(", ", HiddenLayerSizes)}], HiddenActivation = {HiddenActivation}, DenseConnections = {DenseConnections}, " +
        $"SkipConnection = {SkipConnection}, OutputBuckets = {OutputBuckets}, ActivationQuantization = {ActivationQuantization}, " +
        $"OutputWeightQuantization = {OutputWeightQuantization}, OutputScale = {OutputScale} }}";
}
