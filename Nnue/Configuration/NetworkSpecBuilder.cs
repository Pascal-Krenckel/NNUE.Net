namespace Nnue.Configuration;

/// <summary>
/// Fluent construction of a <see cref="NetworkSpec"/>. Every method documents the choice it makes; <see cref="Build"/> validates the combination
/// and throws an <see cref="ArgumentException"/> that names the conflicting options.
/// <code>
/// var spec = NetworkSpec.CreateBuilder(768).WithAccumulatorSize(256).WithSlots(2).WithOutputBuckets(8)
///                       .WithInt8(FeatureTransformerActivation.ClippedReLU, [16, 32]).Build();
/// </code>
/// </summary>
public sealed class NetworkSpecBuilder
{
    private int _accumulatorSize = 128, _slots = 1, _outputBuckets = 1, _outputScale = 400;
    private NetworkPrecision _precision = NetworkPrecision.Int16;
    private FeatureTransformerActivation _activation = FeatureTransformerActivation.ClippedReLU;
    private HiddenActivation _hiddenActivation = HiddenActivation.ClippedReLU;
    private System.Collections.Immutable.ImmutableArray<int> _hiddenLayerSizes = [];
    private bool _denseConnections, _skipConnection;
    private int? _activationQuantization, _outputWeightQuantization;

    private readonly int _featureCount;

    internal NetworkSpecBuilder(int featureCount) => _featureCount = featureCount;

    /// <summary>Number of values per accumulator (a multiple of 32; of 64 for pairwise activation).</summary>
    public NetworkSpecBuilder WithAccumulatorSize(int accumulatorSize) { _accumulatorSize = accumulatorSize; return this; }

    /// <summary>How many accumulators one evaluation consumes, in the order the caller passes them (1 to 64).</summary>
    public NetworkSpecBuilder WithSlots(int slots) { _slots = slots; return this; }

    /// <summary>Independent dense stacks (e.g. by game phase); one is selected per evaluation.</summary>
    public NetworkSpecBuilder WithOutputBuckets(int outputBuckets) { _outputBuckets = outputBuckets; return this; }

    /// <summary>Score units per unit of network output.</summary>
    public NetworkSpecBuilder WithOutputScale(int outputScale) { _outputScale = outputScale; return this; }

    /// <summary>Single int16 output layer (no hidden layers).</summary>
    public NetworkSpecBuilder WithInt16(FeatureTransformerActivation activation = FeatureTransformerActivation.ClippedReLU)
    {
        _precision = NetworkPrecision.Int16;
        _activation = activation;
        _hiddenLayerSizes = [];
        _hiddenActivation = HiddenActivation.ClippedReLU;
        _denseConnections = _skipConnection = false;
        return this;
    }

    /// <summary>Stockfish arithmetic with the given hidden dense layer sizes (multiples of 16; may be empty for a single int8 output layer).</summary>
    public NetworkSpecBuilder WithInt8(FeatureTransformerActivation activation, ReadOnlySpan<int> hiddenLayerSizes)
    {
        _precision = NetworkPrecision.Int8;
        _activation = activation;
        _hiddenLayerSizes = [.. hiddenLayerSizes];
        return this;
    }

    /// <summary>Stockfish's <c>[sqr(z) | clip(z)]</c> concatenation after every hidden layer instead of plain clipped ReLU (Int8 only).</summary>
    public NetworkSpecBuilder WithSqrClippedHiddenActivation() { _hiddenActivation = HiddenActivation.SqrClippedReLU; return this; }

    /// <summary>Later dense layers see the activations of all earlier layers (Int8 only).</summary>
    public NetworkSpecBuilder WithDenseConnections(bool enabled = true) { _denseConnections = enabled; return this; }

    /// <summary>Stockfish skip connection: <c>output += z0[n-2] - z0[n-1]</c> of the first hidden layer (Int8 only).</summary>
    public NetworkSpecBuilder WithSkipConnection(bool enabled = true) { _skipConnection = enabled; return this; }

    /// <summary>Overrides the default quantization scales (only needed to import externally trained weights).</summary>
    public NetworkSpecBuilder WithQuantization(int activationQuantization, int outputWeightQuantization)
    {
        _activationQuantization = activationQuantization;
        _outputWeightQuantization = outputWeightQuantization;
        return this;
    }

    /// <summary>Creates the spec and validates it.</summary>
    /// <exception cref="ArgumentException">The options do not form a supported network.</exception>
    public NetworkSpec Build()
    {
        NetworkSpec spec = new()
        {
            FeatureCount = _featureCount,
            AccumulatorSize = _accumulatorSize,
            Slots = _slots,
            Precision = _precision,
            Activation = _activation,
            HiddenLayerSizes = _hiddenLayerSizes,
            HiddenActivation = _hiddenActivation,
            DenseConnections = _denseConnections,
            SkipConnection = _skipConnection,
            OutputBuckets = _outputBuckets,
            OutputScale = _outputScale,
        };
        if (_activationQuantization is { } qa)
            spec = spec with { ActivationQuantization = qa };
        if (_outputWeightQuantization is { } qb)
            spec = spec with { OutputWeightQuantization = qb };
        spec.Validate();
        return spec;
    }
}
