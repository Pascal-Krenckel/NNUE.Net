using Nnue.Training.Quantization;

namespace Nnue.Training.Optimization;
/// <summary>Hyperparameters of <see cref="NnueTrainer"/> (Adam optimizer).</summary>
public sealed class TrainerOptions
{
    /// <summary>Samples per gradient step.</summary>
    public int BatchSize { get; set; } = 16384;
    /// <summary>Adam step size; multiplied by <see cref="LearningRateDecay"/> after every epoch of <see cref="NnueTrainer.Train"/>.</summary>
    public float LearningRate { get; set; } = 1e-3f;
    /// <summary>Multiplied into the learning rate after every epoch in <see cref="NnueTrainer.Train"/>.</summary>
    public float LearningRateDecay { get; set; } = 1f;
    /// <summary>The loss to minimise.</summary>
    public LossFunction Loss { get; set; } = LossFunction.SigmoidMse;
    /// <summary>Adam decay rate of the first moment.</summary>
    public float Beta1 { get; set; } = 0.9f;
    /// <summary>Adam decay rate of the second moment.</summary>
    public float Beta2 { get; set; } = 0.999f;
    /// <summary>Adam denominator term against division by zero.</summary>
    public float Epsilon { get; set; } = 1e-8f;
    /// <summary>Feature transformer weights/biases are clamped to +-this after each update (keeps int16 accumulators safe).</summary>
    public float FeatureWeightClip { get; set; } = 2f;
    /// <summary>Dense layer weight clamp. null (default) = exactly the range the quantized format can represent
    /// (<see cref="NnueQuantizer.GetMaxWeight"/>: +-127/64 for Int8), so training can never leave it.</summary>
    public float? DenseWeightClip { get; set; }
    /// <summary>Worker threads per batch (each keeps its own gradient buffer).</summary>
    public int Threads { get; set; } = Environment.ProcessorCount;
    /// <summary>Seed of the shuffling between epochs.</summary>
    public int Seed { get; set; } = 1;

    internal TrainerOptions Clone() => (TrainerOptions)MemberwiseClone();

    /// <summary>Throws <see cref="ArgumentException"/> for values the optimizer cannot work with.</summary>
    internal void Validate()
    {
        static bool Positive(float x) => float.IsFinite(x) && x > 0f;
        if (BatchSize <= 0)
            throw new ArgumentException("BatchSize must be > 0.", nameof(BatchSize));
        if (!Positive(LearningRate))
            throw new ArgumentException("LearningRate must be a positive number.", nameof(LearningRate));
        if (!Positive(LearningRateDecay))
            throw new ArgumentException("LearningRateDecay must be a positive number.", nameof(LearningRateDecay));
        if (Beta1 is not (>= 0f and < 1f))
            throw new ArgumentException("Beta1 must be in [0, 1).", nameof(Beta1));
        if (Beta2 is not (>= 0f and < 1f))
            throw new ArgumentException("Beta2 must be in [0, 1).", nameof(Beta2));
        if (!Positive(Epsilon))
            throw new ArgumentException("Epsilon must be a positive number.", nameof(Epsilon));
        if (!Positive(FeatureWeightClip))
            throw new ArgumentException("FeatureWeightClip must be a positive number.", nameof(FeatureWeightClip));
        if (DenseWeightClip is { } clip && !Positive(clip))
            throw new ArgumentException("DenseWeightClip must be a positive number or null.", nameof(DenseWeightClip));
        if (!Enum.IsDefined(Loss))
            throw new ArgumentException($"Unknown loss function {(int)Loss}.", nameof(Loss));
    }
}
