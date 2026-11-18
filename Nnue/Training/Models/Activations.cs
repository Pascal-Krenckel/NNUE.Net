using Nnue.Configuration;


namespace Nnue.Training.Models;

/// <summary>The float activation functions (what the trainer optimizes; the integer kernels approximate them).</summary>
public static class Activations
{
    /// <summary>Clipped ReLU: clamp(x, 0, 1).</summary>
    public static float ClippedReLU(float x) => x < 0f ? 0f : x > 1f ? 1f : x;

    /// <summary>d ClippedReLU / dx</summary>
    public static float ClippedReLUDerivative(float x) => x is > 0f and < 1f ? 1f : 0f;

    /// <summary>Squared clipped ReLU on the feature transformer: clamp(x, 0, 1)^2.</summary>
    public static float SquaredClippedReLU(float x) { float c = ClippedReLU(x); return c * c; }

    /// <summary>d SquaredClippedReLU / dx</summary>
    public static float SquaredClippedReLUDerivative(float x) => x is > 0f and < 1f ? 2f * x : 0f;

    /// <summary>Stockfish SqrClippedReLU: min(x^2 * 127/128, 1). The square is taken of the unclipped x.</summary>
    public static float SqrClippedReLU(float x) => MathF.Min(x * x * NetworkSpec.SqrScale, 1f);

    /// <summary>d SqrClippedReLU / dx (zero once saturated).</summary>
    public static float SqrClippedReLUDerivative(float x) => x * x * NetworkSpec.SqrScale < 1f ? 2f * x * NetworkSpec.SqrScale : 0f;
}
