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

    /// <summary>
    /// Quantmoid4, the integer-friendly approximation of sigmoid(4x): with v = 1 - min(|x|, 1) and q = 127/256 * v^2 the result is
    /// q for x &lt; 0 and 126/127 - q otherwise (range [0, 126/127], 0.496 at x = 0).
    /// </summary>
    public static float Quantmoid4(float x)
    {
        float v = 1f - MathF.Min(MathF.Abs(x), 1f);
        float q = 127f / 256f * v * v;
        return x < 0f ? q : 126f / 127f - q;
    }

    /// <summary>d Quantmoid4 / dx = 127/128 * (1 - min(|x|, 1)); continuous, zero once saturated.</summary>
    public static float Quantmoid4Derivative(float x) => 127f / 128f * (1f - MathF.Min(MathF.Abs(x), 1f));

    /// <summary>Max pooling of two clipped values: max(ClippedReLU(a), ClippedReLU(b)).</summary>
    public static float MaxPool(float a, float b) => MathF.Max(ClippedReLU(a), ClippedReLU(b));

    /// <summary>Average pooling of two clipped values: (ClippedReLU(a) + ClippedReLU(b)) / 2.</summary>
    public static float AveragePool(float a, float b) => 0.5f * (ClippedReLU(a) + ClippedReLU(b));
}
