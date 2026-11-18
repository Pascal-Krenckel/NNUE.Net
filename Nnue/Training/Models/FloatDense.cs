namespace Nnue.Training.Models;
/// <summary>One float dense layer. Weights are row-major per output: Weights[o * InputLength + i].</summary>
public sealed class FloatDense
{
    /// <summary>Number of inputs.</summary>
    public int InputLength { get; }
    /// <summary>Number of outputs.</summary>
    public int OutputLength { get; }
    /// <summary>Weights, row-major per output: <c>Weights[o * InputLength + i]</c>.</summary>
    public float[] Weights { get; }
    /// <summary>One bias per output.</summary>
    public float[] Biases { get; }

    /// <summary>Creates a zeroed layer.</summary>
    public FloatDense(int inputLength, int outputLength)
    {
        InputLength = inputLength;
        OutputLength = outputLength;
        Weights = new float[inputLength * outputLength];
        Biases = new float[outputLength];
    }
}
