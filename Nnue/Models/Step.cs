namespace Nnue.Models;
/// <summary>Resolved pointers and dimensions of one dense layer of one bucket (built once per model).</summary>
internal unsafe struct Step
{
    public sbyte* Weights;
    public int* Biases;
    public int InputLength, PaddedOutputLength, OutputLength, InputOffset, ActivationOffset;
}
