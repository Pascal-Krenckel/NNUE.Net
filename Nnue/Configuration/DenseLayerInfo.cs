namespace Nnue.Configuration;
/// <summary>Resolved shape of one dense layer.</summary>
/// <param name="InputLength">number of inputs</param>
/// <param name="OutputLength">number of outputs (1 for the last layer)</param>
/// <param name="PaddedOutputLength">OutputLength rounded up to a multiple of 16 (Int8 storage)</param>
/// <param name="InputOffset">offset of the input inside the activation buffer (unused for layer 0, which reads the feature transformer output)</param>
/// <param name="ActivationOffset">offset of this layer's activation block inside the activation buffer (unused for the last layer)</param>
internal readonly record struct DenseLayerInfo(int InputLength, int OutputLength, int PaddedOutputLength, int InputOffset, int ActivationOffset);
