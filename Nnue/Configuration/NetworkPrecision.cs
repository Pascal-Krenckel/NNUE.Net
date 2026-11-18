namespace Nnue.Configuration;
/// <summary>Arithmetic used after the feature transformer.</summary>
public enum NetworkPrecision
{
    /// <summary>int16 output layer (single layer, no hidden dense layers). Highest accuracy, simplest.</summary>
    Int16 = 0,
    /// <summary>Stockfish style: uint8 activations (1.0 = 127), int8 weights (scale 64), int32 biases, any number of dense layers.</summary>
    Int8 = 1,
}
