namespace Nnue.Configuration;
/// <summary>Activation applied to the output of every hidden dense layer.</summary>
public enum HiddenActivation
{
    /// <summary>clamp(z, 0, 1): one value per output.</summary>
    ClippedReLU = 0,
    /// <summary>
    /// Stockfish: two values per output, concatenated [ sqr(z) | clip(z) ] with clip(z) = clamp(z, 0, 1) and
    /// sqr(z) = min(z^2 * 127/128, 1). The square is taken of the UNclipped z (negative z gives a positive value).
    /// Int8: sqr = min(127, z_int^2 >> 19), clip = clamp(z_int >> 6, 0, 127).
    /// </summary>
    SqrClippedReLU = 1,
}
