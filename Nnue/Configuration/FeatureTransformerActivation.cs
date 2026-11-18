namespace Nnue.Configuration;
/// <summary>How the int16 accumulators are turned into the input of the first dense layer / output layer.</summary>
public enum FeatureTransformerActivation
{
    /// <summary>clamp(a, 0, 1) of every slot, concatenated: Slots * AccumulatorSize inputs.</summary>
    ClippedReLU = 0,
    /// <summary>Stockfish style: clamp(a_j) * clamp(a_{j+H/2}) of the two halves of every slot: Slots * AccumulatorSize / 2 inputs (Int8 only).</summary>
    Pairwise = 1,
    /// <summary>Squared clipped ReLU, clamp(a, 0, 1)^2, every slot concatenated. Int16: out = sum((c*w)*c) with c = clamp(acc, 0, ActivationQuantization); Int8: u8 = (c*c) >> 9.</summary>
    SquaredClippedReLU = 2,
}
