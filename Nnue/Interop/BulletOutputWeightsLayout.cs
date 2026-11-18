namespace Nnue.Interop;

/// <summary>How the output layer weights of several output buckets are ordered in a bullet <c>quantised.bin</c>.</summary>
public enum BulletOutputWeightsLayout
{
    /// <summary>[bucket][input]: all weights of one bucket are contiguous (bullet's <c>l1w</c> saved with <c>.transpose()</c>). Single-bucket networks need no choice.</summary>
    BucketMajor = 0,
    /// <summary>[input][bucket]: the weights of all buckets for one input are contiguous (bullet's <c>l1w</c> saved as is).</summary>
    InputMajor = 1,
}
