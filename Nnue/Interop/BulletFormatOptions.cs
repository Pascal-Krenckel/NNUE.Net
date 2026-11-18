namespace Nnue.Interop;

/// <summary>Variations of the raw, header-less binary file that bullet's <c>SavedFormat</c> list produces.</summary>
public sealed record BulletFormatOptions
{
    /// <summary>Order of the output weights when the network has more than one output bucket. Default: bucket-major.</summary>
    public BulletOutputWeightsLayout OutputWeightsLayout { get; init; } = BulletOutputWeightsLayout.BucketMajor;

    /// <summary>True if the output biases were saved as int32 (<c>quantise::&lt;i32&gt;</c>) instead of int16 (the usual choice).</summary>
    public bool OutputBiasesAreInt32 { get; init; }
}
