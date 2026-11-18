namespace Nnue.Configuration;
/// <summary>
/// Everything derived from a <see cref="NetworkSpec"/>: sizes, activation buffer offsets and the byte offsets of every tensor inside the
/// model blob. Computed once. All tensors start at multiples of 64 bytes (measured: 64-byte aligned loads make the kernels 15-25 % faster).
/// </summary>
internal sealed class NetworkLayout
{
    public NetworkSpec Spec { get; }
    public int FeatureCount { get; }
    public int AccumulatorSize { get; }
    public int Slots { get; }
    /// <summary>values one slot contributes to the head input (H for CReLU/SCReLU, H/2 for pairwise)</summary>
    public int SlotOutputSize { get; }
    /// <summary>input size of the first dense layer / of the output layer</summary>
    public int FeatureTransformerOutputSize { get; }
    public IReadOnlyList<DenseLayerInfo> DenseLayers { get; }
    public int TotalActivationSize { get; }
    public int MaxPaddedOutputLength { get; }
    /// <summary>4-byte chunks of the feature transformer output</summary>
    public int ChunkCount => FeatureTransformerOutputSize >> 2;

    // ---- blob layout (bytes) ----
    public long FeatureWeightsOffset { get; }
    public long FeatureBiasOffset { get; }
    public long OutputWeightsOffset { get; }          // Int16: [bucket][Slots * H] shorts
    public long OutputWeightsBucketStride { get; }    // bytes
    public long OutputBiasOffset { get; }             // Int16: [bucket] ints
    public long BlobSize { get; }
    private readonly long[] _denseWeights, _denseBias;   // [bucket * layers + layer]

    public long GetDenseWeightsOffset(int bucket, int layer) => _denseWeights[(bucket * DenseLayers.Count) + layer];
    public long GetDenseBiasOffset(int bucket, int layer) => _denseBias[(bucket * DenseLayers.Count) + layer];

    /// <summary>
    /// Index of weight (input i, output o) of a dense layer. Hidden layers use a blocked layout: 4 inputs per output are contiguous, and
    /// the weights of all outputs for one group of 4 inputs follow each other, so one SIMD load serves several outputs.
    /// The LAST layer has a single output and is stored as a plain contiguous vector of its <c>In</c> weights (o must be 0): the kernel is a
    /// dot product, not a matrix product. The bytes after the weights (up to the next multiple of 64) are zero.
    /// </summary>
    public int GetDenseWeightIndex(int layer, int i, int o) => layer == DenseLayers.Count - 1
            ? o != 0 ? throw new ArgumentOutOfRangeException(nameof(o), "The last layer has one output.") : i
            : ((i >> 2) * DenseLayers[layer].PaddedOutputLength * 4) + (o * 4) + (i & 3);

    /// <summary>Number of weight bytes the caller writes for a dense layer (the last layer: In, the others: In * PaddedOut).</summary>
    public int GetDenseWeightCount(int layer) => layer == DenseLayers.Count - 1 ? DenseLayers[layer].InputLength : DenseLayers[layer].InputLength * DenseLayers[layer].PaddedOutputLength;

    private static long Align(long x) => (x + 63) & ~63L;

    private NetworkLayout(NetworkSpec spec)
    {
        Spec = spec;
        FeatureCount = spec.FeatureCount;
        AccumulatorSize = spec.AccumulatorSize;
        Slots = spec.Slots;
        SlotOutputSize = spec.Activation == FeatureTransformerActivation.Pairwise ? AccumulatorSize / 2 : AccumulatorSize;
        FeatureTransformerOutputSize = Slots * SlotOutputSize;

        int mult = spec.HiddenActivation == HiddenActivation.SqrClippedReLU ? 2 : 1;
        int hidden = spec.HiddenLayerSizes.Length;
        int[] actSizes = spec.HiddenLayerSizes.Select(n => n * mult).ToArray();
        int[] actOffsets = new int[hidden];
        int total = 0;
        for (int i = 0; i < hidden; i++)
        { actOffsets[i] = total; total += actSizes[i]; }
        TotalActivationSize = total;

        DenseLayerInfo[] dense = new DenseLayerInfo[hidden + 1];
        for (int k = 0; k <= hidden; k++)
        {
            int outDim = k < hidden ? spec.HiddenLayerSizes[k] : 1;
            int inDim, inOff = 0;
            if (k == 0)
                inDim = FeatureTransformerOutputSize;
            else if (spec.DenseConnections)
            { inDim = actOffsets[k - 1] + actSizes[k - 1]; inOff = 0; }
            else
            { inDim = actSizes[k - 1]; inOff = actOffsets[k - 1]; }
            dense[k] = new DenseLayerInfo(inDim, outDim, NetworkSpec.GetPaddedOutputLength(outDim), inOff, k < hidden ? actOffsets[k] : 0);
        }
        DenseLayers = dense;
        MaxPaddedOutputLength = dense.Max(d => d.PaddedOutputLength);

        long pos = 0;
        FeatureWeightsOffset = pos;
        pos = Align(pos + ((long)FeatureCount * AccumulatorSize * sizeof(short)));
        FeatureBiasOffset = pos;
        pos = Align(pos + ((long)AccumulatorSize * sizeof(short)));
        _denseWeights = new long[spec.OutputBuckets * dense.Length];
        _denseBias = new long[spec.OutputBuckets * dense.Length];
        if (spec.Precision == NetworkPrecision.Int16)
        {
            OutputWeightsOffset = pos;
            OutputWeightsBucketStride = Align((long)FeatureTransformerOutputSize * sizeof(short));
            pos += spec.OutputBuckets * OutputWeightsBucketStride;
            OutputBiasOffset = pos;
            pos = Align(pos + ((long)spec.OutputBuckets * sizeof(int)));
        }
        else
        {
            for (int b = 0; b < spec.OutputBuckets; b++)
                for (int k = 0; k < dense.Length; k++)
                {
                    _denseWeights[(b * dense.Length) + k] = pos;
                    pos = Align(pos + (k == dense.Length - 1 ? dense[k].InputLength : (long)dense[k].InputLength * dense[k].PaddedOutputLength));
                    _denseBias[(b * dense.Length) + k] = pos;
                    pos = Align(pos + ((long)dense[k].PaddedOutputLength * sizeof(int)));
                }
        }
        BlobSize = pos;
    }

    public static NetworkLayout Create(NetworkSpec spec)
    {
        spec.Validate();
        return new NetworkLayout(spec);
    }
}
