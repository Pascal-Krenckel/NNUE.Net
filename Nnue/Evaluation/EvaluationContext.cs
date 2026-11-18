using Nnue.Configuration;
using Nnue.Models;

namespace Nnue.Evaluation;
/// <summary>Everything the hot path needs, resolved once per evaluator: raw pointers into the model blob and plain integers (no spec lookups, no LINQ).</summary>
internal sealed unsafe class EvaluationContext
{
    public NnueModel Model = null!;
    public short* FeatureWeights, FeatureBias;
    public int AccumulatorSize, SlotCount, FeatureCount, TransformerOutputSize, ChunkCount, TotalActivationSize, MaxPaddedOutputLength, LayerCount, BucketCount, ClipMax, OutputWeightsStride;
    public bool IsInt8, UsesSqrHidden, HasSkipConnection;
    public FeatureTransformerActivation Activation;
    public Step* Steps;
    public short* OutputWeights;
    public int* OutputBiases;
    public SparseInputMode SparseMode;
    public double Scale;     // raw output -> score units
}
