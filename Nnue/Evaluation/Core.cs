using System.Runtime.CompilerServices;
using Nnue.Backends;
using Nnue.Configuration;
using Nnue.Models;

namespace Nnue.Evaluation;
/// <summary>
/// The evaluation core, specialised by the JIT for every backend (<typeparamref name="TB"/> is a struct, so all kernel calls are resolved at
/// compile time). Everything here is unchecked: callers have validated the arguments.
/// </summary>
internal static unsafe class Core<TB> where TB : struct, IBackend
{
    public static void RefreshAccumulator(EvaluationContext context, short* accumulator, ReadOnlySpan<int> activeFeatures)
        => TB.RefreshAccumulator(context.FeatureWeights, context.FeatureBias, activeFeatures, accumulator, context.AccumulatorSize);

    public static void UpdateAccumulator(EvaluationContext context, short* source, short* destination, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures)
        => TB.UpdateAccumulator(context.FeatureWeights, source, removedFeatures, addedFeatures, destination, context.AccumulatorSize);

    public static void AddFeature(EvaluationContext context, short* accumulator, int feature)
        => TB.AddRowToAccumulator(context.FeatureWeights + (long)feature * context.AccumulatorSize, accumulator, context.AccumulatorSize);

    public static void RemoveFeature(EvaluationContext context, short* accumulator, int feature)
        => TB.SubtractRowFromAccumulator(context.FeatureWeights + (long)feature * context.AccumulatorSize, accumulator, context.AccumulatorSize);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* AlignTo64Bytes(byte* pointer) => (byte*)(((nuint)pointer + 63) & ~(nuint)63);

    public static int Evaluate(EvaluationContext context, short** accumulators, int bucket)
        => context.IsInt8 ? EvaluateInt8(context, accumulators, bucket) : EvaluateInt16(context, accumulators, bucket);

    private static int EvaluateInt16(EvaluationContext context, short** accumulators, int bucket)
    {
        short* weights = context.OutputWeights + (long)bucket * context.OutputWeightsStride;
        int sum = context.Activation == FeatureTransformerActivation.SquaredClippedReLU
            ? TB.DotProductSCReLU(accumulators, weights, context.SlotCount, context.AccumulatorSize, context.ClipMax)
            : TB.DotProductCReLU(accumulators, weights, context.SlotCount, context.AccumulatorSize, context.ClipMax);
        return sum + context.OutputBiases[bucket];
    }

    /// <summary>
    /// Stockfish-style pipeline: accumulators -> u8 (CReLU / pairwise / SCReLU) -> [u8 x i8 dense -> activation -> u8]* -> i32 (+ skip).
    /// The scratch memory is stackalloc without zeroing (zeroing cost 5-12 % in the benchmarks).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int EvaluateInt8(EvaluationContext context, short** accumulators, int bucket)
    {
        // transformerOutput and layerOutput are read / written with aligned vector instructions, so they are aligned to 64 bytes by hand
        // (stackalloc only guarantees 8 or 16)
        byte* transformerOutputRaw = stackalloc byte[context.TransformerOutputSize + 128];
        byte* transformerOutput = AlignTo64Bytes(transformerOutputRaw);
        byte* activations = stackalloc byte[context.TotalActivationSize + 64];     // + 64: the last layer's dot product reads up to a multiple of 64
        byte* layerOutputRaw = stackalloc byte[context.MaxPaddedOutputLength * sizeof(int) + 64];
        int* layerOutput = (int*)AlignTo64Bytes(layerOutputRaw);
        int* nonZeroChunks = stackalloc int[context.ChunkCount + 16];     // + 16: FindNonZeroChunks writes whole vectors
        int accumulatorSize = context.AccumulatorSize, slotCount = context.SlotCount;

        switch (context.Activation)
        {
            case FeatureTransformerActivation.ClippedReLU:
                for (int slot = 0; slot < slotCount; slot++) TB.ApplyCReLU(accumulators[slot], transformerOutput + slot * accumulatorSize, accumulatorSize);
                break;
            case FeatureTransformerActivation.SquaredClippedReLU:
                for (int slot = 0; slot < slotCount; slot++) TB.ApplySCReLU(accumulators[slot], transformerOutput + slot * accumulatorSize, accumulatorSize);
                break;
            default:
            {
                int halfLength = accumulatorSize >> 1;
                for (int slot = 0; slot < slotCount; slot++) TB.ApplyPairwise(accumulators[slot], transformerOutput + slot * halfLength, halfLength);
                break;
            }
        }

        Step* steps = context.Steps + bucket * context.LayerCount;
        int lastLayer = context.LayerCount - 1, skipValue = 0, chunkCount = context.ChunkCount;
        var sparseMode = context.SparseMode;
        for (int layer = 0; layer <= lastLayer; layer++)
        {
            Step* step = steps + layer;
            byte* input = transformerOutput;
            if (layer > 0) input = activations + step->InputOffset;

            if (layer == lastLayer)
                return TB.AffineTransformSingleOutput(input, step->Weights, *step->Biases, (step->InputLength + 63) & ~63) + skipValue;

            int nonZeroCount = -1;
            if (layer == 0 && sparseMode != SparseInputMode.Off)
            {
                // cheap popcount pass first; the index list is only built when it pays off
                if (sparseMode == SparseInputMode.Always || TB.CountNonZeroChunks(input, step->InputLength) * 5 <= chunkCount * 2)    // <= 40 % non-zero
                    nonZeroCount = TB.FindNonZeroChunks(input, nonZeroChunks, step->InputLength);
            }
            if (nonZeroCount >= 0) TB.AffineTransformSparse(input, step->Weights, step->Biases, nonZeroChunks, nonZeroCount, layerOutput, step->PaddedOutputLength);
            else TB.AffineTransform(input, step->Weights, step->Biases, layerOutput, step->InputLength, step->PaddedOutputLength);

            // raw int32, same scale as the final output
            if (layer == 0 && context.HasSkipConnection) skipValue = layerOutput[step->OutputLength - 2] - layerOutput[step->OutputLength - 1];

            byte* destination = activations + step->ActivationOffset;
            if (context.UsesSqrHidden)
            {
                TB.ActivateSqrClippedReLU(layerOutput, destination, step->OutputLength);                        // [ sqr | clip ]
                TB.ActivateClippedReLU(layerOutput, destination + step->OutputLength, step->OutputLength);
            }
            else TB.ActivateClippedReLU(layerOutput, destination, step->OutputLength);
        }
        throw new InvalidOperationException();
    }
}
