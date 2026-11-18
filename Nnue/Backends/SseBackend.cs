using System.Runtime.CompilerServices;
using Nnue.Kernels;

namespace Nnue.Backends;

/// <summary>x86 128 bit (SSSE3). Shares the plain Vector128 kernels with NEON; pmaddwd/pmaddubsw kernels are its own.</summary>
internal readonly unsafe struct SseBackend : IBackend
{
    public static SimdBackend Id => SimdBackend.Sse;

    public static void AddRowToAccumulator(short* row, short* accumulator, int length) => Portable128Kernels.AddRowToAccumulator(row, accumulator, length);
    public static void SubtractRowFromAccumulator(short* row, short* accumulator, int length) => Portable128Kernels.SubtractRowFromAccumulator(row, accumulator, length);
    public static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize) => Portable128Kernels.RefreshAccumulator(featureWeights, featureBias, activeFeatures, accumulator, accumulatorSize);
    public static void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize) => Portable128Kernels.UpdateAccumulator(featureWeights, source, removedFeatures, addedFeatures, destination, accumulatorSize);
    public static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => SseKernels.DotProductCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => SseKernels.DotProductSCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static void ApplyCReLU(short* accumulator, byte* output, int length) => Portable128Kernels.ApplyCReLU(accumulator, output, length);
    public static void ApplySCReLU(short* accumulator, byte* output, int length) => Portable128Kernels.ApplySCReLU(accumulator, output, length);
    public static void ApplyPairwise(short* accumulator, byte* output, int halfLength) => Portable128Kernels.ApplyPairwise(accumulator, output, halfLength);
    public static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength) => SseKernels.AffineTransform(input, weights, biases, output, inputLength, paddedOutputLength);
    public static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength) => SseKernels.AffineTransformSparse(input, weights, biases, nonZeroChunks, nonZeroCount, output, paddedOutputLength);
    public static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength) => SseKernels.AffineTransformSingleOutput(input, weights, bias, paddedInputLength);
    public static int CountNonZeroChunks(byte* input, int inputLength) => Portable128Kernels.CountNonZeroChunks((int*)input, inputLength >> 2);
    public static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength) => Portable128Kernels.FindNonZeroChunks(input, nonZeroChunks, inputLength);
    public static void ActivateClippedReLU(int* source, byte* destination, int length) => Portable128Kernels.ActivateClippedReLU(source, destination, length);
    public static void ActivateSqrClippedReLU(int* source, byte* destination, int length) => SseKernels.ActivateSqrClippedReLU(source, destination, length);
}
