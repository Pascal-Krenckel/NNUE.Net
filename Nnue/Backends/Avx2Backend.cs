using System.Runtime.CompilerServices;
using Nnue.Kernels;

namespace Nnue.Backends;

/// <summary>x86 256 bit (AVX2).</summary>
internal readonly unsafe struct Avx2Backend : IBackend
{
    public static SimdBackend Id => SimdBackend.Avx2;

    public static void AddRowToAccumulator(short* row, short* accumulator, int length) => Avx2Kernels.AddRowToAccumulator(row, accumulator, length);
    public static void SubtractRowFromAccumulator(short* row, short* accumulator, int length) => Avx2Kernels.SubtractRowFromAccumulator(row, accumulator, length);
    public static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize) => Avx2Kernels.RefreshAccumulator(featureWeights, featureBias, activeFeatures, accumulator, accumulatorSize);
    public static void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize) => Avx2Kernels.UpdateAccumulator(featureWeights, source, removedFeatures, addedFeatures, destination, accumulatorSize);
    public static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => Avx2Kernels.DotProductCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => Avx2Kernels.DotProductSCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static void ApplyCReLU(short* accumulator, byte* output, int length) => Avx2Kernels.ApplyCReLU(accumulator, output, length);
    public static void ApplySCReLU(short* accumulator, byte* output, int length) => Avx2Kernels.ApplySCReLU(accumulator, output, length);
    public static void ApplyPairwise(short* accumulator, byte* output, int halfLength) => Avx2Kernels.ApplyPairwise(accumulator, output, halfLength);
    public static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength) => Avx2Kernels.AffineTransform(input, weights, biases, output, inputLength, paddedOutputLength);
    public static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength) => Avx2Kernels.AffineTransformSparse(input, weights, biases, nonZeroChunks, nonZeroCount, output, paddedOutputLength);
    public static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength) => Avx2Kernels.AffineTransformSingleOutput(input, weights, bias, paddedInputLength);
    public static int CountNonZeroChunks(byte* input, int inputLength) => Avx2Kernels.CountNonZeroChunks((int*)input, inputLength >> 2);
    public static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength) => Avx2Kernels.FindNonZeroChunks(input, nonZeroChunks, inputLength);
    public static void ActivateClippedReLU(int* source, byte* destination, int length) => Portable128Kernels.ActivateClippedReLU(source, destination, length);
    public static void ActivateSqrClippedReLU(int* source, byte* destination, int length) => SseKernels.ActivateSqrClippedReLU(source, destination, length);
}
