using System.Runtime.CompilerServices;
using Nnue.Kernels;

namespace Nnue.Backends;

/// <summary>AArch64 NEON with the dot-product extension (SDOT).</summary>
internal readonly unsafe struct NeonDotProdBackend : IBackend
{
    public static SimdBackend Id => SimdBackend.NeonDotProd;

    public static void AddRowToAccumulator(short* row, short* accumulator, int length) => Portable128Kernels.AddRowToAccumulator(row, accumulator, length);
    public static void SubtractRowFromAccumulator(short* row, short* accumulator, int length) => Portable128Kernels.SubtractRowFromAccumulator(row, accumulator, length);
    public static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize) => Portable128Kernels.RefreshAccumulator(featureWeights, featureBias, activeFeatures, accumulator, accumulatorSize);
    public static void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize) => Portable128Kernels.UpdateAccumulator(featureWeights, source, removedFeatures, addedFeatures, destination, accumulatorSize);
    public static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => NeonKernels.DotProductCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => NeonKernels.DotProductSCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static void ApplyCReLU(short* accumulator, byte* output, int length) => Portable128Kernels.ApplyCReLU(accumulator, output, length);
    public static void ApplySCReLU(short* accumulator, byte* output, int length) => Portable128Kernels.ApplySCReLU(accumulator, output, length);
    public static void ApplyPairwise(short* accumulator, byte* output, int halfLength) => Portable128Kernels.ApplyPairwise(accumulator, output, halfLength);
    public static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength) => NeonKernels.AffineTransformDotProd(input, weights, biases, output, inputLength, paddedOutputLength);
    public static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength) => NeonKernels.AffineTransformSparseDotProd(input, weights, biases, nonZeroChunks, nonZeroCount, output, paddedOutputLength);
    public static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength) => NeonKernels.AffineTransformSingleOutputDotProd(input, weights, bias, paddedInputLength);
    public static int CountNonZeroChunks(byte* input, int inputLength) => Portable128Kernels.CountNonZeroChunks((int*)input, inputLength >> 2);
    public static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength) => Portable128Kernels.FindNonZeroChunks(input, nonZeroChunks, inputLength);
    public static void ActivateClippedReLU(int* source, byte* destination, int length) => Portable128Kernels.ActivateClippedReLU(source, destination, length);
    public static void ActivateSqrClippedReLU(int* source, byte* destination, int length) => Portable128Kernels.ActivateSqrClippedReLU(source, destination, length);
}
