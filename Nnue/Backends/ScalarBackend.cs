using Nnue.Kernels;

namespace Nnue.Backends;

/// <summary>Plain C# reference.</summary>
internal readonly unsafe struct ScalarBackend : IBackend
{
    public static SimdBackend Id => SimdBackend.Scalar;

    public static void AddRowToAccumulator(short* row, short* accumulator, int length) => ScalarKernels.AddRowToAccumulator(row, accumulator, length);
    public static void SubtractRowFromAccumulator(short* row, short* accumulator, int length) => ScalarKernels.SubtractRowFromAccumulator(row, accumulator, length);
    public static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize) => ScalarKernels.RefreshAccumulator(featureWeights, featureBias, activeFeatures, accumulator, accumulatorSize);
    public static void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize) => ScalarKernels.UpdateAccumulator(featureWeights, source, removedFeatures, addedFeatures, destination, accumulatorSize);
    public static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => ScalarKernels.DotProductCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax) => ScalarKernels.DotProductSCReLU(accumulators, weights, accumulatorCount, accumulatorSize, (short)clipMax);
    public static void ApplyCReLU(short* accumulator, byte* output, int length) => ScalarKernels.ApplyCReLU(accumulator, output, length);
    public static void ApplySCReLU(short* accumulator, byte* output, int length) => ScalarKernels.ApplySCReLU(accumulator, output, length);
    public static void ApplyPairwise(short* accumulator, byte* output, int halfLength) => ScalarKernels.ApplyPairwise(accumulator, output, halfLength);
    public static void ApplyMaxPooling(short* accumulator, byte* output, int halfLength) => ScalarKernels.ApplyMaxPooling(accumulator, output, halfLength);
    public static void ApplyAveragePooling(short* accumulator, byte* output, int halfLength) => ScalarKernels.ApplyAveragePooling(accumulator, output, halfLength);
    public static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength) => ScalarKernels.AffineTransform(input, weights, biases, output, inputLength, paddedOutputLength);
    public static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength) => ScalarKernels.AffineTransformSparse(input, weights, biases, nonZeroChunks, nonZeroCount, output, paddedOutputLength);
    public static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength) => ScalarKernels.AffineTransformSingleOutput(input, weights, bias, paddedInputLength);
    public static int CountNonZeroChunks(byte* input, int inputLength) => ScalarKernels.CountNonZeroChunks((int*)input, inputLength >> 2);
    public static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength) => ScalarKernels.FindNonZeroChunks(input, nonZeroChunks, inputLength);
    public static void ActivateClippedReLU(int* source, byte* destination, int length) => ScalarKernels.ActivateClippedReLU(source, destination, length);
    public static void ActivateSqrClippedReLU(int* source, byte* destination, int length) => ScalarKernels.ActivateSqrClippedReLU(source, destination, length);
    public static void ActivateQuantmoid4(int* source, byte* destination, int length) => ScalarKernels.ActivateQuantmoid4(source, destination, length);
}
