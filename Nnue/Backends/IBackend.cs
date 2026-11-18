namespace Nnue.Backends;

/// <summary>
/// Everything an instruction set has to provide. Each backend is one struct in its own file; the evaluation core is generic over it
/// (<c>Core&lt;TBackend&gt;</c>), so the JIT resolves every call at compile time and the compiler checks that no operation is missing.
/// All backends must produce bit-identical results to <see cref="ScalarBackend"/>.
/// </summary>
internal unsafe interface IBackend
{
    static abstract SimdBackend Id { get; }

    static abstract void AddRowToAccumulator(short* row, short* accumulator, int length);
    static abstract void SubtractRowFromAccumulator(short* row, short* accumulator, int length);
    static abstract void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize);
    static abstract void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize);
    static abstract int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax);
    static abstract int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, int clipMax);
    static abstract void ApplyCReLU(short* accumulator, byte* output, int length);
    static abstract void ApplySCReLU(short* accumulator, byte* output, int length);
    static abstract void ApplyPairwise(short* accumulator, byte* output, int halfLength);
    static abstract void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength);
    static abstract void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength);
    static abstract int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength);
    static abstract int CountNonZeroChunks(byte* input, int inputLength);
    static abstract int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength);
    static abstract void ActivateClippedReLU(int* source, byte* destination, int length);
    static abstract void ActivateSqrClippedReLU(int* source, byte* destination, int length);
}
