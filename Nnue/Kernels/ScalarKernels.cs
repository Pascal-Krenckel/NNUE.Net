using System.Runtime.CompilerServices;

namespace Nnue.Kernels;

/// <summary>Plain C# reference implementation of every operation. All other backends must match it bit for bit.</summary>
internal static unsafe class ScalarKernels
{
    internal static void AddRowToAccumulator(short* row, short* accumulator, int length)
    {
        for (int i = 0; i < length; i++)
            accumulator[i] = (short)(accumulator[i] + row[i]);
    }

    internal static void SubtractRowFromAccumulator(short* row, short* accumulator, int length)
    {
        for (int i = 0; i < length; i++)
            accumulator[i] = (short)(accumulator[i] - row[i]);
    }

    internal static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize)
    {
        for (int i = 0; i < accumulatorSize; i++)
        {
            int s = featureBias[i];
            foreach (int f in activeFeatures)
                s += featureWeights[((long)f * accumulatorSize) + i];
            accumulator[i] = (short)s;
        }
    }

    internal static void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize)
    {
        for (int i = 0; i < accumulatorSize; i++)
        {
            int s = source[i];
            foreach (int f in removedFeatures)
                s -= featureWeights[((long)f * accumulatorSize) + i];
            foreach (int f in addedFeatures)
                s += featureWeights[((long)f * accumulatorSize) + i];
            destination[i] = (short)s;
        }
    }

    internal static void ApplyCReLU(short* accumulator, byte* output, int length)
    {
        for (int i = 0; i < length; i++)
            output[i] = (byte)Math.Clamp((int)accumulator[i], 0, 127);
    }

    internal static void ApplySCReLU(short* accumulator, byte* output, int length)
    {
        for (int i = 0; i < length; i++)
        {
            int c = Math.Clamp((int)accumulator[i], 0, 255);
            output[i] = (byte)((c * c) >> 9);
        }
    }

    internal static void ApplyPairwise(short* accumulator, byte* output, int halfLength)
    {
        for (int j = 0; j < halfLength; j++)
        {
            int x = Math.Clamp((int)accumulator[j], 0, 255);
            int y = Math.Clamp((int)accumulator[j + halfLength], 0, 255);
            output[j] = (byte)((x * y) >> 9);
        }
    }

    internal static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
    {
        for (int o = 0; o < paddedOutputLength; o++)
        {
            int s = biases[o];
            for (int i = 0; i < inputLength; i++)
                s += input[i] * weights[((i >> 2) * paddedOutputLength * 4) + (o * 4) + (i & 3)];
            output[o] = s;
        }
    }

    internal static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
    {
        for (int o = 0; o < paddedOutputLength; o++)
        {
            int s = biases[o];
            for (int j = 0; j < nonZeroCount; j++)
            {
                int c = nonZeroChunks[j];
                for (int t = 0; t < 4; t++)
                    s += input[(c * 4) + t] * weights[(c * paddedOutputLength * 4) + (o * 4) + t];
            }
            output[o] = s;
        }
    }

    internal static int CountNonZeroChunks(int* input, int chunkCount)
    {
        int count = 0;
        for (int c = 0; c < chunkCount; c++)
            if (input[c] != 0)
                count++;
        return count;
    }

    internal static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength)
    {
        int count = 0;
        for (int c = 0; c < (inputLength >> 2); c++)
            if (Unsafe.ReadUnaligned<int>(input + (c * 4)) != 0)
                nonZeroChunks[count++] = c;
        return count;
    }

    internal static void ActivateClippedReLU(int* source, byte* destination, int length)
    {
        for (int i = 0; i < length; i++)
            destination[i] = (byte)Math.Clamp(source[i] >> 6, 0, 127);
    }

    internal static void ActivateSqrClippedReLU(int* source, byte* destination, int length)
    {
        for (int i = 0; i < length; i++)
        {
            long v = source[i];
            destination[i] = (byte)Math.Min(127L, (v * v) >> 19);
        }
    }

    /// <summary>Int16 output layer over all slots: sum over the slots s of clamp(slot_s, 0, qa) * w[s*n + i].</summary>
    internal static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        int sum = 0;
        for (int s = 0; s < accumulatorCount; s++)
            for (int i = 0; i < accumulatorSize; i++)
                sum += Math.Clamp((int)accumulators[s][i], 0, clipMax) * weights[(s * accumulatorSize) + i];
        return sum;
    }

    /// <summary>The same for SCReLU: ((c * w) as int16) * c; the (short) cast reproduces the int16 product of the SIMD path.</summary>
    internal static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        int sum = 0;
        for (int s = 0; s < accumulatorCount; s++)
            for (int i = 0; i < accumulatorSize; i++)
            {
                int c = Math.Clamp((int)accumulators[s][i], 0, clipMax);
                sum += (short)(c * weights[(s * accumulatorSize) + i]) * c;
            }
        return sum;
    }

    /// <summary>Last dense layer (one output): bias + sum of input[i] * w[i], contiguous weights. n is a multiple of 64; weights past In are zero
    /// and the input buffer is readable up to n, so the bytes beyond In contribute nothing.</summary>
    internal static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength)
    {
        int s = bias;
        for (int i = 0; i < paddedInputLength; i++)
            s += input[i] * weights[i];
        return s;
    }
}
