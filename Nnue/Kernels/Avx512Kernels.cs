using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Nnue.Kernels;

/// <summary>512-bit x86 kernels (AVX-512 F + BW).</summary>
internal static unsafe class Avx512Kernels
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AddRowToAccumulator(short* row, short* accumulator, int length)
    {
        for (long i = 0; i < length; i += 32)
            Vector512.Add(Vector512.LoadAligned(accumulator + i), Vector512.LoadAligned(row + i)).StoreAligned(accumulator + i);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void SubtractRowFromAccumulator(short* row, short* accumulator, int length)
    {
        for (long i = 0; i < length; i += 32)
            Vector512.Subtract(Vector512.LoadAligned(accumulator + i), Vector512.LoadAligned(row + i)).StoreAligned(accumulator + i);
    }

    /// <summary>
    /// accumulator = bias + the rows of all active features. The accumulator is processed in tiles of 4 vectors (128 values) held in
    /// independent registers, so the adds of one feature row do not wait for each other (measured: 20-40 % faster than one vector per pass
    /// while the feature matrix is in cache, equal when it is not); the tail of sizes that are not a multiple of 128 goes one vector at a time.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize)
    {
        long i = 0, size = accumulatorSize;
        for (; i + 128 <= size; i += 128)
        {
            Vector512<short> a0 = Vector512.LoadAligned(featureBias + i);
            Vector512<short> a1 = Vector512.LoadAligned(featureBias + i + 32);
            Vector512<short> a2 = Vector512.LoadAligned(featureBias + i + 64);
            Vector512<short> a3 = Vector512.LoadAligned(featureBias + i + 96);
            foreach (int f in activeFeatures)
            {
                short* row = featureWeights + (f * size) + i;
                a0 += Vector512.LoadAligned(row);
                a1 += Vector512.LoadAligned(row + 32);
                a2 += Vector512.LoadAligned(row + 64);
                a3 += Vector512.LoadAligned(row + 96);
            }
            a0.StoreAligned(accumulator + i);
            a1.StoreAligned(accumulator + i + 32);
            a2.StoreAligned(accumulator + i + 64);
            a3.StoreAligned(accumulator + i + 96);
        }
        for (; i < size; i += 32)
        {
            Vector512<short> acc = Vector512.LoadAligned(featureBias + i);
            foreach (int f in activeFeatures)
                acc += Vector512.LoadAligned(featureWeights + (f * size) + i);
            acc.StoreAligned(accumulator + i);
        }
    }

    /// <summary>destination = source - removed rows + added rows, tiled like <see cref="RefreshAccumulator"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize)
    {
        long i = 0, size = accumulatorSize;
        for (; i + 128 <= size; i += 128)
        {
            Vector512<short> a0 = Vector512.LoadAligned(source + i);
            Vector512<short> a1 = Vector512.LoadAligned(source + i + 32);
            Vector512<short> a2 = Vector512.LoadAligned(source + i + 64);
            Vector512<short> a3 = Vector512.LoadAligned(source + i + 96);
            foreach (int f in removedFeatures)
            {
                short* row = featureWeights + (f * size) + i;
                a0 -= Vector512.LoadAligned(row);
                a1 -= Vector512.LoadAligned(row + 32);
                a2 -= Vector512.LoadAligned(row + 64);
                a3 -= Vector512.LoadAligned(row + 96);
            }
            foreach (int f in addedFeatures)
            {
                short* row = featureWeights + (f * size) + i;
                a0 += Vector512.LoadAligned(row);
                a1 += Vector512.LoadAligned(row + 32);
                a2 += Vector512.LoadAligned(row + 64);
                a3 += Vector512.LoadAligned(row + 96);
            }
            a0.StoreAligned(destination + i);
            a1.StoreAligned(destination + i + 32);
            a2.StoreAligned(destination + i + 64);
            a3.StoreAligned(destination + i + 96);
        }
        for (; i < size; i += 32)
        {
            Vector512<short> acc = Vector512.LoadAligned(source + i);
            foreach (int f in removedFeatures)
                acc -= Vector512.LoadAligned(featureWeights + (f * size) + i);
            foreach (int f in addedFeatures)
                acc += Vector512.LoadAligned(featureWeights + (f * size) + i);
            acc.StoreAligned(destination + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyCReLU(short* accumulator, byte* output, int length)
    {
        Vector512<short> zero = Vector512<short>.Zero;
        Vector512<short> max = Vector512.Create((short)127);
        long i = 0, len = length;
        for (; i + 64 <= len; i += 64)
        {
            Vector512<short> a = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + i), zero), max);
            Vector512<short> c = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + (i + 32)), zero), max);
            Vector512.Narrow(a.AsUInt16(), c.AsUInt16()).Store(output + i);
        }
        if (i < len)
        {
            Vector512<short> a = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + i), zero), max);
            Vector512.Narrow(a.AsUInt16(), Vector512<ushort>.Zero).GetLower().Store(output + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<ushort> PairProduct512(short* accumulator, long offset, long halfLength, Vector512<short> zero, Vector512<short> max)
    {
        Vector512<ushort> a = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + offset), zero), max).AsUInt16();
        Vector512<ushort> c = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + (offset + halfLength)), zero), max).AsUInt16();
        return Vector512.ShiftRightLogical(a * c, 9);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyPairwise(short* accumulator, byte* output, int halfLength)
    {
        Vector512<short> zero = Vector512<short>.Zero;
        Vector512<short> max = Vector512.Create((short)255);
        long i = 0, len = halfLength;
        for (; i + 64 <= len; i += 64)
            Vector512.Narrow(PairProduct512(accumulator, i, len, zero, max), PairProduct512(accumulator, i + 32, len, zero, max)).Store(output + i);
        if (i < len)
            Vector512.Narrow(PairProduct512(accumulator, i, len, zero, max), Vector512<ushort>.Zero).GetLower().Store(output + i);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplySCReLU(short* accumulator, byte* output, int length)
    {
        Vector512<short> zero = Vector512<short>.Zero;
        Vector512<short> max = Vector512.Create((short)255);
        long i = 0, len = length;
        for (; i + 64 <= len; i += 64)
        {
            Vector512<ushort> a = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + i), zero), max).AsUInt16();
            Vector512<ushort> c = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + (i + 32)), zero), max).AsUInt16();
            Vector512.Narrow(Vector512.ShiftRightLogical(a * a, 9), Vector512.ShiftRightLogical(c * c, 9)).Store(output + i);
        }
        if (i < len)
        {
            Vector512<ushort> a = Vector512.Min(Vector512.Max(Vector512.LoadAligned(accumulator + i), zero), max).AsUInt16();
            Vector512.Narrow(Vector512.ShiftRightLogical(a * a, 9), Vector512<ushort>.Zero).GetLower().Store(output + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
    {
        // 4 input chunks (16 inputs) per iteration with 4 independent accumulators: the latency of
        // pmaddubsw -> pmaddwd -> paddd is hidden instead of serializing every chunk. inputLength % 16 == 0 is guaranteed.
        Vector512<short> ones = Vector512.Create((short)1);
        int stride = paddedOutputLength * 4;                       // bytes between consecutive 4-input chunks
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            Vector512<int> a0 = Vector512.LoadAligned(biases + ((long)o));
            Vector512<int> a1 = Vector512<int>.Zero;
            Vector512<int> a2 = Vector512<int>.Zero;
            Vector512<int> a3 = Vector512<int>.Zero;
            sbyte* wp = weights + (o * 4);
            for (int i = 0; i < inputLength; i += 16)
            {
                sbyte* wc = wp + ((i >> 2) * stride);
                Vector512<byte> i0 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + i)).AsByte();
                Vector512<byte> i1 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (i + 4))).AsByte();
                Vector512<byte> i2 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (i + 8))).AsByte();
                Vector512<byte> i3 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (i + 12))).AsByte();
                a0 = Vector512.Add(a0, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i0, Vector512.LoadAligned(wc)), ones));
                a1 = Vector512.Add(a1, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i1, Vector512.LoadAligned(wc + stride)), ones));
                a2 = Vector512.Add(a2, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i2, Vector512.LoadAligned(wc + (2 * stride))), ones));
                a3 = Vector512.Add(a3, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i3, Vector512.LoadAligned(wc + (3 * stride))), ones));
            }
            Vector512.Add(Vector512.Add(a0, a1), Vector512.Add(a2, a3)).StoreAligned(output + ((long)o));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int CountNonZeroChunks(int* input, int chunkCount)
    {
        int n = 0;
        for (int c = 0; c < chunkCount; c += 16)
            n += BitOperations.PopCount((uint)~Vector512.Equals(Vector512.LoadAligned(input + ((long)c)), Vector512<int>.Zero).ExtractMostSignificantBits() & 0xFFFFu);
        return n;
    }

    /// <summary>Index list of the non-zero 4-byte chunks. Writes up to 8 values past the last index (the caller pads the buffer by 16 ints).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength)
    {
        int count = 0, chunks = inputLength >> 2;
        int* src = (int*)input;
        uint* lut = NnzTable.Lut;
        for (int c = 0; c < chunks; c += 16)
        {
            uint m = (uint)~Vector512.Equals(Vector512.LoadAligned(src + c), Vector512<int>.Zero).ExtractMostSignificantBits() & 0xFFFFu;
            uint m0 = m & 0xFFu, m1 = m >> 8;
            Vector256.Add(Vector256.LoadAligned(lut + (m0 * 8)), Vector256.Create((uint)c)).AsInt32().Store(nonZeroChunks + count);
            count += BitOperations.PopCount(m0);
            Vector256.Add(Vector256.LoadAligned(lut + (m1 * 8)), Vector256.Create((uint)c + 8)).AsInt32().Store(nonZeroChunks + count);
            count += BitOperations.PopCount(m1);
        }
        return count;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
    {
        Vector512<short> ones = Vector512.Create((short)1);
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            Vector512<int> a0 = Vector512.LoadAligned(biases + ((long)o));
            Vector512<int> a1 = Vector512<int>.Zero;
            Vector512<int> a2 = Vector512<int>.Zero;
            Vector512<int> a3 = Vector512<int>.Zero;
            sbyte* wp = weights + (o * 4);
            int j = 0;
            for (; j + 4 <= nonZeroCount; j += 4)
            {
                int c0 = nonZeroChunks[j], c1 = nonZeroChunks[j + 1], c2 = nonZeroChunks[j + 2], c3 = nonZeroChunks[j + 3];
                Vector512<byte> i0 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (c0 * 4))).AsByte();
                Vector512<byte> i1 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (c1 * 4))).AsByte();
                Vector512<byte> i2 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (c2 * 4))).AsByte();
                Vector512<byte> i3 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (c3 * 4))).AsByte();
                a0 = Vector512.Add(a0, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i0, Vector512.LoadAligned(wp + (c0 * stride))), ones));
                a1 = Vector512.Add(a1, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i1, Vector512.LoadAligned(wp + (c1 * stride))), ones));
                a2 = Vector512.Add(a2, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i2, Vector512.LoadAligned(wp + (c2 * stride))), ones));
                a3 = Vector512.Add(a3, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i3, Vector512.LoadAligned(wp + (c3 * stride))), ones));
            }
            for (; j < nonZeroCount; j++)
            {
                int c0 = nonZeroChunks[j];
                Vector512<byte> i0 = Vector512.Create(Unsafe.ReadUnaligned<int>(input + (c0 * 4))).AsByte();
                a0 = Vector512.Add(a0, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(i0, Vector512.LoadAligned(wp + (c0 * stride))), ones));
            }
            Vector512.Add(Vector512.Add(a0, a1), Vector512.Add(a2, a3)).StoreAligned(output + ((long)o));
        }
    }

    /// <summary>
    /// Int16 output layer over all slots: sum over the slots s of clamp(slot_s, 0, qa) * w[s*n .. s*n+n).
    /// The products of all slots accumulate in the same vector registers, there is a single horizontal reduction at the end.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        Vector512<short> zero = Vector512<short>.Zero;
        Vector512<short> max = Vector512.Create(clipMax);
        Vector512<int> s0 = Vector512<int>.Zero;
        Vector512<int> s1 = Vector512<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = weights + ((long)s * accumulatorSize);
            long i = 0, len = accumulatorSize;
            for (; i + 64 <= len; i += 64)
            {
                Vector512<short> x0 = Vector512.Min(Vector512.Max(Vector512.LoadAligned(a + i), zero), max);
                Vector512<short> x1 = Vector512.Min(Vector512.Max(Vector512.LoadAligned(a + (i + 32)), zero), max);
                s0 = Vector512.Add(s0, Avx512BW.MultiplyAddAdjacent(x0, Vector512.LoadAligned(ws + i)));
                s1 = Vector512.Add(s1, Avx512BW.MultiplyAddAdjacent(x1, Vector512.LoadAligned(ws + (i + 32))));
            }
            if (i < len)
            {
                Vector512<short> x0 = Vector512.Min(Vector512.Max(Vector512.LoadAligned(a + i), zero), max);
                s0 = Vector512.Add(s0, Avx512BW.MultiplyAddAdjacent(x0, Vector512.LoadAligned(ws + i)));
            }
        }
        return Vector512.Sum(Vector512.Add(s0, s1));
    }

    /// <summary>The same for SCReLU: sum of ((c * w) as int16) * c with c = clamp(slot, 0, qa).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        Vector512<short> zero = Vector512<short>.Zero;
        Vector512<short> max = Vector512.Create(clipMax);
        Vector512<int> s0 = Vector512<int>.Zero;
        Vector512<int> s1 = Vector512<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = weights + ((long)s * accumulatorSize);
            long i = 0, len = accumulatorSize;
            for (; i + 64 <= len; i += 64)
            {
                Vector512<short> x0 = Vector512.Min(Vector512.Max(Vector512.LoadAligned(a + i), zero), max);
                Vector512<short> x1 = Vector512.Min(Vector512.Max(Vector512.LoadAligned(a + (i + 32)), zero), max);
                s0 = Vector512.Add(s0, Avx512BW.MultiplyAddAdjacent(x0 * Vector512.LoadAligned(ws + i), x0));
                s1 = Vector512.Add(s1, Avx512BW.MultiplyAddAdjacent(x1 * Vector512.LoadAligned(ws + (i + 32)), x1));
            }
            if (i < len)
            {
                Vector512<short> x0 = Vector512.Min(Vector512.Max(Vector512.LoadAligned(a + i), zero), max);
                s0 = Vector512.Add(s0, Avx512BW.MultiplyAddAdjacent(x0 * Vector512.LoadAligned(ws + i), x0));
            }
        }
        return Vector512.Sum(Vector512.Add(s0, s1));
    }

    /// <summary>Last dense layer (one output): bias + sum of input[i] * w[i], contiguous weights. n is a multiple of 64; weights past In are zero
    /// and the input buffer is readable up to n, so the bytes beyond In contribute nothing.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength)
    {
        Vector512<short> ones = Vector512.Create((short)1);
        Vector512<int> a0 = Vector512<int>.Zero;
        for (int i = 0; i < paddedInputLength; i += 64)
            a0 = Vector512.Add(a0, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(Vector512.Load(input + i), Vector512.LoadAligned(weights + i)), ones));
        return bias + Vector512.Sum(a0);
    }
}
