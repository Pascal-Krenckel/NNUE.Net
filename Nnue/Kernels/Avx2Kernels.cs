using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Nnue.Kernels;

/// <summary>256-bit x86 kernels (AVX2).</summary>
internal static unsafe class Avx2Kernels
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AddRowToAccumulator(short* row, short* accumulator, int length)
    {
        for (long i = 0; i < (long)length; i += 16)
            Vector256.Add(Vector256.LoadAligned(accumulator + (i)), Vector256.LoadAligned(row + (i))).StoreAligned(accumulator + (i));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void SubtractRowFromAccumulator(short* row, short* accumulator, int length)
    {
        for (long i = 0; i < (long)length; i += 16)
            Vector256.Subtract(Vector256.LoadAligned(accumulator + (i)), Vector256.LoadAligned(row + (i))).StoreAligned(accumulator + (i));
    }

    /// <summary>
    /// accumulator = bias + the rows of all active features. The accumulator is processed in tiles of 8 vectors (128 values) held in
    /// independent registers, so the adds of one feature row do not wait for each other (measured: 20-40 % faster than one vector per pass
    /// while the feature matrix is in cache, equal when it is not); the tail of sizes that are not a multiple of 128 goes one vector at a time.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize)
    {
        long i = 0, size = accumulatorSize;
        for (; i + 128 <= size; i += 128)
        {
            var a0 = Vector256.LoadAligned(featureBias + i); var a1 = Vector256.LoadAligned(featureBias + i + 16); var a2 = Vector256.LoadAligned(featureBias + i + 32); var a3 = Vector256.LoadAligned(featureBias + i + 48); var a4 = Vector256.LoadAligned(featureBias + i + 64); var a5 = Vector256.LoadAligned(featureBias + i + 80); var a6 = Vector256.LoadAligned(featureBias + i + 96); var a7 = Vector256.LoadAligned(featureBias + i + 112);
            foreach (int f in activeFeatures)
            {
                short* row = featureWeights + f * size + i;
                a0 += Vector256.LoadAligned(row); a1 += Vector256.LoadAligned(row + 16); a2 += Vector256.LoadAligned(row + 32); a3 += Vector256.LoadAligned(row + 48); a4 += Vector256.LoadAligned(row + 64); a5 += Vector256.LoadAligned(row + 80); a6 += Vector256.LoadAligned(row + 96); a7 += Vector256.LoadAligned(row + 112);
            }
            a0.StoreAligned(accumulator + i); a1.StoreAligned(accumulator + i + 16); a2.StoreAligned(accumulator + i + 32); a3.StoreAligned(accumulator + i + 48); a4.StoreAligned(accumulator + i + 64); a5.StoreAligned(accumulator + i + 80); a6.StoreAligned(accumulator + i + 96); a7.StoreAligned(accumulator + i + 112);
        }
        for (; i < size; i += 16)
        {
            var acc = Vector256.LoadAligned(featureBias + i);
            foreach (int f in activeFeatures) acc += Vector256.LoadAligned(featureWeights + f * size + i);
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
            var a0 = Vector256.LoadAligned(source + i); var a1 = Vector256.LoadAligned(source + i + 16); var a2 = Vector256.LoadAligned(source + i + 32); var a3 = Vector256.LoadAligned(source + i + 48); var a4 = Vector256.LoadAligned(source + i + 64); var a5 = Vector256.LoadAligned(source + i + 80); var a6 = Vector256.LoadAligned(source + i + 96); var a7 = Vector256.LoadAligned(source + i + 112);
            foreach (int f in removedFeatures)
            {
                short* row = featureWeights + f * size + i;
                a0 -= Vector256.LoadAligned(row); a1 -= Vector256.LoadAligned(row + 16); a2 -= Vector256.LoadAligned(row + 32); a3 -= Vector256.LoadAligned(row + 48); a4 -= Vector256.LoadAligned(row + 64); a5 -= Vector256.LoadAligned(row + 80); a6 -= Vector256.LoadAligned(row + 96); a7 -= Vector256.LoadAligned(row + 112);
            }
            foreach (int f in addedFeatures)
            {
                short* row = featureWeights + f * size + i;
                a0 += Vector256.LoadAligned(row); a1 += Vector256.LoadAligned(row + 16); a2 += Vector256.LoadAligned(row + 32); a3 += Vector256.LoadAligned(row + 48); a4 += Vector256.LoadAligned(row + 64); a5 += Vector256.LoadAligned(row + 80); a6 += Vector256.LoadAligned(row + 96); a7 += Vector256.LoadAligned(row + 112);
            }
            a0.StoreAligned(destination + i); a1.StoreAligned(destination + i + 16); a2.StoreAligned(destination + i + 32); a3.StoreAligned(destination + i + 48); a4.StoreAligned(destination + i + 64); a5.StoreAligned(destination + i + 80); a6.StoreAligned(destination + i + 96); a7.StoreAligned(destination + i + 112);
        }
        for (; i < size; i += 16)
        {
            var acc = Vector256.LoadAligned(source + i);
            foreach (int f in removedFeatures) acc -= Vector256.LoadAligned(featureWeights + f * size + i);
            foreach (int f in addedFeatures) acc += Vector256.LoadAligned(featureWeights + f * size + i);
            acc.StoreAligned(destination + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyCReLU(short* accumulator, byte* output, int length)
    {
        var zero = Vector256<short>.Zero; var max = Vector256.Create((short)127);
        long i = 0, len = (long)length;
        for (; i + 32 <= len; i += 32)
        {
            var a = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (i)), zero), max);
            var c = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (i + 16)), zero), max);
            Vector256.Narrow(a.AsUInt16(), c.AsUInt16()).Store(output + (i));
        }
        if (i < len)
        {
            var a = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (i)), zero), max);
            Vector256.Narrow(a.AsUInt16(), Vector256<ushort>.Zero).GetLower().Store(output + (i));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<ushort> PairProduct256(short* accumulator, long offset, long halfLength, Vector256<short> zero, Vector256<short> max)
    {
        var a = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (offset)), zero), max).AsUInt16();
        var c = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (offset + halfLength)), zero), max).AsUInt16();
        return Vector256.ShiftRightLogical(a * c, 9);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyPairwise(short* accumulator, byte* output, int halfLength)
    {
        var zero = Vector256<short>.Zero; var max = Vector256.Create((short)255);
        long i = 0, len = (long)halfLength;
        for (; i + 32 <= len; i += 32)
            Vector256.Narrow(PairProduct256(accumulator, i, len, zero, max), PairProduct256(accumulator, i + 16, len, zero, max)).Store(output + (i));
        if (i < len)
            Vector256.Narrow(PairProduct256(accumulator, i, len, zero, max), Vector256<ushort>.Zero).GetLower().Store(output + (i));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplySCReLU(short* accumulator, byte* output, int length)
    {
        var zero = Vector256<short>.Zero; var max = Vector256.Create((short)255);
        long i = 0, len = (long)length;
        for (; i + 32 <= len; i += 32)
        {
            var a = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (i)), zero), max).AsUInt16();
            var c = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (i + 16)), zero), max).AsUInt16();
            Vector256.Narrow(Vector256.ShiftRightLogical(a * a, 9), Vector256.ShiftRightLogical(c * c, 9)).Store(output + (i));
        }
        if (i < len)
        {
            var a = Vector256.Min(Vector256.Max(Vector256.LoadAligned(accumulator + (i)), zero), max).AsUInt16();
            Vector256.Narrow(Vector256.ShiftRightLogical(a * a, 9), Vector256<ushort>.Zero).GetLower().Store(output + (i));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
    {
        // 4 input chunks (16 inputs) per iteration with 4 independent accumulators: the latency of
        // pmaddubsw -> pmaddwd -> paddd is hidden instead of serializing every chunk. inputLength % 16 == 0 is guaranteed.
        var ones = Vector256.Create((short)1);
        int stride = paddedOutputLength * 4;                       // bytes between consecutive 4-input chunks
        for (int o = 0; o < paddedOutputLength; o += 8)
        {
            var a0 = Vector256.LoadAligned(biases + ((long)o));
            var a1 = Vector256<int>.Zero;
            var a2 = Vector256<int>.Zero;
            var a3 = Vector256<int>.Zero;
            sbyte* wp = (weights + (o * 4));
            for (int i = 0; i < inputLength; i += 16)
            {
                sbyte* wc = (wp + ((i >> 2) * stride));
                var i0 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (i)))).AsByte();
                var i1 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (i + 4)))).AsByte();
                var i2 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (i + 8)))).AsByte();
                var i3 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (i + 12)))).AsByte();
                a0 = Vector256.Add(a0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i0, Vector256.LoadAligned(wc)), ones));
                a1 = Vector256.Add(a1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i1, Vector256.LoadAligned((wc + (stride)))), ones));
                a2 = Vector256.Add(a2, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i2, Vector256.LoadAligned((wc + (2 * stride)))), ones));
                a3 = Vector256.Add(a3, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i3, Vector256.LoadAligned((wc + (3 * stride)))), ones));
            }
            Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3)).StoreAligned(output + ((long)o));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int CountNonZeroChunks(int* input, int chunkCount)
    {
        int n = 0;
        for (int c = 0; c < chunkCount; c += 8)
            n += BitOperations.PopCount(~Vector256.Equals(Vector256.LoadAligned(input + ((long)c)), Vector256<int>.Zero).ExtractMostSignificantBits() & 0xFFu);
        return n;
    }

    /// <summary>Index list of the non-zero 4-byte chunks. Writes up to 8 values past the last index (the caller pads the buffer by 16 ints).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength)
    {
        int count = 0, chunks = inputLength >> 2;
        int* src = (int*)input;
        uint* lut = NnzTable.Lut;
        for (int c = 0; c < chunks; c += 8)
        {
            uint m = ~Vector256.Equals(Vector256.LoadAligned(src + c), Vector256<int>.Zero).ExtractMostSignificantBits() & 0xFFu;
            Vector256.Add(Vector256.LoadAligned(lut + m * 8), Vector256.Create((uint)c)).AsInt32().Store(nonZeroChunks + count);
            count += BitOperations.PopCount(m);
        }
        return count;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
    {
        var ones = Vector256.Create((short)1);
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 8)
        {
            var a0 = Vector256.LoadAligned(biases + ((long)o));
            var a1 = Vector256<int>.Zero;
            var a2 = Vector256<int>.Zero;
            var a3 = Vector256<int>.Zero;
            sbyte* wp = (weights + (o * 4));
            int j = 0;
            for (; j + 4 <= nonZeroCount; j += 4)
            {
                int c0 = nonZeroChunks[j], c1 = nonZeroChunks[j + 1], c2 = nonZeroChunks[j + 2], c3 = nonZeroChunks[j + 3];
                var i0 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (c0 * 4)))).AsByte();
                var i1 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (c1 * 4)))).AsByte();
                var i2 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (c2 * 4)))).AsByte();
                var i3 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (c3 * 4)))).AsByte();
                a0 = Vector256.Add(a0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i0, Vector256.LoadAligned((wp + (c0 * stride)))), ones));
                a1 = Vector256.Add(a1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i1, Vector256.LoadAligned((wp + (c1 * stride)))), ones));
                a2 = Vector256.Add(a2, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i2, Vector256.LoadAligned((wp + (c2 * stride)))), ones));
                a3 = Vector256.Add(a3, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i3, Vector256.LoadAligned((wp + (c3 * stride)))), ones));
            }
            for (; j < nonZeroCount; j++)
            {
                int c0 = nonZeroChunks[j];
                var i0 = Vector256.Create(Unsafe.ReadUnaligned<int>((input + (c0 * 4)))).AsByte();
                a0 = Vector256.Add(a0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(i0, Vector256.LoadAligned((wp + (c0 * stride)))), ones));
            }
            Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3)).StoreAligned(output + ((long)o));
        }
    }

    /// <summary>
    /// Int16 output layer over all slots: sum over the slots s of clamp(slot_s, 0, qa) * w[s*n .. s*n+n).
    /// The products of all slots accumulate in the same vector registers, there is a single horizontal reduction at the end.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        var zero = Vector256<short>.Zero; var max = Vector256.Create(clipMax);
        var s0 = Vector256<int>.Zero; var s1 = Vector256<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = (weights + ((long)s * accumulatorSize));
            long i = 0, len = (long)accumulatorSize;
            for (; i + 32 <= len; i += 32)
            {
                var x0 = Vector256.Min(Vector256.Max(Vector256.LoadAligned(a + (i)), zero), max);
                var x1 = Vector256.Min(Vector256.Max(Vector256.LoadAligned(a + (i + 16)), zero), max);
                s0 = Vector256.Add(s0, Avx2.MultiplyAddAdjacent(x0, Vector256.LoadAligned(ws + (i))));
                s1 = Vector256.Add(s1, Avx2.MultiplyAddAdjacent(x1, Vector256.LoadAligned(ws + (i + 16))));
            }
            if (i < len)
            {
                var x0 = Vector256.Min(Vector256.Max(Vector256.LoadAligned(a + (i)), zero), max);
                s0 = Vector256.Add(s0, Avx2.MultiplyAddAdjacent(x0, Vector256.LoadAligned(ws + (i))));
            }
        }
        return Vector256.Sum(Vector256.Add(s0, s1));
    }

    /// <summary>The same for SCReLU: sum of ((c * w) as int16) * c with c = clamp(slot, 0, qa).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        var zero = Vector256<short>.Zero; var max = Vector256.Create(clipMax);
        var s0 = Vector256<int>.Zero; var s1 = Vector256<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = (weights + ((long)s * accumulatorSize));
            long i = 0, len = (long)accumulatorSize;
            for (; i + 32 <= len; i += 32)
            {
                var x0 = Vector256.Min(Vector256.Max(Vector256.LoadAligned(a + (i)), zero), max);
                var x1 = Vector256.Min(Vector256.Max(Vector256.LoadAligned(a + (i + 16)), zero), max);
                s0 = Vector256.Add(s0, Avx2.MultiplyAddAdjacent(x0 * Vector256.LoadAligned(ws + (i)), x0));
                s1 = Vector256.Add(s1, Avx2.MultiplyAddAdjacent(x1 * Vector256.LoadAligned(ws + (i + 16)), x1));
            }
            if (i < len)
            {
                var x0 = Vector256.Min(Vector256.Max(Vector256.LoadAligned(a + (i)), zero), max);
                s0 = Vector256.Add(s0, Avx2.MultiplyAddAdjacent(x0 * Vector256.LoadAligned(ws + (i)), x0));
            }
        }
        return Vector256.Sum(Vector256.Add(s0, s1));
    }

    /// <summary>Last dense layer (one output): bias + sum of input[i] * w[i], contiguous weights. n is a multiple of 64; weights past In are zero
    /// and the input buffer is readable up to n, so the bytes beyond In contribute nothing.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength)
    {
        var ones = Vector256.Create((short)1);
        var a0 = Vector256<int>.Zero; var a1 = Vector256<int>.Zero;
        for (int i = 0; i < paddedInputLength; i += 64)
        {
            a0 = Vector256.Add(a0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Vector256.Load(input + i), Vector256.LoadAligned(weights + i)), ones));
            a1 = Vector256.Add(a1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Vector256.Load(input + i + 32), Vector256.LoadAligned(weights + i + 32)), ones));
        }
        return bias + Vector256.Sum(Vector256.Add(a0, a1));
    }
}
