using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Nnue.Kernels;

/// <summary>Plain Vector128 kernels. They compile to SSE on x86 and to NEON on AArch64, so both backends share them.</summary>
internal static unsafe class Portable128Kernels
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AddRowToAccumulator(short* row, short* accumulator, int length)
    {
        for (long i = 0; i < length; i += 8)
            Vector128.Add(Vector128.LoadAligned(accumulator + i), Vector128.LoadAligned(row + i)).StoreAligned(accumulator + i);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void SubtractRowFromAccumulator(short* row, short* accumulator, int length)
    {
        for (long i = 0; i < length; i += 8)
            Vector128.Subtract(Vector128.LoadAligned(accumulator + i), Vector128.LoadAligned(row + i)).StoreAligned(accumulator + i);
    }

    /// <summary>
    /// accumulator = bias + the rows of all active features. The accumulator is processed in tiles of 8 vectors (64 values) held in
    /// independent registers, so the adds of one feature row do not wait for each other (measured: 20-40 % faster than one vector per pass
    /// while the feature matrix is in cache, equal when it is not); the tail of sizes that are not a multiple of 64 goes one vector at a time.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void RefreshAccumulator(short* featureWeights, short* featureBias, ReadOnlySpan<int> activeFeatures, short* accumulator, int accumulatorSize)
    {
        long i = 0, size = accumulatorSize;
        for (; i + 64 <= size; i += 64)
        {
            Vector128<short> a0 = Vector128.LoadAligned(featureBias + i);
            Vector128<short> a1 = Vector128.LoadAligned(featureBias + i + 8);
            Vector128<short> a2 = Vector128.LoadAligned(featureBias + i + 16);
            Vector128<short> a3 = Vector128.LoadAligned(featureBias + i + 24);
            Vector128<short> a4 = Vector128.LoadAligned(featureBias + i + 32);
            Vector128<short> a5 = Vector128.LoadAligned(featureBias + i + 40);
            Vector128<short> a6 = Vector128.LoadAligned(featureBias + i + 48);
            Vector128<short> a7 = Vector128.LoadAligned(featureBias + i + 56);
            foreach (int f in activeFeatures)
            {
                short* row = featureWeights + (f * size) + i;
                a0 += Vector128.LoadAligned(row);
                a1 += Vector128.LoadAligned(row + 8);
                a2 += Vector128.LoadAligned(row + 16);
                a3 += Vector128.LoadAligned(row + 24);
                a4 += Vector128.LoadAligned(row + 32);
                a5 += Vector128.LoadAligned(row + 40);
                a6 += Vector128.LoadAligned(row + 48);
                a7 += Vector128.LoadAligned(row + 56);
            }
            a0.StoreAligned(accumulator + i);
            a1.StoreAligned(accumulator + i + 8);
            a2.StoreAligned(accumulator + i + 16);
            a3.StoreAligned(accumulator + i + 24);
            a4.StoreAligned(accumulator + i + 32);
            a5.StoreAligned(accumulator + i + 40);
            a6.StoreAligned(accumulator + i + 48);
            a7.StoreAligned(accumulator + i + 56);
        }
        for (; i < size; i += 8)
        {
            Vector128<short> acc = Vector128.LoadAligned(featureBias + i);
            foreach (int f in activeFeatures)
                acc += Vector128.LoadAligned(featureWeights + (f * size) + i);
            acc.StoreAligned(accumulator + i);
        }
    }

    /// <summary>destination = source - removed rows + added rows, tiled like <see cref="RefreshAccumulator"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void UpdateAccumulator(short* featureWeights, short* source, ReadOnlySpan<int> removedFeatures, ReadOnlySpan<int> addedFeatures, short* destination, int accumulatorSize)
    {
        long i = 0, size = accumulatorSize;
        for (; i + 64 <= size; i += 64)
        {
            Vector128<short> a0 = Vector128.LoadAligned(source + i);
            Vector128<short> a1 = Vector128.LoadAligned(source + i + 8);
            Vector128<short> a2 = Vector128.LoadAligned(source + i + 16);
            Vector128<short> a3 = Vector128.LoadAligned(source + i + 24);
            Vector128<short> a4 = Vector128.LoadAligned(source + i + 32);
            Vector128<short> a5 = Vector128.LoadAligned(source + i + 40);
            Vector128<short> a6 = Vector128.LoadAligned(source + i + 48);
            Vector128<short> a7 = Vector128.LoadAligned(source + i + 56);
            foreach (int f in removedFeatures)
            {
                short* row = featureWeights + (f * size) + i;
                a0 -= Vector128.LoadAligned(row);
                a1 -= Vector128.LoadAligned(row + 8);
                a2 -= Vector128.LoadAligned(row + 16);
                a3 -= Vector128.LoadAligned(row + 24);
                a4 -= Vector128.LoadAligned(row + 32);
                a5 -= Vector128.LoadAligned(row + 40);
                a6 -= Vector128.LoadAligned(row + 48);
                a7 -= Vector128.LoadAligned(row + 56);
            }
            foreach (int f in addedFeatures)
            {
                short* row = featureWeights + (f * size) + i;
                a0 += Vector128.LoadAligned(row);
                a1 += Vector128.LoadAligned(row + 8);
                a2 += Vector128.LoadAligned(row + 16);
                a3 += Vector128.LoadAligned(row + 24);
                a4 += Vector128.LoadAligned(row + 32);
                a5 += Vector128.LoadAligned(row + 40);
                a6 += Vector128.LoadAligned(row + 48);
                a7 += Vector128.LoadAligned(row + 56);
            }
            a0.StoreAligned(destination + i);
            a1.StoreAligned(destination + i + 8);
            a2.StoreAligned(destination + i + 16);
            a3.StoreAligned(destination + i + 24);
            a4.StoreAligned(destination + i + 32);
            a5.StoreAligned(destination + i + 40);
            a6.StoreAligned(destination + i + 48);
            a7.StoreAligned(destination + i + 56);
        }
        for (; i < size; i += 8)
        {
            Vector128<short> acc = Vector128.LoadAligned(source + i);
            foreach (int f in removedFeatures)
                acc -= Vector128.LoadAligned(featureWeights + (f * size) + i);
            foreach (int f in addedFeatures)
                acc += Vector128.LoadAligned(featureWeights + (f * size) + i);
            acc.StoreAligned(destination + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyCReLU(short* accumulator, byte* output, int length)
    {
        Vector128<short> zero = Vector128<short>.Zero;
        Vector128<short> max = Vector128.Create((short)127);
        long i = 0, len = length;
        for (; i + 16 <= len; i += 16)
        {
            Vector128<short> a = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + i), zero), max);
            Vector128<short> c = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + (i + 8)), zero), max);
            Vector128.Narrow(a.AsUInt16(), c.AsUInt16()).Store(output + i);
        }
        if (i < len)
        {
            Vector128<short> a = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + i), zero), max);
            Vector128.Narrow(a.AsUInt16(), Vector128<ushort>.Zero).GetLower().Store(output + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<ushort> PairProduct128(short* accumulator, long offset, long halfLength, Vector128<short> zero, Vector128<short> max)
    {
        Vector128<ushort> a = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + offset), zero), max).AsUInt16();
        Vector128<ushort> c = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + (offset + halfLength)), zero), max).AsUInt16();
        return Vector128.ShiftRightLogical(a * c, 9);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyPairwise(short* accumulator, byte* output, int halfLength)
    {
        Vector128<short> zero = Vector128<short>.Zero;
        Vector128<short> max = Vector128.Create((short)255);
        long i = 0, len = halfLength;
        for (; i + 16 <= len; i += 16)
            Vector128.Narrow(PairProduct128(accumulator, i, len, zero, max), PairProduct128(accumulator, i + 8, len, zero, max)).Store(output + i);
        if (i < len)
            Vector128.Narrow(PairProduct128(accumulator, i, len, zero, max), Vector128<ushort>.Zero).GetLower().Store(output + i);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<ushort> Pool128(short* accumulator, long offset, long halfLength, Vector128<short> zero, Vector128<short> max, bool average)
    {
        var a = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + (offset)), zero), max).AsUInt16();
        var c = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + (offset + halfLength)), zero), max).AsUInt16();
        return average ? Vector128.ShiftRightLogical(a + c + Vector128<ushort>.One, 1) : Vector128.Max(a, c);
    }

    /// <summary>Max / average pooling of the two halves of an accumulator, both clamped to [0, 127] (halfLength % 32 == 0).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplyPooling(short* accumulator, byte* output, int halfLength, bool average)
    {
        var zero = Vector128<short>.Zero; var max = Vector128.Create((short)127);
        long i = 0, len = (long)halfLength;
        for (; i + 16 <= len; i += 16)
            Vector128.Narrow(Pool128(accumulator, i, len, zero, max, average), Pool128(accumulator, i + 8, len, zero, max, average)).Store(output + (i));
        if (i < len)
            Vector128.Narrow(Pool128(accumulator, i, len, zero, max, average), Vector128<ushort>.Zero).GetLower().Store(output + (i));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<int> Quantmoid4Lane(Vector128<int> z, Vector128<int> v127, Vector128<int> v126)
    {
        var x = Vector128.ShiftRightArithmetic(z, 6);
        var v = v127 - Vector128.Min(Vector128.Abs(x), v127);
        var q = Vector128.ShiftRightArithmetic(v * v, 8);
        return Vector128.ConditionalSelect(Vector128.LessThan(x, Vector128<int>.Zero), q, v126 - q);
    }

    /// <summary>Quantmoid4 hidden activation (see <see cref="ScalarKernels.ActivateQuantmoid4"/>), n % 16 == 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ActivateQuantmoid4(int* source, byte* destination, int length)
    {
        var v127 = Vector128.Create(127); var v126 = Vector128.Create(126);
        for (long i = 0; i < (long)length; i += 16)
        {
            var a = Quantmoid4Lane(Vector128.LoadAligned(source + (i)), v127, v126);
            var c = Quantmoid4Lane(Vector128.LoadAligned(source + (i + 4)), v127, v126);
            var d = Quantmoid4Lane(Vector128.LoadAligned(source + (i + 8)), v127, v126);
            var e = Quantmoid4Lane(Vector128.LoadAligned(source + (i + 12)), v127, v126);
            Vector128.Narrow(Vector128.Narrow(a, c).AsUInt16(), Vector128.Narrow(d, e).AsUInt16()).Store(destination + (i));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ApplySCReLU(short* accumulator, byte* output, int length)
    {
        Vector128<short> zero = Vector128<short>.Zero;
        Vector128<short> max = Vector128.Create((short)255);
        long i = 0, len = length;
        for (; i + 16 <= len; i += 16)
        {
            Vector128<ushort> a = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + i), zero), max).AsUInt16();
            Vector128<ushort> c = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + (i + 8)), zero), max).AsUInt16();
            Vector128.Narrow(Vector128.ShiftRightLogical(a * a, 9), Vector128.ShiftRightLogical(c * c, 9)).Store(output + i);
        }
        if (i < len)
        {
            Vector128<ushort> a = Vector128.Min(Vector128.Max(Vector128.LoadAligned(accumulator + i), zero), max).AsUInt16();
            Vector128.Narrow(Vector128.ShiftRightLogical(a * a, 9), Vector128<ushort>.Zero).GetLower().Store(output + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int CountNonZeroChunks(int* input, int chunkCount)
    {
        int n = 0;
        for (int c = 0; c < chunkCount; c += 4)
            n += BitOperations.PopCount(~Vector128.Equals(Vector128.LoadAligned(input + ((long)c)), Vector128<int>.Zero).ExtractMostSignificantBits() & 0xFu);
        return n;
    }

    /// <summary>Index list of the non-zero 4-byte chunks. Writes up to 4 values past the last index (the caller pads the buffer by 16 ints).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int FindNonZeroChunks(byte* input, int* nonZeroChunks, int inputLength)
    {
        int count = 0, chunks = inputLength >> 2;
        int* src = (int*)input;
        uint* lut = NnzTable.Lut;
        for (int c = 0; c < chunks; c += 4)
        {
            uint m = ~Vector128.Equals(Vector128.LoadAligned(src + c), Vector128<int>.Zero).ExtractMostSignificantBits() & 0xFu;
            Vector128.Add(Vector128.LoadAligned(lut + (m * 8)), Vector128.Create((uint)c)).AsInt32().Store(nonZeroChunks + count);
            count += BitOperations.PopCount(m);
        }
        return count;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<int> SquareClippedLane(Vector128<int> v, Vector128<int> lo, Vector128<int> hi, Vector128<int> cap)
    {
        Vector128<int> x = Vector128.Min(Vector128.Max(v, lo), hi);
        return Vector128.Min(Vector128.ShiftRightLogical(x * x, 19), cap);
    }

    /// <summary>SqrClippedReLU without the x86 pack/mulhi tricks: clamp to int16 range, square in int32, >> 19, cap at 127.
    /// Identical to min(127, z*z >> 19): for |z| > 32768 the exact result is >= 127 anyway.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ActivateSqrClippedReLU(int* source, byte* destination, int length)
    {
        Vector128<int> lo = Vector128.Create(-32768);
        Vector128<int> hi = Vector128.Create(32767);
        Vector128<int> cap = Vector128.Create(127);
        for (long i = 0; i < length; i += 16)
        {
            Vector128<int> a = SquareClippedLane(Vector128.LoadAligned(source + i), lo, hi, cap);
            Vector128<int> c = SquareClippedLane(Vector128.LoadAligned(source + (i + 4)), lo, hi, cap);
            Vector128<int> d = SquareClippedLane(Vector128.LoadAligned(source + (i + 8)), lo, hi, cap);
            Vector128<int> e = SquareClippedLane(Vector128.LoadAligned(source + (i + 12)), lo, hi, cap);
            Vector128.Narrow(Vector128.Narrow(a, c).AsUInt16(), Vector128.Narrow(d, e).AsUInt16()).Store(destination + i);
        }
    }

    /// <summary>Hidden activation: dst[i] = clamp(z[i] >> 6, 0, 127), n % 16 == 0.</summary>
    internal static void ActivateClippedReLU(int* source, byte* destination, int length)
    {
        Vector128<int> zero = Vector128<int>.Zero;
        Vector128<int> max = Vector128.Create(127);
        for (long i = 0; i < length; i += 16)
        {
            Vector128<int> a = Vector128.Min(Vector128.Max(Vector128.ShiftRightArithmetic(Vector128.LoadAligned(source + i), 6), zero), max);
            Vector128<int> c = Vector128.Min(Vector128.Max(Vector128.ShiftRightArithmetic(Vector128.LoadAligned(source + (i + 4)), 6), zero), max);
            Vector128<int> d = Vector128.Min(Vector128.Max(Vector128.ShiftRightArithmetic(Vector128.LoadAligned(source + (i + 8)), 6), zero), max);
            Vector128<int> e = Vector128.Min(Vector128.Max(Vector128.ShiftRightArithmetic(Vector128.LoadAligned(source + (i + 12)), 6), zero), max);
            Vector128<short> s0 = Vector128.Narrow(a, c);
            Vector128<short> s1 = Vector128.Narrow(d, e);
            Vector128.Narrow(s0.AsUInt16(), s1.AsUInt16()).Store(destination + i);
        }
    }
}
