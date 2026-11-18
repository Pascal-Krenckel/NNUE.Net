using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Nnue.Kernels;

/// <summary>x86 kernels for the multiply-accumulate parts at 128 bit (SSSE3: pmaddwd, pmaddubsw, pack/mulhi).</summary>
internal static unsafe class SseKernels
{

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
    {
        // 4 input chunks (16 inputs) per iteration with 4 independent accumulators: the latency of
        // pmaddubsw -> pmaddwd -> paddd is hidden instead of serializing every chunk. inputLength % 16 == 0 is guaranteed.
        var ones = Vector128.Create((short)1);
        int stride = paddedOutputLength * 4;                       // bytes between consecutive 4-input chunks
        for (int o = 0; o < paddedOutputLength; o += 4)
        {
            var a0 = Vector128.LoadAligned(biases + ((long)o));
            var a1 = Vector128<int>.Zero;
            var a2 = Vector128<int>.Zero;
            var a3 = Vector128<int>.Zero;
            sbyte* wp = (weights + (o * 4));
            for (int i = 0; i < inputLength; i += 16)
            {
                sbyte* wc = (wp + ((i >> 2) * stride));
                var i0 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (i)))).AsByte();
                var i1 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (i + 4)))).AsByte();
                var i2 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (i + 8)))).AsByte();
                var i3 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (i + 12)))).AsByte();
                a0 = Vector128.Add(a0, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i0, Vector128.LoadAligned(wc)), ones));
                a1 = Vector128.Add(a1, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i1, Vector128.LoadAligned((wc + (stride)))), ones));
                a2 = Vector128.Add(a2, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i2, Vector128.LoadAligned((wc + (2 * stride)))), ones));
                a3 = Vector128.Add(a3, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i3, Vector128.LoadAligned((wc + (3 * stride)))), ones));
            }
            Vector128.Add(Vector128.Add(a0, a1), Vector128.Add(a2, a3)).StoreAligned(output + ((long)o));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
    {
        var ones = Vector128.Create((short)1);
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 4)
        {
            var a0 = Vector128.LoadAligned(biases + ((long)o));
            var a1 = Vector128<int>.Zero;
            var a2 = Vector128<int>.Zero;
            var a3 = Vector128<int>.Zero;
            sbyte* wp = (weights + (o * 4));
            int j = 0;
            for (; j + 4 <= nonZeroCount; j += 4)
            {
                int c0 = nonZeroChunks[j], c1 = nonZeroChunks[j + 1], c2 = nonZeroChunks[j + 2], c3 = nonZeroChunks[j + 3];
                var i0 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (c0 * 4)))).AsByte();
                var i1 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (c1 * 4)))).AsByte();
                var i2 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (c2 * 4)))).AsByte();
                var i3 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (c3 * 4)))).AsByte();
                a0 = Vector128.Add(a0, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i0, Vector128.LoadAligned((wp + (c0 * stride)))), ones));
                a1 = Vector128.Add(a1, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i1, Vector128.LoadAligned((wp + (c1 * stride)))), ones));
                a2 = Vector128.Add(a2, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i2, Vector128.LoadAligned((wp + (c2 * stride)))), ones));
                a3 = Vector128.Add(a3, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i3, Vector128.LoadAligned((wp + (c3 * stride)))), ones));
            }
            for (; j < nonZeroCount; j++)
            {
                int c0 = nonZeroChunks[j];
                var i0 = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (c0 * 4)))).AsByte();
                a0 = Vector128.Add(a0, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(i0, Vector128.LoadAligned((wp + (c0 * stride)))), ones));
            }
            Vector128.Add(Vector128.Add(a0, a1), Vector128.Add(a2, a3)).StoreAligned(output + ((long)o));
        }
    }

    /// <summary>
    /// SqrClippedReLU: dst[i] = min(127, z^2 >> 19), as in Stockfish: packs_epi32 -> mulhi_epi16(x, x) -> srli 3 -> packs_epi16.
    /// An int32 input beyond +-32767 saturates to a result >= 2047, which the final clamp turns into 127 - identical to the exact formula.
    /// </summary>
    internal static void ActivateSqrClippedReLU(int* source, byte* destination, int length)
    {
        for (long i = 0; i < (long)length; i += 16)
        {
            var w0 = Sse2.PackSignedSaturate(Vector128.LoadAligned(source + (i)), Vector128.LoadAligned(source + (i + 4)));
            var w1 = Sse2.PackSignedSaturate(Vector128.LoadAligned(source + (i + 8)), Vector128.LoadAligned(source + (i + 12)));
            var s0 = Sse2.ShiftRightLogical(Sse2.MultiplyHigh(w0, w0), 3);
            var s1 = Sse2.ShiftRightLogical(Sse2.MultiplyHigh(w1, w1), 3);
            Sse2.PackSignedSaturate(s0, s1).AsByte().Store(destination + (i));
        }
    }

    /// <summary>
    /// Int16 output layer over all slots: sum over the slots s of clamp(slot_s, 0, qa) * w[s*n .. s*n+n).
    /// The products of all slots accumulate in the same vector registers, there is a single horizontal reduction at the end.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        var zero = Vector128<short>.Zero; var max = Vector128.Create(clipMax);
        var s0 = Vector128<int>.Zero; var s1 = Vector128<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = (weights + ((long)s * accumulatorSize));
            long i = 0, len = (long)accumulatorSize;
            for (; i + 16 <= len; i += 16)
            {
                var x0 = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i)), zero), max);
                var x1 = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i + 8)), zero), max);
                s0 = Vector128.Add(s0, Sse2.MultiplyAddAdjacent(x0, Vector128.LoadAligned(ws + (i))));
                s1 = Vector128.Add(s1, Sse2.MultiplyAddAdjacent(x1, Vector128.LoadAligned(ws + (i + 8))));
            }
            if (i < len)
            {
                var x0 = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i)), zero), max);
                s0 = Vector128.Add(s0, Sse2.MultiplyAddAdjacent(x0, Vector128.LoadAligned(ws + (i))));
            }
        }
        return Vector128.Sum(Vector128.Add(s0, s1));
    }

    /// <summary>The same for SCReLU: sum of ((c * w) as int16) * c with c = clamp(slot, 0, qa).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        var zero = Vector128<short>.Zero; var max = Vector128.Create(clipMax);
        var s0 = Vector128<int>.Zero; var s1 = Vector128<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = (weights + ((long)s * accumulatorSize));
            long i = 0, len = (long)accumulatorSize;
            for (; i + 16 <= len; i += 16)
            {
                var x0 = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i)), zero), max);
                var x1 = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i + 8)), zero), max);
                s0 = Vector128.Add(s0, Sse2.MultiplyAddAdjacent(x0 * Vector128.LoadAligned(ws + (i)), x0));
                s1 = Vector128.Add(s1, Sse2.MultiplyAddAdjacent(x1 * Vector128.LoadAligned(ws + (i + 8)), x1));
            }
            if (i < len)
            {
                var x0 = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i)), zero), max);
                s0 = Vector128.Add(s0, Sse2.MultiplyAddAdjacent(x0 * Vector128.LoadAligned(ws + (i)), x0));
            }
        }
        return Vector128.Sum(Vector128.Add(s0, s1));
    }

    /// <summary>Last dense layer (one output): bias + sum of input[i] * w[i], contiguous weights. n is a multiple of 64; weights past In are zero
    /// and the input buffer is readable up to n, so the bytes beyond In contribute nothing.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength)
    {
        var ones = Vector128.Create((short)1);
        var a0 = Vector128<int>.Zero; var a1 = Vector128<int>.Zero;
        for (int i = 0; i < paddedInputLength; i += 32)
        {
            a0 = Vector128.Add(a0, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(Vector128.Load(input + i), Vector128.LoadAligned(weights + i)), ones));
            a1 = Vector128.Add(a1, Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(Vector128.Load(input + i + 16), Vector128.LoadAligned(weights + i + 16)), ones));
        }
        return bias + Vector128.Sum(Vector128.Add(a0, a1));
    }
}
