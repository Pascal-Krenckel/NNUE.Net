using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Nnue.Kernels;

/// <summary>
/// AVX512-VNNI (<c>vpdpbusd zmm</c>, 64 u8 x i8 products into 16 int32 lanes per instruction) for the int8 dense layers. The instruction is
/// reachable from .NET 11 on (<c>AvxVnni.V512</c>); the kernels are generic over the multiply-accumulate so that a software model
/// (<c>vpmaddubsw</c> + <c>vpmaddwd</c>, bit-identical for the value ranges this library produces) runs the same code on any AVX-512 CPU and
/// on older runtimes.
/// </summary>
internal static unsafe class Avx512VnniKernels
{
    internal interface IVnni512Op
    {
        static abstract Vector512<int> MultiplyAccumulate(Vector512<int> accumulator, Vector512<byte> unsignedValues, Vector512<sbyte> signedValues);
    }

    /// <summary>The hardware instruction (.NET 11 and newer).</summary>
    internal readonly struct Vnni512Hardware : IVnni512Op
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector512<int> MultiplyAccumulate(Vector512<int> accumulator, Vector512<byte> unsignedValues, Vector512<sbyte> signedValues) =>
#if NET11_0_OR_GREATER
            AvxVnni.V512.MultiplyWideningAndAdd(accumulator, unsignedValues, signedValues);
#else
            throw new PlatformNotSupportedException("AVX512-VNNI needs the net11.0 build of the library.");
#endif

    }

    /// <summary>Software model of <c>vpdpbusd zmm</c> on AVX-512 BW. Identical results as long as the pair sums of u*s fit int16 (u &lt;= 127, |s| &lt;= 127).</summary>
    internal readonly struct Vnni512Model : IVnni512Op
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector512<int> MultiplyAccumulate(Vector512<int> accumulator, Vector512<byte> unsignedValues, Vector512<sbyte> signedValues)
            => Vector512.Add(accumulator, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(unsignedValues, signedValues), Vector512.Create((short)1)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<byte> Broadcast(byte* input, int offset) => Vector512.Create(Unsafe.ReadUnaligned<int>(input + offset)).AsByte();

    /// <summary>
    /// Dense int8 layer: 16 outputs per 512-bit block. vpdpbusd has a latency of about 5 cycles at a throughput of 1-2 per cycle, so 8 input
    /// chunks (32 inputs) feed 8 independent accumulators per iteration; a tail of 4 chunks handles input lengths that are an odd multiple of 16.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransform<TOp>(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
        where TOp : struct, IVnni512Op
    {
        int stride = paddedOutputLength * 4;                       // bytes between consecutive 4-input chunks
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            Vector512<int> a0 = Vector512.LoadAligned(biases + (long)o);
            Vector512<int> a1 = default, a2 = default, a3 = default, a4 = default, a5 = default, a6 = default, a7 = default;
            sbyte* wp = weights + (o * 4);
            int i = 0;
            for (; i + 32 <= inputLength; i += 32)
            {
                sbyte* w = wp + ((i >> 2) * stride);
                a0 = TOp.MultiplyAccumulate(a0, Broadcast(input, i), Vector512.LoadAligned(w));
                a1 = TOp.MultiplyAccumulate(a1, Broadcast(input, i + 4), Vector512.LoadAligned(w + stride));
                a2 = TOp.MultiplyAccumulate(a2, Broadcast(input, i + 8), Vector512.LoadAligned(w + (2 * stride)));
                a3 = TOp.MultiplyAccumulate(a3, Broadcast(input, i + 12), Vector512.LoadAligned(w + (3 * stride)));
                a4 = TOp.MultiplyAccumulate(a4, Broadcast(input, i + 16), Vector512.LoadAligned(w + (4 * stride)));
                a5 = TOp.MultiplyAccumulate(a5, Broadcast(input, i + 20), Vector512.LoadAligned(w + (5 * stride)));
                a6 = TOp.MultiplyAccumulate(a6, Broadcast(input, i + 24), Vector512.LoadAligned(w + (6 * stride)));
                a7 = TOp.MultiplyAccumulate(a7, Broadcast(input, i + 28), Vector512.LoadAligned(w + (7 * stride)));
            }
            if (i < inputLength)                                   // 16 inputs left (inputLength % 16 == 0)
            {
                sbyte* w = wp + ((i >> 2) * stride);
                a0 = TOp.MultiplyAccumulate(a0, Broadcast(input, i), Vector512.LoadAligned(w));
                a1 = TOp.MultiplyAccumulate(a1, Broadcast(input, i + 4), Vector512.LoadAligned(w + stride));
                a2 = TOp.MultiplyAccumulate(a2, Broadcast(input, i + 8), Vector512.LoadAligned(w + (2 * stride)));
                a3 = TOp.MultiplyAccumulate(a3, Broadcast(input, i + 12), Vector512.LoadAligned(w + (3 * stride)));
            }
            (a0 + a1 + (a2 + a3) + (a4 + a5 + (a6 + a7))).StoreAligned(output + (long)o);
        }
    }

    /// <summary>Sparse variant: only the non-zero 4-byte chunks listed in <paramref name="nonZeroChunks"/>, eight per iteration.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformSparse<TOp>(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
        where TOp : struct, IVnni512Op
    {
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            Vector512<int> a0 = Vector512.LoadAligned(biases + (long)o);
            Vector512<int> a1 = default, a2 = default, a3 = default, a4 = default, a5 = default, a6 = default, a7 = default;
            sbyte* wp = weights + (o * 4);
            int j = 0;
            for (; j + 8 <= nonZeroCount; j += 8)
            {
                int* c = nonZeroChunks + j;
                a0 = TOp.MultiplyAccumulate(a0, Broadcast(input, c[0] * 4), Vector512.LoadAligned(wp + (c[0] * stride)));
                a1 = TOp.MultiplyAccumulate(a1, Broadcast(input, c[1] * 4), Vector512.LoadAligned(wp + (c[1] * stride)));
                a2 = TOp.MultiplyAccumulate(a2, Broadcast(input, c[2] * 4), Vector512.LoadAligned(wp + (c[2] * stride)));
                a3 = TOp.MultiplyAccumulate(a3, Broadcast(input, c[3] * 4), Vector512.LoadAligned(wp + (c[3] * stride)));
                a4 = TOp.MultiplyAccumulate(a4, Broadcast(input, c[4] * 4), Vector512.LoadAligned(wp + (c[4] * stride)));
                a5 = TOp.MultiplyAccumulate(a5, Broadcast(input, c[5] * 4), Vector512.LoadAligned(wp + (c[5] * stride)));
                a6 = TOp.MultiplyAccumulate(a6, Broadcast(input, c[6] * 4), Vector512.LoadAligned(wp + (c[6] * stride)));
                a7 = TOp.MultiplyAccumulate(a7, Broadcast(input, c[7] * 4), Vector512.LoadAligned(wp + (c[7] * stride)));
            }
            for (; j < nonZeroCount; j++)
            {
                int c0 = nonZeroChunks[j];
                a0 = TOp.MultiplyAccumulate(a0, Broadcast(input, c0 * 4), Vector512.LoadAligned(wp + (c0 * stride)));
            }
            (a0 + a1 + (a2 + a3) + (a4 + a5 + (a6 + a7))).StoreAligned(output + (long)o);
        }
    }

    /// <summary>Last dense layer (one output): bias + sum of input[i] * w[i], contiguous weights. <paramref name="paddedInputLength"/> is a multiple
    /// of 64; weights past the real input length are zero and the input buffer is readable up to it.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int AffineTransformSingleOutput<TOp>(byte* input, sbyte* weights, int bias, int paddedInputLength) where TOp : struct, IVnni512Op
    {
        Vector512<int> a0 = default, a1 = default;
        int i = 0;
        for (; i + 128 <= paddedInputLength; i += 128)
        {
            a0 = TOp.MultiplyAccumulate(a0, Vector512.Load(input + i), Vector512.LoadAligned(weights + i));
            a1 = TOp.MultiplyAccumulate(a1, Vector512.Load(input + i + 64), Vector512.LoadAligned(weights + i + 64));
        }
        if (i < paddedInputLength)
            a0 = TOp.MultiplyAccumulate(a0, Vector512.Load(input + i), Vector512.LoadAligned(weights + i));
        return bias + Vector512.Sum(a0 + a1);
    }
}
