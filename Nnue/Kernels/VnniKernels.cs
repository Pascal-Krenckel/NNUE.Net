using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Nnue.Backends;

namespace Nnue.Kernels;

/// <summary>AVX-VNNI (vpdpbusd, 256 bit). Generic over the dot operation so that a software model can run through the same code.</summary>
internal static unsafe class VnniKernels
{
    internal interface IVnniOp
    {
        static abstract Vector256<int> MultiplyAccumulate(Vector256<int> accumulator, Vector256<byte> unsignedValues, Vector256<sbyte> signedValues);
    }

    /// <summary>The hardware instruction (preview API in .NET 8, see <see cref="SimdSupport"/>).</summary>
    internal readonly struct VnniHw : IVnniOp
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> MultiplyAccumulate(Vector256<int> accumulator, Vector256<byte> unsignedValues, Vector256<sbyte> signedValues) =>
#if NNUE_NO_AVXVNNI
            throw new PlatformNotSupportedException("Compiled with NNUE_NO_AVXVNNI.");
#else
#pragma warning disable CA2252
            AvxVnni.MultiplyWideningAndAdd(accumulator, unsignedValues, signedValues);
#pragma warning restore CA2252
#endif

    }

    /// <summary>Software model of vpdpbusd on AVX2. Identical results as long as |u*s| pair sums fit int16 (u &lt;= 127, |s| &lt;= 127).</summary>
    internal readonly struct VnniModel : IVnniOp
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> MultiplyAccumulate(Vector256<int> accumulator, Vector256<byte> unsignedValues, Vector256<sbyte> signedValues)
            => Vector256.Add(accumulator, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(unsignedValues, signedValues), Vector256.Create((short)1)));
    }

    /// <summary>
    /// Dense int8 layer. 16 outputs (two 8-lane accumulators) per block, four input chunks per iteration = 8 independent accumulator chains:
    /// vpdpbusd has a latency of 5 cycles and a throughput of 2 per cycle, so it needs about 10 chains in flight. inDim % 16 == 0.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformVnni<TOp>(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
        where TOp : struct, IVnniOp
    {
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            Vector256<int> a0 = Vector256.LoadAligned(biases + ((long)o));
            Vector256<int> a1 = Vector256.LoadAligned(biases + ((long)o + 8));
            Vector256<int> b0 = Vector256<int>.Zero;
            Vector256<int> b1 = Vector256<int>.Zero;
            Vector256<int> c0 = Vector256<int>.Zero;
            Vector256<int> c1 = Vector256<int>.Zero;
            Vector256<int> d0 = Vector256<int>.Zero;
            Vector256<int> d1 = Vector256<int>.Zero;
            sbyte* wp = weights + (o * 4);
            for (int i = 0; i < inputLength; i += 16)
            {
                sbyte* wa = wp + ((i >> 2) * stride);
                sbyte* wb = wa + stride, wc = wa + (2 * stride), wd = wa + (3 * stride);
                Vector256<byte> x0 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + i)).AsByte();
                Vector256<byte> x1 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + i + 4)).AsByte();
                Vector256<byte> x2 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + i + 8)).AsByte();
                Vector256<byte> x3 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + i + 12)).AsByte();
                a0 = TOp.MultiplyAccumulate(a0, x0, Vector256.LoadAligned(wa));
                a1 = TOp.MultiplyAccumulate(a1, x0, Vector256.LoadAligned(wa + 32));
                b0 = TOp.MultiplyAccumulate(b0, x1, Vector256.LoadAligned(wb));
                b1 = TOp.MultiplyAccumulate(b1, x1, Vector256.LoadAligned(wb + 32));
                c0 = TOp.MultiplyAccumulate(c0, x2, Vector256.LoadAligned(wc));
                c1 = TOp.MultiplyAccumulate(c1, x2, Vector256.LoadAligned(wc + 32));
                d0 = TOp.MultiplyAccumulate(d0, x3, Vector256.LoadAligned(wd));
                d1 = TOp.MultiplyAccumulate(d1, x3, Vector256.LoadAligned(wd + 32));
            }
            Vector256.Add(Vector256.Add(a0, b0), Vector256.Add(c0, d0)).StoreAligned(output + ((long)o));
            Vector256.Add(Vector256.Add(a1, b1), Vector256.Add(c1, d1)).StoreAligned(output + ((long)o + 8));
        }
    }

    /// <summary>Sparse variant: only the non-zero chunks listed in idx, four per iteration.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformSparseVnni<TOp>(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
        where TOp : struct, IVnniOp
    {
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            Vector256<int> a0 = Vector256.LoadAligned(biases + ((long)o));
            Vector256<int> a1 = Vector256.LoadAligned(biases + ((long)o + 8));
            Vector256<int> b0 = Vector256<int>.Zero;
            Vector256<int> b1 = Vector256<int>.Zero;
            Vector256<int> c0 = Vector256<int>.Zero;
            Vector256<int> c1 = Vector256<int>.Zero;
            Vector256<int> d0 = Vector256<int>.Zero;
            Vector256<int> d1 = Vector256<int>.Zero;
            sbyte* wp = weights + (o * 4);
            int j = 0;
            for (; j + 4 <= nonZeroCount; j += 4)
            {
                int k0 = nonZeroChunks[j], k1 = nonZeroChunks[j + 1], k2 = nonZeroChunks[j + 2], k3 = nonZeroChunks[j + 3];
                sbyte* wa = wp + (k0 * stride);
                sbyte* wb = wp + (k1 * stride);
                sbyte* wc = wp + (k2 * stride);
                sbyte* wd = wp + (k3 * stride);
                Vector256<byte> x0 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + (k0 * 4))).AsByte();
                Vector256<byte> x1 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + (k1 * 4))).AsByte();
                Vector256<byte> x2 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + (k2 * 4))).AsByte();
                Vector256<byte> x3 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + (k3 * 4))).AsByte();
                a0 = TOp.MultiplyAccumulate(a0, x0, Vector256.LoadAligned(wa));
                a1 = TOp.MultiplyAccumulate(a1, x0, Vector256.LoadAligned(wa + 32));
                b0 = TOp.MultiplyAccumulate(b0, x1, Vector256.LoadAligned(wb));
                b1 = TOp.MultiplyAccumulate(b1, x1, Vector256.LoadAligned(wb + 32));
                c0 = TOp.MultiplyAccumulate(c0, x2, Vector256.LoadAligned(wc));
                c1 = TOp.MultiplyAccumulate(c1, x2, Vector256.LoadAligned(wc + 32));
                d0 = TOp.MultiplyAccumulate(d0, x3, Vector256.LoadAligned(wd));
                d1 = TOp.MultiplyAccumulate(d1, x3, Vector256.LoadAligned(wd + 32));
            }
            for (; j < nonZeroCount; j++)
            {
                int k0 = nonZeroChunks[j];
                sbyte* wa = wp + (k0 * stride);
                Vector256<byte> x0 = Vector256.Create(Unsafe.ReadUnaligned<int>(input + (k0 * 4))).AsByte();
                a0 = TOp.MultiplyAccumulate(a0, x0, Vector256.LoadAligned(wa));
                a1 = TOp.MultiplyAccumulate(a1, x0, Vector256.LoadAligned(wa + 32));
            }
            Vector256.Add(Vector256.Add(a0, b0), Vector256.Add(c0, d0)).StoreAligned(output + ((long)o));
            Vector256.Add(Vector256.Add(a1, b1), Vector256.Add(c1, d1)).StoreAligned(output + ((long)o + 8));
        }
    }

    /// <summary>Last dense layer (one output): bias + sum of input[i] * w[i], contiguous weights. n is a multiple of 64; weights past In are zero
    /// and the input buffer is readable up to n, so the bytes beyond In contribute nothing.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int AffineTransformSingleOutputVnni<TOp>(byte* input, sbyte* weights, int bias, int paddedInputLength) where TOp : struct, IVnniOp
    {
        Vector256<int> a0 = Vector256<int>.Zero;
        Vector256<int> a1 = Vector256<int>.Zero;
        for (int i = 0; i < paddedInputLength; i += 64)
        {
            a0 = TOp.MultiplyAccumulate(a0, Vector256.Load(input + i), Vector256.LoadAligned(weights + i));
            a1 = TOp.MultiplyAccumulate(a1, Vector256.Load(input + i + 32), Vector256.LoadAligned(weights + i + 32));
        }
        return bias + Vector256.Sum(Vector256.Add(a0, a1));
    }
}
