using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Nnue.Kernels;

/// <summary>AArch64 NEON (AdvSimd) multiply-accumulate kernels; NeonDotProd uses SDOT.</summary>
internal static unsafe class NeonKernels
{

    /// <summary>
    /// 4 outputs x 4 inputs of one chunk: smull (low 8 bytes) + smull2 (high 8 bytes) give 16 int16 products, addp sums neighbours
    /// (two products per output), sadalp adds the pairs into the int32 accumulator lane of that output.
    /// The activations are u8 &lt;= 127, so reading them as int8 changes nothing; a pair sum is at most 2*127*127 and cannot overflow.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<int> MultiplyAccumulateNeon(Vector128<int> accumulator, Vector128<sbyte> weights, Vector128<sbyte> inputs, Vector64<sbyte> inputsLow)
        => AdvSimd.AddPairwiseWideningAndAdd(accumulator,
            AdvSimd.Arm64.AddPairwise(AdvSimd.MultiplyWideningLower(weights.GetLower(), inputsLow), AdvSimd.MultiplyWideningUpper(weights, inputs)));

    /// <summary>Dense int8 layer. 16 outputs (4 accumulators) per block share one broadcast of the 4 input bytes of a chunk.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransform(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
    {
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            var a0 = Vector128.LoadAligned(biases + ((long)o));
            var a1 = Vector128.LoadAligned(biases + ((long)o + 4));
            var a2 = Vector128.LoadAligned(biases + ((long)o + 8));
            var a3 = Vector128.LoadAligned(biases + ((long)o + 12));
            sbyte* wp = (weights + (o * 4));
            for (int i = 0; i < inputLength; i += 4)
            {
                var x = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (i)))).AsSByte();
                var xl = x.GetLower();
                sbyte* wc = (wp + ((i >> 2) * stride));
                a0 = MultiplyAccumulateNeon(a0, Vector128.LoadAligned(wc), x, xl);
                a1 = MultiplyAccumulateNeon(a1, Vector128.LoadAligned(wc + (16)), x, xl);
                a2 = MultiplyAccumulateNeon(a2, Vector128.LoadAligned(wc + (32)), x, xl);
                a3 = MultiplyAccumulateNeon(a3, Vector128.LoadAligned(wc + (48)), x, xl);
            }
            a0.StoreAligned(output + ((long)o));
            a1.StoreAligned(output + ((long)o + 4));
            a2.StoreAligned(output + ((long)o + 8));
            a3.StoreAligned(output + ((long)o + 12));
        }
    }

    /// <summary>Same with SDOT: one instruction multiplies 16 int8 pairs and adds each group of 4 into an int32 lane.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformDotProd(byte* input, sbyte* weights, int* biases, int* output, int inputLength, int paddedOutputLength)
    {
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            var a0 = Vector128.LoadAligned(biases + ((long)o));
            var a1 = Vector128.LoadAligned(biases + ((long)o + 4));
            var a2 = Vector128.LoadAligned(biases + ((long)o + 8));
            var a3 = Vector128.LoadAligned(biases + ((long)o + 12));
            sbyte* wp = (weights + (o * 4));
            for (int i = 0; i < inputLength; i += 4)
            {
                var x = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (i)))).AsSByte();
                sbyte* wc = (wp + ((i >> 2) * stride));
                a0 = Dp.DotProduct(a0, Vector128.LoadAligned(wc), x);
                a1 = Dp.DotProduct(a1, Vector128.LoadAligned(wc + (16)), x);
                a2 = Dp.DotProduct(a2, Vector128.LoadAligned(wc + (32)), x);
                a3 = Dp.DotProduct(a3, Vector128.LoadAligned(wc + (48)), x);
            }
            a0.StoreAligned(output + ((long)o));
            a1.StoreAligned(output + ((long)o + 4));
            a2.StoreAligned(output + ((long)o + 8));
            a3.StoreAligned(output + ((long)o + 12));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformSparse(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
    {
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            var a0 = Vector128.LoadAligned(biases + ((long)o));
            var a1 = Vector128.LoadAligned(biases + ((long)o + 4));
            var a2 = Vector128.LoadAligned(biases + ((long)o + 8));
            var a3 = Vector128.LoadAligned(biases + ((long)o + 12));
            sbyte* wp = (weights + (o * 4));
            for (int j = 0; j < nonZeroCount; j++)
            {
                int c = nonZeroChunks[j];
                var x = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (c * 4)))).AsSByte();
                var xl = x.GetLower();
                sbyte* wc = (wp + (c * stride));
                a0 = MultiplyAccumulateNeon(a0, Vector128.LoadAligned(wc), x, xl);
                a1 = MultiplyAccumulateNeon(a1, Vector128.LoadAligned(wc + (16)), x, xl);
                a2 = MultiplyAccumulateNeon(a2, Vector128.LoadAligned(wc + (32)), x, xl);
                a3 = MultiplyAccumulateNeon(a3, Vector128.LoadAligned(wc + (48)), x, xl);
            }
            a0.StoreAligned(output + ((long)o));
            a1.StoreAligned(output + ((long)o + 4));
            a2.StoreAligned(output + ((long)o + 8));
            a3.StoreAligned(output + ((long)o + 12));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AffineTransformSparseDotProd(byte* input, sbyte* weights, int* biases, int* nonZeroChunks, int nonZeroCount, int* output, int paddedOutputLength)
    {
        int stride = paddedOutputLength * 4;
        for (int o = 0; o < paddedOutputLength; o += 16)
        {
            var a0 = Vector128.LoadAligned(biases + ((long)o));
            var a1 = Vector128.LoadAligned(biases + ((long)o + 4));
            var a2 = Vector128.LoadAligned(biases + ((long)o + 8));
            var a3 = Vector128.LoadAligned(biases + ((long)o + 12));
            sbyte* wp = (weights + (o * 4));
            for (int j = 0; j < nonZeroCount; j++)
            {
                int c = nonZeroChunks[j];
                var x = Vector128.Create(Unsafe.ReadUnaligned<int>((input + (c * 4)))).AsSByte();
                sbyte* wc = (wp + (c * stride));
                a0 = Dp.DotProduct(a0, Vector128.LoadAligned(wc), x);
                a1 = Dp.DotProduct(a1, Vector128.LoadAligned(wc + (16)), x);
                a2 = Dp.DotProduct(a2, Vector128.LoadAligned(wc + (32)), x);
                a3 = Dp.DotProduct(a3, Vector128.LoadAligned(wc + (48)), x);
            }
            a0.StoreAligned(output + ((long)o));
            a1.StoreAligned(output + ((long)o + 4));
            a2.StoreAligned(output + ((long)o + 8));
            a3.StoreAligned(output + ((long)o + 12));
        }
    }

    /// <summary>
    /// Int16 output layer over all slots: smlal / smlal2 widen int16*int16 into int32 lanes; a single horizontal reduction at the end.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        var zero = Vector128<short>.Zero; var max = Vector128.Create(clipMax);
        var s0 = Vector128<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = (weights + ((long)s * accumulatorSize));
            for (long i = 0; i < (long)accumulatorSize; i += 8)
            {
                var x = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i)), zero), max);
                var wv = Vector128.LoadAligned(ws + (i));
                s0 = AdvSimd.MultiplyWideningUpperAndAdd(AdvSimd.MultiplyWideningLowerAndAdd(s0, x.GetLower(), wv.GetLower()), x, wv);
            }
        }
        return Vector128.Sum(s0);
    }

    /// <summary>The same for SCReLU: ((c * w) as int16) * c.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int DotProductSCReLU(short** accumulators, short* weights, int accumulatorCount, int accumulatorSize, short clipMax)
    {
        var zero = Vector128<short>.Zero; var max = Vector128.Create(clipMax);
        var s0 = Vector128<int>.Zero;
        for (int s = 0; s < accumulatorCount; s++)
        {
            short* a = accumulators[s];
            short* ws = (weights + ((long)s * accumulatorSize));
            for (long i = 0; i < (long)accumulatorSize; i += 8)
            {
                var x = Vector128.Min(Vector128.Max(Vector128.LoadAligned(a + (i)), zero), max);
                var t = x * Vector128.LoadAligned(ws + (i));                 // mul: low 16 bits, as in the scalar reference
                s0 = AdvSimd.MultiplyWideningUpperAndAdd(AdvSimd.MultiplyWideningLowerAndAdd(s0, t.GetLower(), x.GetLower()), t, x);
            }
        }
        return Vector128.Sum(s0);
    }

    /// <summary>Last dense layer (one output): bias + sum of input[i] * w[i], contiguous weights. n is a multiple of 64; weights past In are zero
    /// and the input buffer is readable up to n, so the bytes beyond In contribute nothing.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int AffineTransformSingleOutput(byte* input, sbyte* weights, int bias, int paddedInputLength)
    {
        var a0 = Vector128<int>.Zero;
        for (int i = 0; i < paddedInputLength; i += 16)
        {
            var x = Vector128.Load(input + i).AsSByte();
            a0 = MultiplyAccumulateNeon(a0, Vector128.LoadAligned(weights + i), x, x.GetLower());
        }
        return bias + Vector128.Sum(a0);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static int AffineTransformSingleOutputDotProd(byte* input, sbyte* weights, int bias, int paddedInputLength)
    {
        var a0 = Vector128<int>.Zero; var a1 = Vector128<int>.Zero;
        for (int i = 0; i < paddedInputLength; i += 32)
        {
            a0 = Dp.DotProduct(a0, Vector128.LoadAligned(weights + i), Vector128.Load(input + i).AsSByte());
            a1 = Dp.DotProduct(a1, Vector128.LoadAligned(weights + i + 16), Vector128.Load(input + i + 16).AsSByte());
        }
        return bias + Vector128.Sum(Vector128.Add(a0, a1));
    }
}
