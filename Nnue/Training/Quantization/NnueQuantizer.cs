using Nnue.Configuration;
using Nnue.Models;
using Nnue.Training.Models;

namespace Nnue.Training.Quantization;
/// <summary>Converts a trained <see cref="FloatModel"/> into an integer <see cref="NnueModel"/>.</summary>
public static class NnueQuantizer
{
    /// <summary>
    /// Largest |weight| (float units) of dense layer <paramref name="layer"/> that the quantized format can represent.
    /// Int8: 127/64 = 1.984 for every layer (also keeps pmaddubsw from saturating). Int16: the output layer limit.
    /// The trainer clamps to the same value after every step, so training cannot leave the representable range.
    /// </summary>
    public static float GetMaxWeight(NetworkSpec spec, int layer) => spec.Precision == NetworkPrecision.Int8
            ? NetworkSpec.Int8MaxWeight / (float)NetworkSpec.Int8WeightScale
            : GetMaxInt16OutputWeight(spec) / (float)spec.OutputWeightQuantization;

    /// <summary>
    /// Int16: largest |output weight| in quantized units. CReLU: bounded so the int32 dot product over all slots cannot overflow.
    /// SCReLU: bounded so that the int16 product c*w cannot overflow (32767 / ActivationQuantization).
    /// </summary>
    public static int GetMaxInt16OutputWeight(NetworkSpec spec)
        => spec.Activation == FeatureTransformerActivation.SquaredClippedReLU
            ? short.MaxValue / spec.ActivationQuantization
            : (int)Math.Min(short.MaxValue, int.MaxValue / 2 / ((long)spec.Slots * spec.AccumulatorSize * spec.ActivationQuantization));

    /// <summary>Converts a trained float network into an integer model.</summary>
    /// <param name="net">The trained network.</param>
    /// <param name="maxActiveFeatures">Largest number of simultaneously active features per accumulator; used for the accumulator overflow check.</param>
    public static NnueModel Quantize(FloatModel net, int maxActiveFeatures = 32) => Quantize(net, out _, maxActiveFeatures);

    /// <summary>Converts a trained float network into an integer model and reports clipped values and overflow risks.</summary>
    /// <param name="net">The trained network.</param>
    /// <param name="report">Clipped values, worst-case accumulator and dense sums, overflow verdicts.</param>
    /// <param name="maxActiveFeatures">Largest number of simultaneously active features per accumulator; used for the accumulator overflow check.</param>
    public static NnueModel Quantize(FloatModel net, out QuantizationReport report, int maxActiveFeatures = 32)
    {
        ArgumentNullException.ThrowIfNull(net);
        NetworkSpec spec = net.Spec;
        NnueModelBuilder builder = new(spec);
        NetworkLayout layout = builder.Layout;
        int clipped = 0;
        float maxFtW = 0, maxFtB = 0, maxDense = 0;
        long worstOut = 0;

        Span<short> ftW = builder.FeatureWeights;
        for (int i = 0; i < net.FeatureWeights.Length; i++)
        {
            maxFtW = MathF.Max(maxFtW, MathF.Abs(net.FeatureWeights[i]));
            ftW[i] = ToInt16(net.FeatureWeights[i], spec.ActivationQuantization, short.MaxValue, ref clipped);
        }
        Span<short> ftB = builder.FeatureBiases;
        for (int i = 0; i < net.FeatureBiases.Length; i++)
        {
            maxFtB = MathF.Max(maxFtB, MathF.Abs(net.FeatureBiases[i]));
            ftB[i] = ToInt16(net.FeatureBiases[i], spec.ActivationQuantization, short.MaxValue, ref clipped);
        }

        for (int bucket = 0; bucket < spec.OutputBuckets; bucket++)
        {
            if (spec.Precision == NetworkPrecision.Int16)
            {
                FloatDense d = net.DenseLayers[bucket][0];                 // Slots*H -> 1
                int limit = GetMaxInt16OutputWeight(spec);
                Span<short> ow = builder.GetOutputWeights(bucket);
                for (int i = 0; i < d.Weights.Length; i++)
                {
                    maxDense = MathF.Max(maxDense, MathF.Abs(d.Weights[i]));
                    ow[i] = ToInt16(d.Weights[i], spec.OutputWeightQuantization, limit, ref clipped);
                }
                double actScale = spec.Activation == FeatureTransformerActivation.SquaredClippedReLU ? (double)spec.ActivationQuantization * spec.ActivationQuantization : spec.ActivationQuantization;   // activation^1 or ^2 scale
                int bias = ToInt32(d.Biases[0] * actScale * spec.OutputWeightQuantization, ref clipped);
                builder.OutputBiases[bucket] = bias;
                long sumAbs = 0;
                for (int i = 0; i < ow.Length; i++)
                    sumAbs += Math.Abs((int)ow[i]);
                worstOut = Math.Max(worstOut, (long)(sumAbs * actScale) + Math.Abs((long)bias));
            }
            else
            {
                for (int k = 0; k < layout.DenseLayers.Count; k++)
                {
                    FloatDense fd = net.DenseLayers[bucket][k];
                    Span<sbyte> qw = builder.GetDenseWeights(bucket, k);
                    Span<int> qb = builder.GetDenseBiases(bucket, k);
                    for (int o = 0; o < fd.OutputLength; o++)
                    {
                        long sumAbs = 0;
                        for (int i = 0; i < fd.InputLength; i++)
                        {
                            float w = fd.Weights[(o * fd.InputLength) + i];
                            maxDense = MathF.Max(maxDense, MathF.Abs(w));
                            float r = MathF.Round(w * NetworkSpec.Int8WeightScale);
                            if (r > NetworkSpec.Int8MaxWeight)
                            { r = NetworkSpec.Int8MaxWeight; clipped++; }
                            else if (r < -NetworkSpec.Int8MaxWeight)
                            { r = -NetworkSpec.Int8MaxWeight; clipped++; }
                            qw[layout.GetDenseWeightIndex(k, i, o)] = (sbyte)r;
                            sumAbs += (long)Math.Abs(r);
                        }
                        qb[o] = ToInt32((double)fd.Biases[o] * NetworkSpec.Int8One * NetworkSpec.Int8WeightScale, ref clipped);
                        worstOut = Math.Max(worstOut, (sumAbs * NetworkSpec.Int8One) + Math.Abs((long)qb[o]));
                    }
                }
            }
        }

        NnueModel model = builder.Build();
        report = new QuantizationReport(maxFtW, maxFtB, maxDense, clipped, maxActiveFeatures, WorstCaseAccumulator(model, maxActiveFeatures), worstOut);
        return model;
    }

    /// <summary>
    /// Largest |accumulator value| reachable by any choice of <paramref name="k"/> active features:
    /// bias + sum of the k largest positive (or k most negative) weights, per accumulator unit.
    /// </summary>
    private static int WorstCaseAccumulator(NnueModel model, int k)
    {
        int f = model.Layout.FeatureCount, h = model.Layout.AccumulatorSize;
        k = Math.Clamp(k, 0, f);
        short[] weights = model.FeatureWeights.ToArray();
        short[] bias = model.FeatureBiases.ToArray();
        int[] worst = new int[h];
        _ = Parallel.For(0, h, () => new short[f], (j, _, col) =>
        {
            for (int i = 0; i < f; i++)
                col[i] = weights[(i * h) + j];
            Array.Sort(col);
            long up = bias[j], lo = bias[j];
            for (int i = 0; i < k; i++)
            {
                if (col[i] < 0)
                    lo += col[i];
                if (col[f - 1 - i] > 0)
                    up += col[f - 1 - i];
            }
            worst[j] = (int)Math.Min(int.MaxValue, Math.Max(up, -lo));
            return col;
        }, _ => { });
        return worst.Max();
    }

    private static short ToInt16(float value, int scale, int limit, ref int clipped)
    {
        float r = MathF.Round(value * scale);
        if (r > limit)
        { clipped++; return (short)limit; }
        if (r < -limit)
        { clipped++; return (short)-limit; }
        return (short)r;
    }

    private static int ToInt32(double value, ref int clipped)
    {
        double r = Math.Round(value);
        const double lim = int.MaxValue / 2;
        if (r is > lim or < -lim)
        { clipped++; r = Math.Clamp(r, -lim, lim); }
        return (int)r;
    }
}
