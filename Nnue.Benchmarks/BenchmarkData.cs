using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;

namespace Nnue.Benchmarks;

/// <summary>Random networks and accumulator contents for the benchmarks (the values do not matter for speed, only their ranges: they decide how many activations are zero).</summary>
internal static class BenchmarkData
{
    public static NnueModel CreateRandomModel(NetworkSpec spec, Random random)
    {
        NnueModelBuilder builder = new(spec);
        foreach (ref short w in builder.FeatureWeights)
            w = (short)random.Next(-80, 80);
        foreach (ref short w in builder.FeatureBiases)
            w = (short)random.Next(-40, 40);
        if (spec.Precision == NetworkPrecision.Int16)
        {
            for (int bucket = 0; bucket < spec.OutputBuckets; bucket++)
                foreach (ref short w in builder.GetOutputWeights(bucket))
                    w = (short)random.Next(-3000, 3000);
            foreach (ref int bias in builder.OutputBiases)
                bias = random.Next(-1_000_000, 1_000_000);
        }
        else
            for (int bucket = 0; bucket < spec.OutputBuckets; bucket++)
                for (int layer = 0; layer < spec.DenseLayerCount; layer++)
                {
                    foreach (ref sbyte w in builder.GetDenseWeights(bucket, layer))
                        w = (sbyte)random.Next(-127, 128);
                    foreach (ref int bias in builder.GetDenseBiases(bucket, layer))
                        bias = random.Next(-2_000_000, 2_000_000);
                }
        return builder.Build();
    }

    /// <summary>An accumulator filled with values in [<paramref name="low"/>, <paramref name="high"/>).</summary>
    public static void Fill(Accumulator accumulator, Random random, int low, int high)
    {
        Span<short> values = accumulator.AsSpan();
        for (int i = 0; i < values.Length; i++)
            values[i] = (short)random.Next(low, high);
    }
}
