using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;

namespace Nnue.Tests;

/// <summary>An exception assertion that, unlike <c>Assert.ThrowsException</c>, also accepts exceptions derived from the expected type
/// (an <see cref="ArgumentOutOfRangeException"/> satisfies <c>Throws&lt;ArgumentException&gt;</c>).</summary>
internal static class ExceptionAssert
{
    public static void Throws<TException>(Action action, string message) where TException : Exception
    {
        try
        { action(); }
        catch (TException) { return; }
        catch (Exception e) { Assert.Fail($"{message}: expected {typeof(TException).Name}, got {e.GetType().Name}: {e.Message}"); }
        Assert.Fail($"{message}: no exception was thrown (expected {typeof(TException).Name})");
    }
}

/// <summary>Random weights through the public builder, as an external user (or the importer) would fill a model.</summary>
internal static class RandomModel
{
    public static NnueModel Make(NetworkSpec spec, Random r)
    {
        NnueModelBuilder b = new(spec);
        foreach (ref short w in b.FeatureWeights)
            w = (short)r.Next(-80, 80);
        foreach (ref short w in b.FeatureBiases)
            w = (short)r.Next(-40, 40);
        if (spec.Precision == NetworkPrecision.Int16)
        {
            int limit = spec.Activation == FeatureTransformerActivation.SquaredClippedReLU ? 32767 / spec.ActivationQuantization : 3000;     // SCReLU: c*w must fit int16
            for (int k = 0; k < spec.OutputBuckets; k++)
                foreach (ref short w in b.GetOutputWeights(k))
                    w = (short)r.Next(-limit, limit + 1);
            foreach (ref int x in b.OutputBiases)
                x = r.Next(-1_000_000, 1_000_000);
        }
        else
        {
            for (int k = 0; k < spec.OutputBuckets; k++)
                for (int l = 0; l < spec.DenseLayerCount; l++)
                {
                    foreach (ref sbyte w in b.GetDenseWeights(k, l))
                        w = (sbyte)r.Next(-127, 128);
                    foreach (ref int x in b.GetDenseBiases(k, l))
                        x = r.Next(-2_000_000, 2_000_000);
                }
        }
        return b.Build();
    }

    /// <summary>Accumulator contents covering negatives, in-range values and values above every clamp.
    /// pattern 0: everything clamped to zero, 1: everything non-zero, else mixed.</summary>
    public static void Fill(Accumulator acc, Random r, int pattern)
    {
        Span<short> v = acc.AsSpan();
        for (int i = 0; i < v.Length; i++)
            v[i] = pattern == 0 ? (short)r.Next(-300, 0) : pattern == 1 ? (short)r.Next(1, 700) : (short)r.Next(-300, 700);
    }
}
