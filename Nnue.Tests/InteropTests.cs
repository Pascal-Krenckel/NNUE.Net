using System.Buffers.Binary;
using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Interop;
using Nnue.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Nnue.Tests;

[TestClass]
public sealed class InteropTests
{
    /// <summary>Writes a bullet file with hand-written code (independent of <see cref="BulletFormat.Save"/>): l0w, l0b, l1w, l1b, padded to 64 bytes.</summary>
    private static byte[] WriteBulletFile(short[] l0w, short[] l0b, short[] l1w, int[] l1b, bool int32Biases)
    {
        using var ms = new MemoryStream();
        var two = new byte[2]; var four = new byte[4];
        foreach (short v in l0w.Concat(l0b).Concat(l1w)) { BinaryPrimitives.WriteInt16LittleEndian(two, v); ms.Write(two); }
        foreach (int v in l1b)
            if (int32Biases) { BinaryPrimitives.WriteInt32LittleEndian(four, v); ms.Write(four); }
            else { BinaryPrimitives.WriteInt16LittleEndian(two, (short)v); ms.Write(two); }
        while (ms.Length % 64 != 0) ms.WriteByte(0);
        return ms.ToArray();
    }

    [TestMethod]
    public void BulletFormat_RoundTripsAndMatchesScalarReference()
    {
        var random = new Random(21);
        foreach (int buckets in new[] { 1, 4 })
        foreach (var layout in new[] { BulletOutputWeightsLayout.BucketMajor, BulletOutputWeightsLayout.InputMajor })
        foreach (bool int32Biases in new[] { false, true })
        foreach (var activation in new[] { FeatureTransformerActivation.ClippedReLU, FeatureTransformerActivation.SquaredClippedReLU })
        {
            const int Features = 40, Hidden = 64, Slots = 2;
            var spec = NetworkSpec.CreateBuilder(Features).WithAccumulatorSize(Hidden).WithSlots(Slots).WithOutputBuckets(buckets).WithInt16(activation).Build();
            int qa = spec.ActivationQuantization, qb = spec.OutputWeightQuantization;
            string label = $"buckets={buckets} {layout} {(int32Biases ? "i32" : "i16")} bias {activation}";

            int limit = activation == FeatureTransformerActivation.SquaredClippedReLU ? 32767 / qa : 120;
            var l0w = Enumerable.Range(0, Features * Hidden).Select(_ => (short)random.Next(-90, 90)).ToArray();
            var l0b = Enumerable.Range(0, Hidden).Select(_ => (short)random.Next(-60, 60)).ToArray();
            var weights = new short[buckets][];                         // [bucket][slot * Hidden + i], the logical content
            for (int b = 0; b < buckets; b++) weights[b] = Enumerable.Range(0, Slots * Hidden).Select(_ => (short)random.Next(-limit, limit + 1)).ToArray();
            var l1w = new short[buckets * Slots * Hidden];              // the file order
            for (int b = 0; b < buckets; b++)
                for (int i = 0; i < Slots * Hidden; i++)
                    l1w[layout == BulletOutputWeightsLayout.BucketMajor ? b * Slots * Hidden + i : i * buckets + b] = weights[b][i];
            var l1b = Enumerable.Range(0, buckets).Select(_ => random.Next(-3000, 3000)).ToArray();

            var options = new BulletFormatOptions { OutputWeightsLayout = layout, OutputBiasesAreInt32 = int32Biases };
            byte[] file = WriteBulletFile(l0w, l0b, l1w, l1b, int32Biases);
            using var model = BulletFormat.Load(file, spec, options);

            bool same = model.FeatureWeights.SequenceEqual(l0w) && model.FeatureBiases.SequenceEqual(l0b);
            int biasScale = activation == FeatureTransformerActivation.SquaredClippedReLU ? qa : 1;
            for (int b = 0; b < buckets; b++) same &= model.GetOutputWeights(b).SequenceEqual(weights[b]) && model.GetOutputBias(b) == l1b[b] * biasScale;
            Assert.IsTrue(same, $"{label}: arrays land in the right places");

            // evaluate against an independent scalar computation straight from the file's arrays
            var evaluator = new NnueEvaluator(model);
            using var buffer = model.CreateAccumulators(Slots);
            var features = new[] { new[] { 1, 5, 9, 30 }, new[] { 2, 5, 39 } };
            for (int s = 0; s < Slots; s++) evaluator.RefreshAccumulator(buffer[s], features[s]);
            for (int bucket = 0; bucket < buckets; bucket++)
            {
                long sum = 0;
                for (int s = 0; s < Slots; s++)
                    for (int i = 0; i < Hidden; i++)
                    {
                        int accumulator = l0b[i] + features[s].Sum(f => l0w[f * Hidden + i]);
                        int clipped = Math.Clamp(accumulator, 0, qa);
                        sum += activation == FeatureTransformerActivation.SquaredClippedReLU ? (long)((short)(clipped * weights[bucket][s * Hidden + i])) * clipped : clipped * weights[bucket][s * Hidden + i];
                    }
                sum += l1b[bucket] * biasScale;
                double scale = activation == FeatureTransformerActivation.SquaredClippedReLU ? (double)spec.OutputScale / ((double)qa * qa * qb) : (double)spec.OutputScale / ((double)qa * qb);
                int expected = (int)(sum * scale);
                Assert.IsTrue(evaluator.Evaluate(buffer[0], buffer[1], bucket) == expected, $"{label}: evaluation of bucket {bucket} equals the scalar reference");
            }

            using var saved = new MemoryStream();
            BulletFormat.Save(model, saved, options);
            Assert.IsTrue(saved.ToArray().SequenceEqual(file), $"{label}: Save reproduces the file byte for byte");
        }

        var simple = NetworkSpec.CreateSimple(40, 64);
        byte[] good = WriteBulletFile(new short[40 * 64], new short[64], new short[128], [0], false);
        Assert.IsTrue(BulletFormat.Load(good[..(good.Length - 0)], simple).Spec.FeatureCount == 40, "exact size accepted");
        Assert.IsTrue(BulletFormat.Load(good.AsSpan(0, 40 * 64 * 2 + 64 * 2 + 128 * 2 + 2), simple).Spec.Slots == 2, "size without the 64-byte padding accepted");
        ExceptionAssert.Throws<InvalidDataException>(() => BulletFormat.Load(good.AsSpan(0, good.Length - 70), simple), "truncated file rejected");
        ExceptionAssert.Throws<InvalidDataException>(() => BulletFormat.Load(good, NetworkSpec.CreateSimple(41, 64)), "file does not match the spec");
        ExceptionAssert.Throws<ArgumentException>(() => BulletFormat.Load(good, NetworkSpec.CreateInt8(40, 64, [16])), "Int8 spec rejected");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => BulletFormat.Load(good, simple, new BulletFormatOptions { OutputWeightsLayout = (BulletOutputWeightsLayout)3 }), "undefined layout rejected");
        ExceptionAssert.Throws<ArgumentException>(() => { using var o = new MemoryStream(); BulletFormat.Save(RandomModel.Make(NetworkSpec.CreateSimple(40, 64), new Random(1)), o); }, "random biases (too large for int16) rejected on save");
    }
}
