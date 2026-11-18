using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;
using Nnue.Training.Models;
using Nnue.Training.Optimization;
using Nnue.Training.Quantization;


namespace Nnue.Tests;

[TestClass]
public sealed class TrainingTests
{
    // ------------------------------------------------------------------------------------------------ gradient check
    [TestMethod]
    public void Gradients_MatchFiniteDifferences()
    {
        static NetworkSpec I8(int h, int k, FeatureTransformerActivation a, int[] layers, HiddenActivation hid = HiddenActivation.ClippedReLU, bool dense = false, bool skip = false, int buckets = 1) => new()
        {
            FeatureCount = 300,
            AccumulatorSize = h,
            Slots = k,
            Precision = NetworkPrecision.Int8,
            Activation = a,
            HiddenLayerSizes = [.. layers],
            HiddenActivation = hid,
            DenseConnections = dense,
            SkipConnection = skip,
            OutputBuckets = buckets,
        };
        (string, NetworkSpec)[] configs = new (string, NetworkSpec)[]
        {
            ("Int16 CReLU K=1",                    new NetworkSpec { FeatureCount = 300, AccumulatorSize = 64 }),
            ("Int16 SCReLU K=2",                   new NetworkSpec { FeatureCount = 300, AccumulatorSize = 64, Slots = 2, Activation = FeatureTransformerActivation.SquaredClippedReLU }),
            ("Int8 CReLU K=2 16-32",               I8(64, 2, FeatureTransformerActivation.ClippedReLU, [16, 32])),
            ("Int8 SCReLU K=1 16",                 I8(64, 1, FeatureTransformerActivation.SquaredClippedReLU, [16])),
            ("Int8 Pairwise K=1 32-32",            I8(128, 1, FeatureTransformerActivation.Pairwise, [32, 32])),
            ("Int8 SqrClippedReLU K=2 32-16",            I8(64, 2, FeatureTransformerActivation.ClippedReLU, [32, 16], HiddenActivation.SqrClippedReLU)),
            ("Int8 dense+skip K=3 16x3",           I8(64, 3, FeatureTransformerActivation.ClippedReLU, [16, 16, 16], HiddenActivation.ClippedReLU, true, true)),
            ("Stockfish-like K=2, 4 buckets",      NetworkSpec.CreateStockfishLike(300, 128, 32, 32, 4)),
        };
        foreach ((string? name, NetworkSpec? spec) in configs)
        {
            foreach (LossFunction loss in new[] { LossFunction.SigmoidMse, LossFunction.Mse })
            {
                if (loss == LossFunction.Mse && !name.StartsWith("Int8 CReLU K=2") && !name.StartsWith("Int16 CReLU"))
                    continue;
                FloatModel net = new(spec);
                net.Randomize(11, 12);
                NnueTrainer tr = new(net, new TrainerOptions { Threads = 1, Loss = loss });
                Random r = new(21);
                int probes = 0, fails = 0, bad = 0, samples = 20;
                for (int s = 0; s < samples; s++)
                {
                    TrainingSample sample = new()
                    {
                        Features = Enumerable.Range(0, spec.Slots).Select(_ => Enumerable.Range(0, r.Next(6, 14)).Select(_ => r.Next(spec.FeatureCount)).Distinct().ToArray()).ToArray(),
                        Bucket = r.Next(spec.OutputBuckets),
                        Target = loss == LossFunction.SigmoidMse ? 0.1f + (0.8f * (float)r.NextDouble()) : (float)((r.NextDouble() * 2) - 1),
                    };
                    GradientCheckResult res = tr.CheckGradients(sample, epsilon: 2e-3f);
                    probes += res.Probes;
                    fails += res.Failures;
                    if (res.Failures > 0)
                        bad++;
                }
                // Activations have kinks: a unit within ~1e-3 of a clamp boundary makes a few probes of that one sample disagree, but that
                // never repeats across samples. A real backprop error fails in (nearly) every sample.
                bool ok = bad * 5 <= samples && probes > 0;
                Assert.IsTrue(ok, $"{name} ({loss}): {fails} failed probes in {bad}/{samples} samples");
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ hand-computed scales
    [TestMethod]
    public void Quantization_MatchesHandComputedOutputs()
    {
        void Case(string name, NetworkSpec spec, Action<FloatModel> setup, int[][] features, double expected, int bucket = 0)
        {
            FloatModel net = new(spec);
            setup(net);
            NnueModel model = NnueQuantizer.Quantize(net);
            NnueEvaluator ev = new(model);
            double f = net.PredictScore(features, bucket);
            int i = ev.EvaluateFeatures(features, bucket);
            bool ok = Math.Abs(f - expected) < 0.5 && Math.Abs(i - expected) <= 2.5;
            Assert.IsTrue(ok, $"{name}: expected {expected:F1}, float {f:F1}, int {i}");
        }
        NetworkSpec Int8(int slots = 1, int h = 64, params int[] layers) => new() { FeatureCount = 8, AccumulatorSize = h, Slots = slots, Precision = NetworkPrecision.Int8, HiddenLayerSizes = [.. layers] };
        int[][] One(params int[] f) => [f];
        int[][] Two(int[] a, int[] b) => [a, b];
        int[] none = [];

        // feature 0 drives accumulator unit 0 to exactly 1.0
        Case("Int8 CReLU hidden: clip(0.75) * 1.875", Int8(1, 64, 16), n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 0.75f; n.DenseLayers[0][1].Weights[0] = 1.875f; }, One(0), 0.75 * 1.875 * 400);
        Case("Int8 SqrClippedReLU hidden: sqr(+1.0) * 1.875", Int8(1, 64, 16) with { HiddenActivation = HiddenActivation.SqrClippedReLU }, n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 1f; n.DenseLayers[0][1].Weights[0] = 1.875f; }, One(0), 127.0 / 128 * 1.875 * 400);
        Case("Int8 SqrClippedReLU hidden: sqr(-0.796875), sign dropped", Int8(1, 64, 16) with { HiddenActivation = HiddenActivation.SqrClippedReLU }, n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = -0.796875f; n.DenseLayers[0][1].Weights[0] = 1.875f; }, One(0), 0.796875 * 0.796875 * 127.0 / 128 * 1.875 * 400);
        Case("Int8 SqrClippedReLU hidden: clip half is read at index 16", Int8(1, 64, 16) with { HiddenActivation = HiddenActivation.SqrClippedReLU }, n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 0.625f; n.DenseLayers[0][1].Weights[16] = 1.5f; }, One(0), 0.625 * 1.5 * 400);
        Case("Int8 skip connection: z0[14] - z0[15] = 1.0 - 0.25", Int8(1, 64, 16) with { SkipConnection = true }, n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[14 * 64] = 1f; n.DenseLayers[0][0].Weights[15 * 64] = 0.25f; }, One(0), 0.75 * 400);
        Case("Int8 dense connections: final layer reads layer 0", Int8(1, 64, 16, 16) with { DenseConnections = true }, n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 0.625f; n.DenseLayers[0][2].Weights[0] = 1.5f; }, One(0), 0.625 * 1.5 * 400);
        Case("Int8 no dense connections: final layer reads layer 1 only", Int8(1, 64, 16, 16), n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 0.625f; n.DenseLayers[0][2].Weights[0] = 1.5f; }, One(0), 0.0);
        Case("Int8 pairwise: clip(1.0) * clip(1.0) * 1.875", Int8(1, 128) with { Activation = FeatureTransformerActivation.Pairwise }, n => { n.FeatureWeights[0] = 1f; n.FeatureWeights[64] = 1f; n.DenseLayers[0][0].Weights[0] = 1.875f; }, One(0), 1.875 * 400);
        Case("Int8 SCReLU FT: (192/255)^2 * 1.875", Int8(1, 64) with { Activation = FeatureTransformerActivation.SquaredClippedReLU }, n => { n.FeatureWeights[0] = 192f / 255f; n.DenseLayers[0][0].Weights[0] = 1.875f; }, One(0), Math.Pow(192.0 / 255, 2) * 1.875 * 400);
        Case("Int16 SCReLU: (192/255)^2 * 1.5", new NetworkSpec { FeatureCount = 8, AccumulatorSize = 64, Activation = FeatureTransformerActivation.SquaredClippedReLU }, n => { n.FeatureWeights[0] = 192f / 255f; n.DenseLayers[0][0].Weights[0] = 1.5f; }, One(0), Math.Pow(192.0 / 255, 2) * 1.5 * 400);
        Case("Int16 CReLU: clip(1.0) * 1.5", new NetworkSpec { FeatureCount = 8, AccumulatorSize = 64 }, n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 1.5f; }, One(0), 1.5 * 400);

        // slots: concatenation order and per-slot weights
        NetworkSpec s16 = new()
        { FeatureCount = 8, AccumulatorSize = 64, Slots = 2 };
        void Weights16(FloatModel n)
        { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 1.5f; n.DenseLayers[0][0].Weights[64] = 0.5f; }
        Case("Int16 two slots, both active: 1.5 + 0.5", s16, Weights16, Two([0], [0]), 2.0 * 400);
        Case("Int16 two slots, only slot 0 active (weight 1.5)", s16, Weights16, Two([0], none), 1.5 * 400);
        Case("Int16 two slots, only slot 1 active (weight 0.5)", s16, Weights16, Two(none, [0]), 0.5 * 400);
        NetworkSpec s8 = Int8(2, 64);
        void Weights8(FloatModel n)
        { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 1.5f; n.DenseLayers[0][0].Weights[64] = 0.5f; }
        Case("Int8 two slots, only slot 0 active (weight 1.5)", s8, Weights8, Two([0], none), 1.5 * 400);
        Case("Int8 two slots, only slot 1 active (weight 0.5)", s8, Weights8, Two(none, [0]), 0.5 * 400);
        NetworkSpec s3 = new()
        { FeatureCount = 8, AccumulatorSize = 64, Slots = 3 };
        Case("Int16 three slots: weights 1.0 / 0.5 / 0.25 on slots 0 / 1 / 2", s3,
            n => { n.FeatureWeights[0] = 1f; n.DenseLayers[0][0].Weights[0] = 1f; n.DenseLayers[0][0].Weights[64] = 0.5f; n.DenseLayers[0][0].Weights[128] = 0.25f; }, [[0], [0], [0]], 1.75 * 400);
    }

    // ------------------------------------------------------------------------------------------------ bucket routing
    [TestMethod]
    public void Trainer_RoutesSamplesToTheirOutputBucket()
    {
        int[] target = { -300, -100, 100, 300 };
        foreach (int buckets in new[] { 1, 4 })
        {
            NetworkSpec spec = new()
            { FeatureCount = 8, AccumulatorSize = 64, Precision = NetworkPrecision.Int8, HiddenLayerSizes = [16], OutputBuckets = buckets };
            TrainingSample[] set = Enumerable.Range(0, 4000).Select(i => new TrainingSample
            {
                Features = [[0, 1, 2]],
                Bucket = buckets == 1 ? 0 : i & 3,
                Target = TargetFunctions.ScoreToTarget(target[i & 3], spec.OutputScale),
            }).ToArray();
            FloatModel net = new(spec);
            net.Randomize(3, 3);
            new NnueTrainer(net, new TrainerOptions { BatchSize = 256, LearningRate = 0.01f, Threads = 1 }).Train(set, 80);
            NnueEvaluator ev = new(NnueQuantizer.Quantize(net, 3));
            int[] model = Enumerable.Range(0, buckets).Select(b => ev.EvaluateFeatures(new[] { new[] { 0, 1, 2 } }, b)).ToArray();
            if (buckets == 4)
                for (int b = 0; b < 4; b++)
                    Assert.IsTrue(Math.Abs(model[b] - target[b]) < 25, $"bucket {b} learned its own target ({model[b]} vs {target[b]})");
            else
                Assert.IsTrue(model[0] is > -150 and < 150, "without buckets only the mean can be learned");
        }
    }

    // ------------------------------------------------------------------------------------------------ training + quantization end to end
    private sealed class Synthetic
    {
        private static readonly int[] Value = { 100, 300, 300, 500, 900, 0 };
        /// <summary>A chess-like toy: two slots (side to move / other side) built ABOVE the library; nothing of this is known to the core.</summary>
        public static TrainingSample Generate(Random rng, float outputScale)
        {
            int own = rng.Next(3, 9), their = rng.Next(3, 9), n = own + their, phase = 16 - n;
            int[] squares = Enumerable.Range(0, 64).OrderBy(_ => rng.Next()).Take(n).ToArray();
            int[] a = new int[n];
            int[] b = new int[n];
            int score = 0, ownB = 0, theirB = 0;
            for (int i = 0; i < n; i++)
            {
                bool isOwn = i < own;
                int type = rng.Next(6), s = squares[i];
                a[i] = ((((isOwn ? 0 : 1) * 6) + type) * 64) + s;
                b[i] = ((((isOwn ? 1 : 0) * 6) + type) * 64) + (s ^ 56);
                score += isOwn ? Value[type] : -Value[type];
                if (type == 0)
                    score += (isOwn ? 1 : -1) * (s >> 3) * (5 + (3 * phase));
                if (type == 2)
                { if (isOwn) ownB++; else theirB++; }
            }
            int pair = n <= 9 ? 90 : 30;
            if (ownB >= 2)
                score += pair;
            if (theirB >= 2)
                score -= pair;
            return new TrainingSample { Features = [a, b], Bucket = Math.Min(3, (n - 6) / 3), Target = TargetFunctions.ScoreToTarget(score, outputScale) };
        }
    }

    [TestMethod]
    [TestCategory("Slow")]
    public void Training_ReducesLossAndQuantizationStaysClose()
    {
        (string, NetworkSpec, double)[] configs = new (string, NetworkSpec, double)[]
        {
            ("Int16 CReLU, 2 slots",           new NetworkSpec { FeatureCount = 768, AccumulatorSize = 128, Slots = 2 }, 15),
            ("Int8 CReLU 16-32, 2 slots",      new NetworkSpec { FeatureCount = 768, AccumulatorSize = 128, Slots = 2, Precision = NetworkPrecision.Int8, HiddenLayerSizes = [16, 32] }, 40),
            ("Stockfish-like, 4 stacks",       NetworkSpec.CreateStockfishLike(768, 128, 32, 32, 4), 80),
        };
        foreach ((string? name, NetworkSpec? spec, double tolerance) in configs)
        {
            Random rng = new(42);
            TrainingSample[] train = Enumerable.Range(0, 40_000).Select(_ => Synthetic.Generate(rng, spec.OutputScale)).ToArray();
            TrainingSample[] test = Enumerable.Range(0, 2_000).Select(_ => Synthetic.Generate(rng, spec.OutputScale)).ToArray();
            if (spec.OutputBuckets == 1)
            { train = Clear(train); test = Clear(test); }
            FloatModel net = new(spec);
            net.Randomize(7, 16);
            NnueTrainer tr = new(net, new TrainerOptions { BatchSize = 4096, LearningRate = 0.01f, LearningRateDecay = 0.8f });
            double before = tr.ComputeLoss(test);
            tr.Train(train, 10);
            double after = tr.ComputeLoss(test);
            NnueModel model = NnueQuantizer.Quantize(net, out QuantizationReport report);
            NnueEvaluator ev = new(model);
            double sum = 0, max = 0;
            foreach (TrainingSample? s in test)
            {
                double d = Math.Abs(net.PredictScore(s.Features, s.Bucket) - ev.EvaluateFeatures(s.Features, s.Bucket));
                sum += d;
                max = Math.Max(max, d);
            }
            double mean = sum / test.Length;
            Assert.IsTrue(after < before * 0.5, $"{name}: test loss {before:F4} -> {after:F4} (expected at least a halving)");
            Assert.IsTrue(mean < tolerance, $"{name}: mean |float - int| = {mean:F1} (limit {tolerance})");
            Assert.IsTrue(report.AccumulatorSafe && report.OutputSafe, $"{name}: no overflow possible ({report})");
        }

        static TrainingSample[] Clear(TrainingSample[] a) => a.Select(s => new TrainingSample { Features = s.Features, Target = s.Target, Bucket = 0 }).ToArray();
    }

    [TestMethod]
    public void Trainer_ReportsProgressAndHonoursCancellation()
    {
        NetworkSpec spec = NetworkSpec.CreateSimple(64, 32, slots: 1);
        Random rng = new(2);
        TrainingSample[] data = Enumerable.Range(0, 2000).Select(_ => new TrainingSample { Features = [[rng.Next(64), rng.Next(64)]], Target = (float)rng.NextDouble() }).ToArray();
        FloatModel net = new(spec);
        net.Randomize(1, 2);
        NnueTrainer trainer = new(net, new TrainerOptions { BatchSize = 100, LearningRate = 0.01f, LearningRateDecay = 0.5f, Threads = 1 });
        List<TrainingProgress> reports = [];
        trainer.Train(data, 3, new Recorder(reports));
        Assert.IsTrue(reports.Count == 3 && reports.Select(r => r.Epoch).SequenceEqual([1, 2, 3]) && reports.All(r => r.EpochCount == 3), "one report per epoch");
        Assert.IsTrue(reports[0].LearningRate == 0.01f && reports[1].LearningRate == 0.005f, "the learning rate of each epoch, decayed in between");

        using CancellationTokenSource cancellation = new();
        cancellation.Cancel();
        float before = net.FeatureWeights[0];
        ExceptionAssert.Throws<OperationCanceledException>(() => trainer.Train(data, 5, cancellationToken: cancellation.Token), "cancelled before the first batch");
        Assert.IsTrue(net.FeatureWeights[0] == before, "no batch ran after cancellation");
    }

    [TestMethod]
    public void Trainer_CopiesItsOptions_AndRejectsBadSamplesUpFront()
    {
        NetworkSpec spec = NetworkSpec.CreateSimple(64, 32, slots: 2, outputBuckets: 2);
        TrainerOptions options = new()
        { BatchSize = 64, LearningRate = 0.01f, LearningRateDecay = 0.5f, Threads = 2 };
        FloatModel net = new(spec);
        net.Randomize(1, 2);
        NnueTrainer trainer = new(net, options);
        TrainingSample good = new()
        { Features = [[1, 2], [3]], Bucket = 1, Target = 0.5f };
        trainer.Train([good, good], 2);
        Assert.IsTrue(options.LearningRate == 0.01f && trainer.Options.LearningRate == 0.0025f, "the caller's options keep their learning rate");

        void Rejects(TrainingSample sample, string what) => ExceptionAssert.Throws<ArgumentException>(() => trainer.TrainEpoch([good, sample]), what);
        Rejects(new TrainingSample { Features = [[1, 64], [3]], Target = 0.5f }, "feature index == FeatureCount");
        Rejects(new TrainingSample { Features = [[-1], [3]], Target = 0.5f }, "negative feature index");
        Rejects(new TrainingSample { Features = [[1]], Target = 0.5f }, "one feature list for two slots");
        Rejects(new TrainingSample { Features = [[1], [3]], Bucket = 2, Target = 0.5f }, "bucket out of range");
        Rejects(new TrainingSample { Features = [[1], [3]], Target = float.NaN }, "NaN target");
        ExceptionAssert.Throws<ArgumentException>(() => trainer.ComputeLoss([new TrainingSample { Features = [[1, 99], [3]] }]), "ComputeLoss validates too");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => net.Predict([[1, 64], [3]]), "Predict: feature index == FeatureCount");

        foreach ((TrainerOptions? bad, string? what) in new (TrainerOptions, string)[]
        {
            (new TrainerOptions { LearningRate = 0 }, "zero learning rate"), (new TrainerOptions { LearningRate = float.NaN }, "NaN learning rate"),
            (new TrainerOptions { Beta1 = 1f }, "Beta1 = 1"), (new TrainerOptions { Epsilon = -1e-8f }, "negative epsilon"),
            (new TrainerOptions { BatchSize = 0 }, "zero batch size"), (new TrainerOptions { DenseWeightClip = 0 }, "zero dense clip"),
            (new TrainerOptions { Loss = (LossFunction)5 }, "undefined loss"),
        })
            ExceptionAssert.Throws<ArgumentException>(() => new NnueTrainer(net, bad), "options: " + what);
    }

    private sealed class Recorder(List<TrainingProgress> reports) : IProgress<TrainingProgress>
    {
        public void Report(TrainingProgress value) => reports.Add(value);
    }

    // ------------------------------------------------------------------------------------------------ report, checkpoints
    [TestMethod]
    public void QuantizationReport_AndFloatCheckpoint()
    {
        NetworkSpec spec = new()
        { FeatureCount = 768, AccumulatorSize = 64 };
        FloatModel big = new(spec);
        Array.Fill(big.FeatureWeights, 2.0f);
        _ = NnueQuantizer.Quantize(big, out QuantizationReport safe, maxActiveFeatures: 32);
        _ = NnueQuantizer.Quantize(big, out QuantizationReport risky, maxActiveFeatures: 100);
        Assert.IsTrue(safe.AccumulatorSafe && !risky.AccumulatorSafe, $"overflow check: 32 active safe={safe.AccumulatorSafe} (worst {safe.WorstCaseAccumulator}), 100 active safe={risky.AccumulatorSafe} (worst {risky.WorstCaseAccumulator})");

        foreach (NetworkSpec? s in new[] { spec, NetworkSpec.CreateStockfishLike(100, 128, 32, 32, 2) })
        {
            FloatModel net = new(s);
            net.Randomize(4, 8);
            using MemoryStream ms = new();
            net.Save(ms);
            ms.Position = 0;
            FloatModel loaded = FloatModel.Load(ms);
            int[][] f = Enumerable.Range(0, s.Slots).Select(i => new[] { 1 + i, 7, 9 }).ToArray();
            Assert.IsTrue(net.Predict(f, s.OutputBuckets - 1) == loaded.Predict(f, s.OutputBuckets - 1), "float checkpoint round trip");
        }
        ExceptionAssert.Throws<InvalidDataException>(() => FloatModel.Load(new MemoryStream(new byte[40])), "garbage checkpoint rejected");
    }
}
