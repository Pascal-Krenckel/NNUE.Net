using System.Runtime.InteropServices;
using Nnue.Backends;
using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Nnue.Tests;

[TestClass]
public sealed unsafe class CoreTests
{
    public static IEnumerable<(string Name, NetworkSpec Spec)> Specs()
    {
        NetworkSpec I16(int h, int k, int buckets = 1, FeatureTransformerActivation a = FeatureTransformerActivation.ClippedReLU) => new() { FeatureCount = 400, AccumulatorSize = h, Slots = k, OutputBuckets = buckets, Activation = a };
        NetworkSpec I8(int h, int k, FeatureTransformerActivation a, int[] layers, HiddenActivation hid = HiddenActivation.ClippedReLU, bool dense = false, bool skip = false, int buckets = 1) => new()
        {
            FeatureCount = 400, AccumulatorSize = h, Slots = k, Precision = NetworkPrecision.Int8, Activation = a, HiddenLayerSizes = [.. layers], HiddenActivation = hid,
            DenseConnections = dense, SkipConnection = skip, OutputBuckets = buckets,
        };
        yield return ("Int16 K=1", I16(128, 1));
        yield return ("Int16 K=2", I16(128, 2));
        yield return ("Int16 K=3 H=96, 3 buckets", I16(96, 3, 3));
        yield return ("Int16 SCReLU K=1", I16(128, 1, 1, FeatureTransformerActivation.SquaredClippedReLU));
        yield return ("Int16 SCReLU K=2, 3 buckets", I16(96, 2, 3, FeatureTransformerActivation.SquaredClippedReLU));
        yield return ("Int16 SCReLU K=3", I16(64, 3, 1, FeatureTransformerActivation.SquaredClippedReLU));
        yield return ("Int8 CReLU K=1, no hidden", I8(64, 1, FeatureTransformerActivation.ClippedReLU, []));
        yield return ("Int8 CReLU K=2 H=96, no hidden", I8(96, 2, FeatureTransformerActivation.ClippedReLU, []));
        yield return ("Int8 CReLU K=3 16-32", I8(64, 3, FeatureTransformerActivation.ClippedReLU, [16, 32]));
        yield return ("Int8 CReLU K=2 48, 4 buckets", I8(224, 2, FeatureTransformerActivation.ClippedReLU, [48], buckets: 4));
        yield return ("Int8 SCReLU K=2 16-32", I8(128, 2, FeatureTransformerActivation.SquaredClippedReLU, [16, 32]));
        yield return ("Int8 SCReLU K=1 SqrClippedReLU", I8(64, 1, FeatureTransformerActivation.SquaredClippedReLU, [32, 16], HiddenActivation.SqrClippedReLU));
        yield return ("Int8 Pairwise K=1 32-32", I8(128, 1, FeatureTransformerActivation.Pairwise, [32, 32]));
        yield return ("Int8 Pairwise K=2 SqrClippedReLU dense skip", I8(128, 2, FeatureTransformerActivation.Pairwise, [32, 32], HiddenActivation.SqrClippedReLU, true, true));
        yield return ("Int8 Pairwise K=3 H=128 16", I8(128, 3, FeatureTransformerActivation.Pairwise, [16]));
        yield return ("Int8 SqrClippedReLU dense 16x3, 5 buckets", I8(128, 2, FeatureTransformerActivation.ClippedReLU, [16, 16, 16], HiddenActivation.SqrClippedReLU, true, false, 5));
        yield return ("Int8 CReLU dense+skip 16x3", I8(96, 2, FeatureTransformerActivation.ClippedReLU, [16, 16, 16], HiddenActivation.ClippedReLU, true, true));
        yield return ("Stockfish-like K=2 H=256, 3 stacks", NetworkSpec.CreateStockfishLike(400, 256, 32, 32, 3));
        yield return ("Stockfish-like K=2 H=1024, 32 stacks", NetworkSpec.CreateStockfishLike(400, 1024, 32, 32, 32));
    }

    /// <summary>The model configurations as MSTest data rows: (name, spec).</summary>
    public static IEnumerable<object[]> SpecCases() => Specs().Select(s => new object[] { s.Name, s.Spec });

    public static string SpecCaseName(System.Reflection.MethodInfo method, object[] data) => $"{method.Name} ({data[0]})";

    // ------------------------------------------------------------------------------------------------ backends vs scalar
    private const int BackendTrials = 400, AccumulatorTrials = 40;

    [TestMethod]
    [DynamicData(nameof(SpecCases), DynamicDataSourceType.Method, DynamicDataDisplayName = nameof(SpecCaseName))]
    public void Backends_MatchTheScalarReferenceBitForBit(string name, NetworkSpec spec)
    {
        int trials = BackendTrials;
        var backends = SimdSupport.AvailableForTesting.ToArray();
        var r = new Random(123 + name.GetHashCode() % 1000);
        {
            var model = RandomModel.Make(spec, r);
            var reference = new NnueEvaluator(model, SimdBackend.Scalar, SparseInputMode.Off);
            var evals = new List<(NnueEvaluator E, string Label)>();
            foreach (var b in backends)
                foreach (var mode in spec.Precision == NetworkPrecision.Int8 ? Enum.GetValues<SparseInputMode>() : [SparseInputMode.Off])
                    evals.Add((new NnueEvaluator(model, b, mode), $"{b}/{mode}"));
            var acc = model.CreateAccumulators(spec.Slots);
            var slots = Enumerable.Range(0, spec.Slots).Select(i => acc[i]).ToArray();
            var bad = new HashSet<string>();
            for (int t = 0; t < trials; t++)
            {
                int pattern = t % 5;
                foreach (var s in slots) RandomModel.Fill(s, r, pattern);
                int bucket = r.Next(spec.OutputBuckets);
                int expect = reference.EvaluateRaw(slots, bucket);
                foreach (var (e, label) in evals)
                    if (e.EvaluateRaw(slots, bucket) != expect) bad.Add(label);
            }
            Assert.IsTrue(bad.Count == 0, $"{name}: mismatching backends: {string.Join(", ", bad)}");
        }
    }

    // ------------------------------------------------------------------------------------------------ accumulator operations
    [TestMethod]
    [DynamicData(nameof(SpecCases), DynamicDataSourceType.Method, DynamicDataDisplayName = nameof(SpecCaseName))]
    public void AccumulatorOperations_RefreshAddRemoveAndUpdateAgree(string name, NetworkSpec spec)
    {
        int trials = AccumulatorTrials;
        var r = new Random(7);
        {
            var model = RandomModel.Make(spec, r);
            var reference = new NnueEvaluator(model, SimdBackend.Scalar);
            int F = spec.FeatureCount;
            foreach (var b in SimdSupport.AvailableForTesting)
            {
                var e = new NnueEvaluator(model, b);
                var buf = model.CreateAccumulators(4);
                var inc = buf[0]; var viaUpdate = buf[1]; var fresh = buf[2]; var expect = buf[3];
                int mismatches = 0;
                for (int t = 0; t < trials; t++)
                {
                    int[] a = Enumerable.Range(0, r.Next(1, 20)).Select(_ => r.Next(F)).Distinct().ToArray();
                    int[] bset = Enumerable.Range(0, r.Next(1, 20)).Select(_ => r.Next(F)).Distinct().ToArray();
                    var removed = a.Except(bset).ToArray(); var added = bset.Except(a).ToArray();

                    e.RefreshAccumulator(inc, a);
                    foreach (int f in removed) e.RemoveFeature(inc, f);
                    foreach (int f in added) e.AddFeature(inc, f);

                    e.RefreshAccumulator(viaUpdate, a);
                    e.UpdateAccumulator(viaUpdate, viaUpdate, removed, added);

                    e.RefreshAccumulator(fresh, bset);
                    reference.RefreshAccumulator(expect, bset);

                    if (!inc.AsSpan().SequenceEqual(fresh.AsSpan()) || !viaUpdate.AsSpan().SequenceEqual(fresh.AsSpan()) || !fresh.AsSpan().SequenceEqual(expect.AsSpan())) mismatches++;
                }
                Assert.IsTrue(mismatches == 0, $"{name} on {b}: refresh / add+remove / update disagree in {mismatches} of {trials} trials");
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ checked vs unchecked
    [TestMethod]
    public void Evaluator_ValidatesArguments_AndUncheckedMatchesChecked()
    {
        var r = new Random(9);
        var spec = new NetworkSpec { FeatureCount = 100, AccumulatorSize = 64, Slots = 2, OutputBuckets = 3 };
        var model = RandomModel.Make(spec, r);
        var other = RandomModel.Make(new NetworkSpec { FeatureCount = 100, AccumulatorSize = 96, Slots = 2 }, r);
        var e = new NnueEvaluator(model);
        var buf = model.CreateAccumulators(3);
        var foreign = other.CreateAccumulator();
        e.RefreshAccumulator(buf[0], new[] { 1, 2, 3 }); e.RefreshAccumulator(buf[1], new[] { 4, 5 });

        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.RefreshAccumulator(buf[0], new[] { 100 }), "feature == FeatureCount");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.RefreshAccumulator(buf[0], new[] { -1 }), "negative feature");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.UpdateAccumulator(buf[0], buf[1], new[] { 1 }, new[] { 1000 }), "update: added feature out of range");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.AddFeature(buf[0], 100), "add: feature out of range");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.Evaluate(buf[0], buf[1], 3), "bucket == OutputBuckets");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.Evaluate(buf[0], buf[1], -1), "negative bucket");
        ExceptionAssert.Throws<ArgumentException>(() => e.Evaluate(buf[0], 0), "one slot given, model needs two");
        ExceptionAssert.Throws<ArgumentException>(() => e.Evaluate(new[] { buf[0], buf[1], buf[2] }, 0), "three slots given, model needs two");
        ExceptionAssert.Throws<ArgumentException>(() => e.Evaluate(buf[0], foreign, 0), "accumulator of another model (size mismatch)");
        ExceptionAssert.Throws<ArgumentException>(() => e.ResetAccumulator(default), "default accumulator");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => { var _ = buf[3]; }, "buffer index out of range");

        // unchecked and checked give the same result on valid input
        for (int bucket = 0; bucket < 3; bucket++)
        {
            Assert.IsTrue(e.Evaluate(buf[0], buf[1], bucket) == e.EvaluateUnchecked(buf[0], buf[1], bucket), $"Evaluate == EvaluateUnchecked, bucket {bucket}");
            Assert.IsTrue(e.Evaluate(new[] { buf[0], buf[1] }, bucket) == e.Evaluate(buf[0], buf[1], bucket), $"span overload == pair overload, bucket {bucket}");
        }
        e.UpdateAccumulatorUnchecked(buf[0], buf[2], new[] { 1 }, new[] { 7 });
        e.UpdateAccumulator(buf[0], buf[1], new[] { 1 }, new[] { 7 });
        Assert.IsTrue(buf[1].AsSpan().SequenceEqual(buf[2].AsSpan()), "Update == UpdateAccumulatorUnchecked");
        int viaFeatures = e.EvaluateFeatures(new[] { new[] { 2, 3, 7 }, new[] { 4, 5 } }, 1);
        e.RefreshAccumulator(buf[0], new[] { 2, 3, 7 }); e.RefreshAccumulator(buf[1], new[] { 4, 5 });
        Assert.IsTrue(viaFeatures == e.Evaluate(buf[0], buf[1], 1), "Evaluate(feature lists) == Evaluate(accumulators)");

        // a model with one slot accepts the single-accumulator overload only
        var one = RandomModel.Make(new NetworkSpec { FeatureCount = 50, AccumulatorSize = 64, Slots = 1 }, r);
        var e1 = new NnueEvaluator(one); var a1 = one.CreateAccumulator();
        e1.RefreshAccumulator(a1, new[] { 3, 4 });
        Assert.IsTrue(e1.Evaluate(a1) == e1.EvaluateUnchecked(a1), "single slot: checked == unchecked");
        ExceptionAssert.Throws<ArgumentException>(() => e1.Evaluate(a1, a1, 0), "single-slot model rejects two slots");
    }

    // ------------------------------------------------------------------------------------------------ layout, alignment, validation
    [TestMethod]
    [DynamicData(nameof(SpecCases), DynamicDataSourceType.Method, DynamicDataDisplayName = nameof(SpecCaseName))]
    public void Layout_StartsEveryTensorOn64ByteBoundaries(string name, NetworkSpec spec)
    {
        {
            var l = NetworkLayout.Create(spec);
            bool aligned = l.FeatureWeightsOffset % 64 == 0 && l.FeatureBiasOffset % 64 == 0 && l.BlobSize % 64 == 0;
            if (spec.Precision == NetworkPrecision.Int16) aligned &= l.OutputWeightsOffset % 64 == 0 && l.OutputBiasOffset % 64 == 0 && l.OutputWeightsBucketStride % 64 == 0;
            else for (int b = 0; b < spec.OutputBuckets; b++) for (int k = 0; k < l.DenseLayers.Count; k++) aligned &= l.GetDenseWeightsOffset(b, k) % 64 == 0 && l.GetDenseBiasOffset(b, k) % 64 == 0;
            Assert.IsTrue(aligned, $"{name}: all tensors start on 64-byte boundaries");
            var m = RandomModel.Make(spec, new Random(1));
            Assert.IsTrue((long)m.FtWeightsPtrValue() % 64 == 0, $"{name}: blob base is 64-byte aligned");
        }
    }

    [TestMethod]
    public void NetworkSpec_HasValueEquality()
    {
        var a = NetworkSpec.CreateInt8(768, 256, [16, 32]);
        var b = NetworkSpec.CreateInt8(768, 256, [16, 32]);
        Assert.IsTrue(a == b && a.Equals(b) && a.GetHashCode() == b.GetHashCode(), "equal specs with separately created layer lists");
        Assert.IsTrue(a != (a with { HiddenLayerSizes = [16, 16] }), "different layer sizes");
        Assert.IsTrue(a == (a with { ActivationQuantization = a.DefaultActivationQuantization }), "explicit default quantization equals the implicit one");
        Assert.IsTrue(a.ToString().Contains("HiddenLayerSizes = [16, 32]"), "ToString shows the layer sizes");
        Assert.IsTrue(new NetworkSpec { HiddenLayerSizes = default }.HiddenLayerSizes.IsEmpty, "default(ImmutableArray) becomes empty");
    }

    [TestMethod]
    public void GetNonZeroInputFraction_ValidatesAccumulators()
    {
        using var model = RandomModel.Make(NetworkSpec.CreateInt8(100, 64, [16]), new Random(3));
        var evaluator = new NnueEvaluator(model);
        using var buffer = model.CreateAccumulators(2);
        evaluator.RefreshAccumulator(buffer[0], [1, 2]); evaluator.RefreshAccumulator(buffer[1], [3]);
        double fraction = evaluator.GetNonZeroInputFraction([buffer[0], buffer[1]]);
        Assert.IsTrue(fraction is >= 0 and <= 1, "fraction in [0, 1]");
        ExceptionAssert.Throws<ArgumentException>(() => evaluator.GetNonZeroInputFraction([buffer[0], default]), "default accumulator rejected instead of crashing");
    }

    [TestMethod]
    public void DenseWeights_CanBeSetAndReadPerInputOutputPair()
    {
        var spec = NetworkSpec.CreateInt8(100, 64, [16, 32], outputBuckets: 2);
        Assert.IsTrue(spec.DenseLayerCount == 3, "hidden layers + output layer");
        var builder = new NnueModelBuilder(spec);
        builder.SetDenseWeight(1, 0, 70, 5, 42);       // first layer: 128 inputs (2 slots x 64), 16 outputs
        builder.SetDenseWeight(1, 2, 31, 0, -17);      // output layer: 32 inputs, 1 output
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => builder.SetDenseWeight(0, 0, 128, 0, 1), "input out of range");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => builder.SetDenseWeight(0, 2, 0, 1, 1), "output layer has one output");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => builder.SetDenseWeight(2, 0, 0, 0, 1), "bucket out of range");
        using var model = builder.Build();
        Assert.IsTrue(model.GetDenseWeight(1, 0, 70, 5) == 42 && model.GetDenseWeight(1, 2, 31, 0) == -17 && model.GetDenseWeight(0, 0, 70, 5) == 0, "values read back at the same coordinates");
        ExceptionAssert.Throws<InvalidOperationException>(() => RandomModel.Make(NetworkSpec.CreateSimple(100, 64), new Random(1)).GetDenseWeight(0, 0, 0, 0), "Int16 models have no dense Int8 layers");
    }

    [TestMethod]
    public void CheckedCalls_RefuseDisposedModelsAndBuffers()
    {
        var model = RandomModel.Make(NetworkSpec.CreateSimple(50, 64), new Random(8));
        var evaluator = new NnueEvaluator(model);
        var buffer = model.CreateAccumulators(3);
        evaluator.RefreshAccumulator(buffer[0], [1, 2]); evaluator.RefreshAccumulator(buffer[1], [3]);
        Accumulator first = buffer[0], second = buffer[1], third = buffer[2];
        buffer.Dispose();
        ExceptionAssert.Throws<ObjectDisposedException>(() => evaluator.Evaluate(first, second), "evaluate on a disposed buffer");
        ExceptionAssert.Throws<ObjectDisposedException>(() => evaluator.UpdateAccumulator(first, third, [1], [4]), "update on a disposed buffer");

        using var fresh = model.CreateAccumulators(2);
        Accumulator a = fresh[0], b = fresh[1];
        evaluator.RefreshAccumulator(a, [1]); evaluator.RefreshAccumulator(b, [2]);
        model.Dispose();
        ExceptionAssert.Throws<ObjectDisposedException>(() => evaluator.Evaluate(a, b), "evaluate with a disposed model");
        ExceptionAssert.Throws<ObjectDisposedException>(() => evaluator.RefreshAccumulator(a, [1]), "refresh with a disposed model");
        ExceptionAssert.Throws<ObjectDisposedException>(() => evaluator.AddFeature(a, 1), "add with a disposed model");
    }

    [TestMethod]
    public void InvalidSpecs_AreRejected()
    {
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { AccumulatorSize = 48 }), "H not a multiple of 32");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { Slots = 0 }), "zero slots");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { HiddenLayerSizes = [16] }), "hidden layers need Int8");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { Activation = FeatureTransformerActivation.Pairwise }), "pairwise needs Int8");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { Precision = NetworkPrecision.Int8, AccumulatorSize = 96, Slots = 1 }), "Int8: Slots*H must be a multiple of 64");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { Precision = NetworkPrecision.Int8, HiddenLayerSizes = [20] }), "hidden size not a multiple of 16");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { Precision = NetworkPrecision.Int8, SkipConnection = true }), "skip without hidden layer");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkLayout.Create(new NetworkSpec { OutputBuckets = 0 }), "zero buckets");
        ExceptionAssert.Throws<ArgumentException>(() => new NetworkSpec { Precision = (NetworkPrecision)7 }.Validate(), "undefined precision");
        ExceptionAssert.Throws<ArgumentException>(() => new NetworkSpec { Activation = (FeatureTransformerActivation)9 }.Validate(), "undefined activation");
        ExceptionAssert.Throws<ArgumentException>(() => new NetworkSpec { HiddenActivation = (HiddenActivation)4 }.Validate(), "undefined hidden activation");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkSpec.CreateInt8(100, 64, [NetworkSpec.MaxHiddenLayerSize + 16]), "hidden layer larger than the stack scratch allows");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkSpec.CreateBuilder(100).WithAccumulatorSize(64).WithInt8(FeatureTransformerActivation.ClippedReLU, [4096, 4096, 4096])
            .WithSqrClippedHiddenActivation().Build(), "too many hidden activations in total");
        var model = RandomModel.Make(NetworkSpec.CreateSimple(10, 32), new Random(1));
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => new NnueEvaluator(model, (SimdBackend)42), "undefined backend");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => new NnueEvaluator(model, sparseInput: (SparseInputMode)42), "undefined sparse mode");
    }

    // ------------------------------------------------------------------------------------------------ storage
    [TestMethod]
    public void Model_SurvivesSaveAndLoad()
    {
        var r = new Random(5);
        foreach (var (name, spec) in Specs().Where((_, i) => i % 3 == 0))
        {
            var model = RandomModel.Make(spec, r);
            using var ms = new MemoryStream();
            model.Save(ms);
            ms.Position = 0;
            var loaded = NnueModel.Load(ms);
            Assert.IsTrue(loaded.Spec == model.Spec, $"{name}: spec round trip");
            bool same = loaded.FeatureWeights.SequenceEqual(model.FeatureWeights) && loaded.FeatureBiases.SequenceEqual(model.FeatureBiases);
            for (int b = 0; b < spec.OutputBuckets; b++)
                if (spec.Precision == NetworkPrecision.Int16) same &= loaded.GetOutputWeights(b).SequenceEqual(model.GetOutputWeights(b)) && loaded.GetOutputBias(b) == model.GetOutputBias(b);
                else for (int l = 0; l < model.Layout.DenseLayers.Count; l++) same &= loaded.GetDenseWeights(b, l).SequenceEqual(model.GetDenseWeights(b, l)) && loaded.GetDenseBiases(b, l).SequenceEqual(model.GetDenseBiases(b, l));
            Assert.IsTrue(same, $"{name}: weights identical after save/load");

            var e1 = new NnueEvaluator(model); var e2 = new NnueEvaluator(loaded);
            var a = model.CreateAccumulators(spec.Slots); var b2 = loaded.CreateAccumulators(spec.Slots);
            var s1 = Enumerable.Range(0, spec.Slots).Select(i => a[i]).ToArray(); var s2 = Enumerable.Range(0, spec.Slots).Select(i => b2[i]).ToArray();
            for (int i = 0; i < spec.Slots; i++)
            {
                var f = Enumerable.Range(0, 12).Select(_ => r.Next(spec.FeatureCount)).Distinct().ToArray();
                e1.RefreshAccumulator(s1[i], f); e2.RefreshAccumulator(s2[i], f);
            }
            Assert.IsTrue(e1.Evaluate(s1, 0) == e2.Evaluate(s2, 0), $"{name}: same evaluation after save/load");
        }
        ExceptionAssert.Throws<InvalidDataException>(() => NnueModel.Load(new MemoryStream(new byte[64])), "garbage file rejected");
        ExceptionAssert.Throws<InvalidDataException>(() => { var ms = new MemoryStream(); var m = RandomModel.Make(new NetworkSpec { FeatureCount = 10, AccumulatorSize = 32 }, r); m.Save(ms); ms.SetLength(ms.Length - 100); ms.Position = 0; NnueModel.Load(ms); }, "truncated file rejected");
    }

    // ------------------------------------------------------------------------------------------------ accumulator buffer
    [TestMethod]
    public void AccumulatorBuffer_CopiesChecksBoundsAndDisposes()
    {
        var model = RandomModel.Make(new NetworkSpec { FeatureCount = 30, AccumulatorSize = 64, Slots = 2 }, new Random(3));
        var buf = model.CreateAccumulators(6);
        for (int i = 0; i < 6; i++) buf[i].AsSpan().Fill((short)(i + 1));
        buf.Copy(0, 3, 2);                                           // [1 2 3 | 1 2 6]
        Assert.IsTrue(buf[3].AsSpan()[0] == 1 && buf[4].AsSpan()[0] == 2 && buf[5].AsSpan()[0] == 6 && buf[2].AsSpan()[0] == 3, "Copy of a range");
        buf.Copy(0, 1, 3);                                           // overlapping, [1 1 2 3 ...]
        Assert.IsTrue(buf[0].AsSpan()[0] == 1 && buf[1].AsSpan()[0] == 1 && buf[2].AsSpan()[0] == 2 && buf[3].AsSpan()[0] == 3, "overlapping Copy behaves like memmove");
        Assert.IsTrue(((nuint)System.Runtime.CompilerServices.Unsafe.AsPointer(ref buf[2].AsSpan()[0])) % 64 == 0, "accumulators are 64-byte aligned");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => buf.Copy(5, 0, 2), "Copy range out of bounds");
        buf.Dispose(); buf.Dispose();                                // idempotent
        var m2 = RandomModel.Make(new NetworkSpec { FeatureCount = 30, AccumulatorSize = 64, Slots = 2 }, new Random(4));
        m2.Dispose(); m2.Dispose();
        ExceptionAssert.Throws<ObjectDisposedException>(() => m2.CreateAccumulators(1), "disposed model refuses new accumulators");
        ExceptionAssert.Throws<ObjectDisposedException>(() => { var _ = buf[0]; }, "disposed buffer refuses access");
        Assert.IsTrue(model.FtWeightsPtrValue() % 64 == 0, "model blob is 64-byte aligned");
        Assert.IsTrue(buf.IsDisposed, "IsDisposed after Dispose");
        Assert.IsTrue(default(Accumulator).IsEmpty && default(Accumulator).AsSpan().IsEmpty && !model.CreateAccumulator().IsEmpty, "IsEmpty / AsSpan of a default handle");
        using (var small = model.CreateAccumulators(3))
        {
            Assert.IsTrue(small[1] == small[1] && small[1] != small[2] && small[1].Equals(small[1]) && small[0].Length == small.AccumulatorSize, "handle equality, Length");
            Assert.IsTrue(small.Count == 3 && small.Select(a => a.Length).All(l => l == small.AccumulatorSize) && ((IReadOnlyList<Accumulator>)small)[2] == small[2], "buffer enumerates and indexes its accumulators");
            ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => small.Copy(2, 0, 2), "Copy: source range exceeds the buffer");
            ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => small.Copy(0, 2, 2), "Copy: destination range exceeds the buffer");
        }
    }
}

internal static class ModelExtensions
{
    /// <summary>The address of the weight blob (for the alignment test).</summary>
    public static unsafe nuint FtWeightsPtrValue(this NnueModel m) => (nuint)System.Runtime.CompilerServices.Unsafe.AsPointer(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(m.FeatureWeights)) - (nuint)m.Layout.FeatureWeightsOffset;
}
