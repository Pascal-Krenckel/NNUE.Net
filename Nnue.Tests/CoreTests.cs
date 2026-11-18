using Nnue.Backends;
using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;

namespace Nnue.Tests;

[TestClass]
public sealed unsafe class CoreTests
{
    public static IEnumerable<(string Name, NetworkSpec Spec)> Specs()
    {
        NetworkSpec I16(int h, int k, int buckets = 1, FeatureTransformerActivation a = FeatureTransformerActivation.ClippedReLU) => new() { FeatureCount = 400, AccumulatorSize = h, Slots = k, OutputBuckets = buckets, Activation = a };
        NetworkSpec I8(int h, int k, FeatureTransformerActivation a, int[] layers, HiddenActivation hid = HiddenActivation.ClippedReLU, bool dense = false, bool skip = false, int buckets = 1) => new()
        {
            FeatureCount = 400,
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

    /// <summary>
    /// The model configurations as MSTest data rows. Only the name is passed: MSTest serializes row arguments (Test Explorer, other
    /// processes), and a <see cref="NetworkSpec"/> with its immutable array does not survive that, so tests look the spec up by name.
    /// </summary>
    public static IEnumerable<object[]> SpecCases() => Specs().Select(s => new object[] { s.Name });

    private static NetworkSpec SpecNamed(string name) => Specs().First(s => s.Name == name).Spec;

    public static string SpecCaseName(System.Reflection.MethodInfo method, object[] data) => $"{method.Name} ({data[0]})";

    // ------------------------------------------------------------------------------------------------ backends vs scalar
    private const int BackendTrials = 400, AccumulatorTrials = 40;

    [TestMethod]
    [DynamicData(nameof(SpecCases), DynamicDataDisplayName = nameof(SpecCaseName))]
    public void Backends_MatchTheScalarReferenceBitForBit(string name)
    {
        NetworkSpec spec = SpecNamed(name);
        int trials = BackendTrials;
        SimdBackend[] backends = [.. SimdSupport.AvailableForTesting];
        Random r = new(123 + (name.GetHashCode() % 1000));
        {
            NnueModel model = RandomModel.Make(spec, r);
            NnueEvaluator reference = new(model, SimdBackend.Scalar, SparseInputMode.Off);
            List<(NnueEvaluator E, string Label)> evals = [];
            foreach (SimdBackend b in backends)
                foreach (SparseInputMode mode in spec.Precision == NetworkPrecision.Int8 ? Enum.GetValues<SparseInputMode>() : [SparseInputMode.Off])
                    evals.Add((new NnueEvaluator(model, b, mode), $"{b}/{mode}"));
            AccumulatorBuffer acc = model.CreateAccumulators(spec.Slots);
            Accumulator[] slots = [.. Enumerable.Range(0, spec.Slots).Select(i => acc[i])];
            HashSet<string> bad = [];
            for (int t = 0; t < trials; t++)
            {
                int pattern = t % 5;
                foreach (Accumulator s in slots)
                    RandomModel.Fill(s, r, pattern);
                int bucket = r.Next(spec.OutputBuckets);
                int expect = reference.EvaluateRaw(slots, bucket);
                foreach ((NnueEvaluator? e, string? label) in evals)
                    if (e.EvaluateRaw(slots, bucket) != expect)
                        _ = bad.Add(label);
            }
            Assert.IsEmpty(bad, $"{name}: mismatching backends: {string.Join(", ", bad)}");
        }
    }

    // ------------------------------------------------------------------------------------------------ accumulator operations
    [TestMethod]
    [DynamicData(nameof(SpecCases), DynamicDataDisplayName = nameof(SpecCaseName))]
    public void AccumulatorOperations_RefreshAddRemoveAndUpdateAgree(string name)
    {
        NetworkSpec spec = SpecNamed(name);
        int trials = AccumulatorTrials;
        Random r = new(7);
        {
            NnueModel model = RandomModel.Make(spec, r);
            NnueEvaluator reference = new(model, SimdBackend.Scalar);
            int F = spec.FeatureCount;
            foreach (SimdBackend b in SimdSupport.AvailableForTesting)
            {
                NnueEvaluator e = new(model, b);
                AccumulatorBuffer buf = model.CreateAccumulators(4);
                Accumulator inc = buf[0];
                Accumulator viaUpdate = buf[1];
                Accumulator fresh = buf[2];
                Accumulator expect = buf[3];
                int mismatches = 0;
                for (int t = 0; t < trials; t++)
                {
                    int[] a = [.. Enumerable.Range(0, r.Next(1, 20)).Select(_ => r.Next(F)).Distinct()];
                    int[] bset = [.. Enumerable.Range(0, r.Next(1, 20)).Select(_ => r.Next(F)).Distinct()];
                    int[] removed = [.. a.Except(bset)];
                    int[] added = [.. bset.Except(a)];

                    e.RefreshAccumulator(inc, a);
                    foreach (int f in removed)
                        e.RemoveFeature(inc, f);
                    foreach (int f in added)
                        e.AddFeature(inc, f);

                    e.RefreshAccumulator(viaUpdate, a);
                    e.UpdateAccumulator(viaUpdate, viaUpdate, removed, added);

                    e.RefreshAccumulator(fresh, bset);
                    reference.RefreshAccumulator(expect, bset);

                    if (!inc.AsSpan().SequenceEqual(fresh.AsSpan()) || !viaUpdate.AsSpan().SequenceEqual(fresh.AsSpan()) || !fresh.AsSpan().SequenceEqual(expect.AsSpan()))
                        mismatches++;
                }
                Assert.AreEqual(0, mismatches, $"{name} on {b}: refresh / add+remove / update disagree in {mismatches} of {trials} trials");
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ checked vs unchecked
    [TestMethod]
    public void Evaluator_ValidatesArguments_AndUncheckedMatchesChecked()
    {
        Random r = new(9);
        NetworkSpec spec = new()
        { FeatureCount = 100, AccumulatorSize = 64, Slots = 2, OutputBuckets = 3 };
        NnueModel model = RandomModel.Make(spec, r);
        NnueModel other = RandomModel.Make(new NetworkSpec { FeatureCount = 100, AccumulatorSize = 96, Slots = 2 }, r);
        NnueEvaluator e = new(model);
        AccumulatorBuffer buf = model.CreateAccumulators(3);
        Accumulator foreign = other.CreateAccumulator();
        e.RefreshAccumulator(buf[0], [1, 2, 3]);
        e.RefreshAccumulator(buf[1], [4, 5]);

        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.RefreshAccumulator(buf[0], [100]), "feature == FeatureCount");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.RefreshAccumulator(buf[0], [-1]), "negative feature");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.UpdateAccumulator(buf[0], buf[1], [1], [1000]), "update: added feature out of range");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.AddFeature(buf[0], 100), "add: feature out of range");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.Evaluate(buf[0], buf[1], 3), "bucket == OutputBuckets");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => e.Evaluate(buf[0], buf[1], -1), "negative bucket");
        ExceptionAssert.Throws<ArgumentException>(() => e.Evaluate(buf[0], 0), "one slot given, model needs two");
        ExceptionAssert.Throws<ArgumentException>(() => e.Evaluate([buf[0], buf[1], buf[2]], 0), "three slots given, model needs two");
        ExceptionAssert.Throws<ArgumentException>(() => e.Evaluate(buf[0], foreign, 0), "accumulator of another model (size mismatch)");
        ExceptionAssert.Throws<ArgumentException>(() => e.ResetAccumulator(default), "default accumulator");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => { Accumulator _ = buf[3]; }, "buffer index out of range");

        // unchecked and checked give the same result on valid input
        for (int bucket = 0; bucket < 3; bucket++)
        {
            Assert.AreEqual(e.EvaluateUnchecked(buf[0], buf[1], bucket), e.Evaluate(buf[0], buf[1], bucket), $"Evaluate == EvaluateUnchecked, bucket {bucket}");
            Assert.AreEqual(e.Evaluate(buf[0], buf[1], bucket), e.Evaluate([buf[0], buf[1]], bucket), $"span overload == pair overload, bucket {bucket}");
        }
        e.UpdateAccumulatorUnchecked(buf[0], buf[2], [1], [7]);
        e.UpdateAccumulator(buf[0], buf[1], [1], [7]);
        Assert.IsTrue(buf[1].AsSpan().SequenceEqual(buf[2].AsSpan()), "Update == UpdateAccumulatorUnchecked");
        int viaFeatures = e.EvaluateFeatures([[2, 3, 7], [4, 5]], 1);
        e.RefreshAccumulator(buf[0], [2, 3, 7]);
        e.RefreshAccumulator(buf[1], [4, 5]);
        Assert.AreEqual(e.Evaluate(buf[0], buf[1], 1), viaFeatures, "Evaluate(feature lists) == Evaluate(accumulators)");

        // a model with one slot accepts the single-accumulator overload only
        NnueModel one = RandomModel.Make(new NetworkSpec { FeatureCount = 50, AccumulatorSize = 64, Slots = 1 }, r);
        NnueEvaluator e1 = new(one);
        Accumulator a1 = one.CreateAccumulator();
        e1.RefreshAccumulator(a1, [3, 4]);
        Assert.AreEqual(e1.EvaluateUnchecked(a1), e1.Evaluate(a1), "single slot: checked == unchecked");
        ExceptionAssert.Throws<ArgumentException>(() => e1.Evaluate(a1, a1, 0), "single-slot model rejects two slots");
    }

    // ------------------------------------------------------------------------------------------------ layout, alignment, validation
    [TestMethod]
    [DynamicData(nameof(SpecCases), DynamicDataDisplayName = nameof(SpecCaseName))]
    public void Layout_StartsEveryTensorOn64ByteBoundaries(string name)
    {
        NetworkSpec spec = SpecNamed(name);
        {
            NetworkLayout l = NetworkLayout.Create(spec);
            bool aligned = l.FeatureWeightsOffset % 64 == 0 && l.FeatureBiasOffset % 64 == 0 && l.BlobSize % 64 == 0;
            if (spec.Precision == NetworkPrecision.Int16)
                aligned &= l.OutputWeightsOffset % 64 == 0 && l.OutputBiasOffset % 64 == 0 && l.OutputWeightsBucketStride % 64 == 0;
            else
                for (int b = 0; b < spec.OutputBuckets; b++)
                    for (int k = 0; k < l.DenseLayers.Count; k++)
                        aligned &= l.GetDenseWeightsOffset(b, k) % 64 == 0 && l.GetDenseBiasOffset(b, k) % 64 == 0;
            Assert.IsTrue(aligned, $"{name}: all tensors start on 64-byte boundaries");
            NnueModel m = RandomModel.Make(spec, new Random(1));
            Assert.AreEqual(0, (long)m.FtWeightsPtrValue() % 64, $"{name}: blob base is 64-byte aligned");
        }
    }

    [TestMethod]
    public void NetworkSpec_HasValueEquality()
    {
        NetworkSpec a = NetworkSpec.CreateInt8(768, 256, [16, 32]);
        NetworkSpec b = NetworkSpec.CreateInt8(768, 256, [16, 32]);
        Assert.IsTrue(a == b && a.Equals(b) && a.GetHashCode() == b.GetHashCode(), "equal specs with separately created layer lists");
        Assert.IsTrue(a != (a with { HiddenLayerSizes = [16, 16] }), "different layer sizes");
        Assert.IsTrue(a == (a with { ActivationQuantization = a.DefaultActivationQuantization }), "explicit default quantization equals the implicit one");
        Assert.Contains("HiddenLayerSizes = [16, 32]", a.ToString(), "ToString shows the layer sizes");
        Assert.IsTrue(new NetworkSpec { HiddenLayerSizes = default }.HiddenLayerSizes.IsEmpty, "default(ImmutableArray) becomes empty");
    }

    [TestMethod]
    public void GetNonZeroInputFraction_ValidatesAccumulators()
    {
        using NnueModel model = RandomModel.Make(NetworkSpec.CreateInt8(100, 64, [16]), new Random(3));
        NnueEvaluator evaluator = new(model);
        using AccumulatorBuffer buffer = model.CreateAccumulators(2);
        evaluator.RefreshAccumulator(buffer[0], [1, 2]);
        evaluator.RefreshAccumulator(buffer[1], [3]);
        double fraction = evaluator.GetNonZeroInputFraction([buffer[0], buffer[1]]);
        Assert.IsTrue(fraction is >= 0 and <= 1, "fraction in [0, 1]");
        ExceptionAssert.Throws<ArgumentException>(() => evaluator.GetNonZeroInputFraction([buffer[0], default]), "default accumulator rejected instead of crashing");
    }

    [TestMethod]
    public void DenseWeights_CanBeSetAndReadPerInputOutputPair()
    {
        NetworkSpec spec = NetworkSpec.CreateInt8(100, 64, [16, 32], outputBuckets: 2);
        Assert.AreEqual(3, spec.DenseLayerCount, "hidden layers + output layer");
        NnueModelBuilder builder = new(spec);
        builder.SetDenseWeight(1, 0, 70, 5, 42);       // first layer: 128 inputs (2 slots x 64), 16 outputs
        builder.SetDenseWeight(1, 2, 31, 0, -17);      // output layer: 32 inputs, 1 output
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => builder.SetDenseWeight(0, 0, 128, 0, 1), "input out of range");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => builder.SetDenseWeight(0, 2, 0, 1, 1), "output layer has one output");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => builder.SetDenseWeight(2, 0, 0, 0, 1), "bucket out of range");
        using NnueModel model = builder.Build();
        Assert.IsTrue(model.GetDenseWeight(1, 0, 70, 5) == 42 && model.GetDenseWeight(1, 2, 31, 0) == -17 && model.GetDenseWeight(0, 0, 70, 5) == 0, "values read back at the same coordinates");
        ExceptionAssert.Throws<InvalidOperationException>(() => RandomModel.Make(NetworkSpec.CreateSimple(100, 64), new Random(1)).GetDenseWeight(0, 0, 0, 0), "Int16 models have no dense Int8 layers");
    }

    [TestMethod]
    public void CheckedCalls_RefuseDisposedModelsAndBuffers()
    {
        NnueModel model = RandomModel.Make(NetworkSpec.CreateSimple(50, 64), new Random(8));
        NnueEvaluator evaluator = new(model);
        AccumulatorBuffer buffer = model.CreateAccumulators(3);
        evaluator.RefreshAccumulator(buffer[0], [1, 2]);
        evaluator.RefreshAccumulator(buffer[1], [3]);
        Accumulator first = buffer[0], second = buffer[1], third = buffer[2];
        buffer.Dispose();
        ExceptionAssert.Throws<ObjectDisposedException>(() => evaluator.Evaluate(first, second), "evaluate on a disposed buffer");
        ExceptionAssert.Throws<ObjectDisposedException>(() => evaluator.UpdateAccumulator(first, third, [1], [4]), "update on a disposed buffer");

        using AccumulatorBuffer fresh = model.CreateAccumulators(2);
        Accumulator a = fresh[0], b = fresh[1];
        evaluator.RefreshAccumulator(a, [1]);
        evaluator.RefreshAccumulator(b, [2]);
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
        NnueModel model = RandomModel.Make(NetworkSpec.CreateSimple(10, 32), new Random(1));
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => _ = new NnueEvaluator(model, (SimdBackend)42), "undefined backend");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => _ =new NnueEvaluator(model, sparseInput: (SparseInputMode)42), "undefined sparse mode");
    }

    // ------------------------------------------------------------------------------------------------ storage
    [TestMethod]
    public void Model_SurvivesSaveAndLoad()
    {
        Random r = new(5);
        foreach ((string? name, NetworkSpec? spec) in Specs().Where((_, i) => i % 3 == 0))
        {
            NnueModel model = RandomModel.Make(spec, r);
            using MemoryStream ms = new();
            model.Save(ms);
            ms.Position = 0;
            NnueModel loaded = NnueModel.Load(ms);
            Assert.IsTrue(loaded.Spec == model.Spec, $"{name}: spec round trip");
            bool same = loaded.FeatureWeights.SequenceEqual(model.FeatureWeights) && loaded.FeatureBiases.SequenceEqual(model.FeatureBiases);
            for (int b = 0; b < spec.OutputBuckets; b++)
                if (spec.Precision == NetworkPrecision.Int16)
                    same &= loaded.GetOutputWeights(b).SequenceEqual(model.GetOutputWeights(b)) && loaded.GetOutputBias(b) == model.GetOutputBias(b);
                else
                    for (int l = 0; l < model.Layout.DenseLayers.Count; l++)
                        same &= loaded.GetDenseWeights(b, l).SequenceEqual(model.GetDenseWeights(b, l)) && loaded.GetDenseBiases(b, l).SequenceEqual(model.GetDenseBiases(b, l));
            Assert.IsTrue(same, $"{name}: weights identical after save/load");

            NnueEvaluator e1 = new(model);
            NnueEvaluator e2 = new(loaded);
            AccumulatorBuffer a = model.CreateAccumulators(spec.Slots);
            AccumulatorBuffer b2 = loaded.CreateAccumulators(spec.Slots);
            Accumulator[] s1 = [.. Enumerable.Range(0, spec.Slots).Select(i => a[i])];
            Accumulator[] s2 = [.. Enumerable.Range(0, spec.Slots).Select(i => b2[i])];
            for (int i = 0; i < spec.Slots; i++)
            {
                int[] f = [.. Enumerable.Range(0, 12).Select(_ => r.Next(spec.FeatureCount)).Distinct()];
                e1.RefreshAccumulator(s1[i], f);
                e2.RefreshAccumulator(s2[i], f);
            }
            Assert.AreEqual(e2.Evaluate(s2, 0), e1.Evaluate(s1, 0), $"{name}: same evaluation after save/load");
        }
        ExceptionAssert.Throws<InvalidDataException>(() => NnueModel.Load(new MemoryStream(new byte[64])), "garbage file rejected");
        ExceptionAssert.Throws<InvalidDataException>(() => { MemoryStream ms = new(); NnueModel m = RandomModel.Make(new NetworkSpec { FeatureCount = 10, AccumulatorSize = 32 }, r); m.Save(ms); ms.SetLength(ms.Length - 100); ms.Position = 0; _ = NnueModel.Load(ms); }, "truncated file rejected");
    }

    // ------------------------------------------------------------------------------------------------ accumulator buffer
    [TestMethod]
    public void AccumulatorBuffer_CopiesChecksBoundsAndDisposes()
    {
        NnueModel model = RandomModel.Make(new NetworkSpec { FeatureCount = 30, AccumulatorSize = 64, Slots = 2 }, new Random(3));
        AccumulatorBuffer buf = model.CreateAccumulators(6);
        for (int i = 0; i < 6; i++)
            buf[i].AsSpan().Fill((short)(i + 1));
        buf.Copy(0, 3, 2);                                           // [1 2 3 | 1 2 6]
        Assert.IsTrue(buf[3].AsSpan()[0] == 1 && buf[4].AsSpan()[0] == 2 && buf[5].AsSpan()[0] == 6 && buf[2].AsSpan()[0] == 3, "Copy of a range");
        buf.Copy(0, 1, 3);                                           // overlapping, [1 1 2 3 ...]
        Assert.IsTrue(buf[0].AsSpan()[0] == 1 && buf[1].AsSpan()[0] == 1 && buf[2].AsSpan()[0] == 2 && buf[3].AsSpan()[0] == 3, "overlapping Copy behaves like memmove");
        Assert.AreEqual(nuint.Zero, ((nuint)System.Runtime.CompilerServices.Unsafe.AsPointer(ref buf[2].AsSpan()[0])) % 64, "accumulators are 64-byte aligned");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => buf.Copy(5, 0, 2), "Copy range out of bounds");
        buf.Dispose();
        buf.Dispose();                                // idempotent
        NnueModel m2 = RandomModel.Make(new NetworkSpec { FeatureCount = 30, AccumulatorSize = 64, Slots = 2 }, new Random(4));
        m2.Dispose();
        m2.Dispose();
        ExceptionAssert.Throws<ObjectDisposedException>(() => m2.CreateAccumulators(1), "disposed model refuses new accumulators");
        ExceptionAssert.Throws<ObjectDisposedException>(() => { Accumulator _ = buf[0]; }, "disposed buffer refuses access");
        Assert.AreEqual(nuint.Zero, model.FtWeightsPtrValue() % 64, "model blob is 64-byte aligned");
        Assert.IsTrue(buf.IsDisposed, "IsDisposed after Dispose");
        Assert.IsTrue(default(Accumulator).IsEmpty && default(Accumulator).AsSpan().IsEmpty && !model.CreateAccumulator().IsEmpty, "IsEmpty / AsSpan of a default handle");
        using AccumulatorBuffer small = model.CreateAccumulators(3);
        Assert.IsTrue(small[1] == small[1] && small[1] != small[2] && small[1].Equals(small[1]) && small[0].Length == small.AccumulatorSize, "handle equality, Length");
        Assert.IsTrue(small.Count == 3 && small.Select(a => a.Length).All(l => l == small.AccumulatorSize) && ((IReadOnlyList<Accumulator>)small)[2] == small[2], "buffer enumerates and indexes its accumulators");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => small.Copy(2, 0, 2), "Copy: source range exceeds the buffer");
        ExceptionAssert.Throws<ArgumentOutOfRangeException>(() => small.Copy(0, 2, 2), "Copy: destination range exceeds the buffer");
    }
}

internal static class ModelExtensions
{
    /// <summary>The address of the weight blob (for the alignment test).</summary>
    public static unsafe nuint FtWeightsPtrValue(this NnueModel m) => (nuint)System.Runtime.CompilerServices.Unsafe.AsPointer(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(m.FeatureWeights)) - (nuint)m.Layout.FeatureWeightsOffset;
}
