using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;

namespace Nnue.Tests;

[TestClass]
public sealed class LoadingTests
{
    /// <summary>A stream that returns at most a few bytes per read and cannot seek, like a network stream.</summary>
    private sealed class TrickleStream(byte[] data, int step) : Stream
    {
        private int _position;
        public override bool CanRead => true; public override bool CanSeek => false; public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => _position; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
        public override int Read(Span<byte> buffer)
        {
            int n = Math.Min(Math.Min(buffer.Length, step), data.Length - _position);
            data.AsSpan(_position, n).CopyTo(buffer);
            _position += n;
            return n;
        }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private static int Evaluate(NnueModel model, int[][] features, int bucket)
    {
        NnueEvaluator evaluator = new(model);
        using AccumulatorBuffer buffer = model.CreateAccumulators(model.Spec.Slots);
        Accumulator[] slots = new Accumulator[model.Spec.Slots];
        for (int i = 0; i < slots.Length; i++)
        { slots[i] = buffer[i]; evaluator.RefreshAccumulator(slots[i], features[i]); }
        return evaluator.Evaluate(slots, bucket);
    }

    [TestMethod]
    public void Model_LoadsFromMemoryStreamsAsyncAndMemoryMap()
    {
        Random random = new(11);
        string path = Path.Combine(Path.GetTempPath(), $"nnue-test-{Guid.NewGuid():N}.nnue");
        try
        {
            foreach ((string? name, NetworkSpec? spec) in CoreTests.Specs().Where((_, i) => i % 4 == 1))
            {
                NnueModel model = RandomModel.Make(spec, random);
                model.Save(path);
                byte[] bytes = File.ReadAllBytes(path);
                int[][] features = Enumerable.Range(0, spec.Slots).Select(_ => Enumerable.Range(0, 10).Select(_ => random.Next(spec.FeatureCount)).Distinct().ToArray()).ToArray();
                int bucket = random.Next(spec.OutputBuckets);
                int expected = Evaluate(model, features, bucket);

                Assert.IsTrue(Evaluate(NnueModel.Load(bytes), features, bucket) == expected, $"{name}: load from memory");
                Assert.IsTrue(Evaluate(NnueModel.Load(new TrickleStream(bytes, 7)), features, bucket) == expected, $"{name}: load from a non-seekable stream with tiny reads");
                Assert.IsTrue(Evaluate(NnueModel.LoadAsync(path).GetAwaiter().GetResult(), features, bucket) == expected, $"{name}: async load from a file");
                Assert.IsTrue(Evaluate(NnueModel.LoadAsync(new TrickleStream(bytes, 1000)).GetAwaiter().GetResult(), features, bucket) == expected, $"{name}: async load from a trickling stream");

                using NnueModel mapped = NnueModel.LoadMemoryMapped(path);
                Assert.IsTrue(Evaluate(mapped, features, bucket) == expected, $"{name}: memory-mapped model evaluates identically");
                Assert.IsTrue(mapped.FeatureWeights.SequenceEqual(model.FeatureWeights) && mapped.FtWeightsPtrValue() % 64 == 0, $"{name}: mapped weights identical and 64-byte aligned");
            }

            // corrupt / truncated input on every path
            NnueModel small = RandomModel.Make(new NetworkSpec { FeatureCount = 10, AccumulatorSize = 32 }, random);
            small.Save(path);
            byte[] good = File.ReadAllBytes(path);
            byte[] truncated = good[..^100];
            ExceptionAssert.Throws<InvalidDataException>(() => NnueModel.Load(truncated), "memory: truncated");
            ExceptionAssert.Throws<InvalidDataException>(() => NnueModel.Load(new byte[40]), "memory: garbage");
            ExceptionAssert.Throws<InvalidDataException>(() => NnueModel.LoadAsync(new MemoryStream(truncated)).GetAwaiter().GetResult(), "async: truncated");
            File.WriteAllBytes(path, truncated);
            ExceptionAssert.Throws<InvalidDataException>(() => { using NnueModel unexpected = NnueModel.LoadMemoryMapped(path); }, "mapped: truncated");
            File.WriteAllBytes(path, new byte[1000]);
            ExceptionAssert.Throws<InvalidDataException>(() => { using NnueModel unexpected = NnueModel.LoadMemoryMapped(path); }, "mapped: garbage");
            CancellationTokenSource cancelled = new();
            cancelled.Cancel();
            ExceptionAssert.Throws<OperationCanceledException>(() => NnueModel.LoadAsync(new MemoryStream(good), cancelled.Token).GetAwaiter().GetResult(), "async: cancelled token");

            // disposing a mapped model releases the mapping and invalidates the model like any other
            File.WriteAllBytes(path, good);
            NnueModel mappedModel = NnueModel.LoadMemoryMapped(path);
            mappedModel.Dispose();
            Assert.IsTrue(mappedModel.IsDisposed, "mapped model: IsDisposed");
            ExceptionAssert.Throws<ObjectDisposedException>(() => mappedModel.CreateAccumulators(1), "mapped model: disposed refuses accumulators");
            File.Delete(path);
            Assert.IsTrue(!File.Exists(path), "the mapping no longer holds the file");
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [TestMethod]
    public void Load_LeavesASeekableStreamRightAfterTheModel()
    {
        NnueModel tiny = RandomModel.Make(new NetworkSpec { FeatureCount = 1, AccumulatorSize = 32 }, new Random(4));     // smaller than the header probe
        using MemoryStream stream = new();
        tiny.Save(stream);
        long end = stream.Position;
        stream.Write([1, 2, 3, 4, 5, 6, 7, 8]);                    // whatever follows the model in the stream
        stream.Position = 0;
        using NnueModel loaded = NnueModel.Load(stream);
        Assert.IsTrue(stream.Position == end, $"position {stream.Position}, model ends at {end}");
        stream.Position = 0;
        using NnueModel loadedAsync = NnueModel.LoadAsync(stream).GetAwaiter().GetResult();
        Assert.IsTrue(stream.Position == end, "async: position right after the model");
        Assert.IsTrue(loaded.FeatureWeights.SequenceEqual(tiny.FeatureWeights), "weights");
    }

    [TestMethod]
    public void Presets_AndBuilder_ProduceValidSpecs()
    {
        NetworkSpec simple = NetworkSpec.CreateSimple(768, 128);
        Assert.IsTrue(simple.Precision == NetworkPrecision.Int16 && simple.Slots == 2 && simple.OutputBuckets == 1 && simple.TryValidate(out _), "CreateSimple");
        NetworkSpec screlu = NetworkSpec.CreateSimple(768, 128, slots: 1, outputBuckets: 8, FeatureTransformerActivation.SquaredClippedReLU);
        Assert.IsTrue(screlu.Activation == FeatureTransformerActivation.SquaredClippedReLU && screlu.OutputWeightQuantization == 64 && screlu.Slots == 1, "CreateSimple with SCReLU");
        NetworkSpec int8 = NetworkSpec.CreateInt8(768, 256, [16, 32], outputBuckets: 4);
        Assert.IsTrue(int8.Precision == NetworkPrecision.Int8 && int8.HiddenLayerSizes.SequenceEqual([16, 32]) && int8.OutputBuckets == 4, "CreateInt8");
        NetworkSpec viaBuilder = NetworkSpec.CreateBuilder(500).WithAccumulatorSize(128).WithSlots(2).WithInt8(FeatureTransformerActivation.Pairwise, [16, 32])
            .WithSqrClippedHiddenActivation().WithDenseConnections().WithSkipConnection().WithOutputBuckets(3).WithOutputScale(300).Build();
        Assert.IsTrue(viaBuilder.Activation == FeatureTransformerActivation.Pairwise && viaBuilder.DenseConnections && viaBuilder.SkipConnection && viaBuilder.OutputScale == 300
                && viaBuilder.HiddenActivation == HiddenActivation.SqrClippedReLU, "builder sets every option");
        Assert.IsTrue(RandomModel.Make(viaBuilder, new Random(1)).Spec.OutputBuckets == 3, "a built spec makes a model");
        Assert.IsTrue(NetworkSpec.CreateBuilder(100).WithQuantization(255, 64).WithAccumulatorSize(64).Build().OutputWeightQuantization == 64, "WithQuantization");

        ExceptionAssert.Throws<ArgumentException>(() => NetworkSpec.CreateBuilder(100).WithAccumulatorSize(48).Build(), "builder: accumulator size not a multiple of 32");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkSpec.CreateBuilder(100).WithInt16(FeatureTransformerActivation.Pairwise).Build(), "builder: pairwise needs Int8");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkSpec.CreateBuilder(100).WithSkipConnection().Build(), "builder: skip connection needs Int8 and a hidden layer");
        ExceptionAssert.Throws<ArgumentException>(() => NetworkSpec.CreateInt8(100, 64, [20]), "CreateInt8: hidden size not a multiple of 16");
        Assert.IsTrue(!new NetworkSpec { Slots = 0 }.TryValidate(out string? error) && error is { Length: > 0 }, "TryValidate reports instead of throwing");
        Assert.IsTrue(new NetworkSpec().TryValidate(out error) && error is null, "TryValidate on a valid spec");
    }
}
