using BenchmarkDotNet.Attributes;
using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;

namespace Nnue.Benchmarks;

/// <summary>
/// Cost of the accumulator operations of the public evaluator, checked against unchecked: an incremental update (2 features removed, 2 added)
/// and a refresh from 16 / 32 features. Feature indices are random and change on every call, so the rows of the feature matrix are
/// read at random positions. Checked and unchecked variants run the same loop (the accumulator handles are taken from the buffer once, outside it) (with 22528 features and H = 1024 the matrix, 46 MB, does not fit in any cache).
/// </summary>
[Config(typeof(NoTieringConfig))]
public class AccumulatorBenchmarks
{
    private const int OperationsPerCall = 1024, PoolSize = 4096;

    [Params(768, 22528)]
    public int FeatureCount { get; set; }

    [Params(256, 1024)]
    public int AccumulatorSize { get; set; }

    private NnueModel _model = null!;
    private NnueEvaluator _evaluator = null!;
    private AccumulatorBuffer _accumulators = null!;
    private int[] _pool = null!;           // consecutive groups of 32 random features

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(5);
        _model = BenchmarkData.CreateRandomModel(NetworkSpec.CreateSimple(FeatureCount, AccumulatorSize, slots: 2), random);
        _evaluator = new NnueEvaluator(_model);
        _accumulators = _model.CreateAccumulators(2);
        BenchmarkData.Fill(_accumulators[0], random, -200, 200);
        _pool = Enumerable.Range(0, PoolSize * 32).Select(_ => random.Next(FeatureCount)).ToArray();
    }

    [GlobalCleanup]
    public void Cleanup() { _accumulators.Dispose(); _model.Dispose(); }

    private ReadOnlySpan<int> Group(int call) => _pool.AsSpan((call & (PoolSize - 1)) * 32, 32);

    [Benchmark(Baseline = true, OperationsPerInvoke = OperationsPerCall)]
    public void UpdateAccumulator()
    {
        var source = _accumulators[0]; var destination = _accumulators[1];
        for (int i = 0; i < OperationsPerCall; i++)
        {
            var features = Group(i);
            _evaluator.UpdateAccumulator(source, destination, features[..2], features.Slice(2, 2));
        }
    }

    [Benchmark(OperationsPerInvoke = OperationsPerCall)]
    public void UpdateAccumulatorUnchecked()
    {
        var source = _accumulators[0]; var destination = _accumulators[1];
        for (int i = 0; i < OperationsPerCall; i++)
        {
            var features = Group(i);
            _evaluator.UpdateAccumulatorUnchecked(source, destination, features[..2], features.Slice(2, 2));
        }
    }

    [Benchmark(OperationsPerInvoke = OperationsPerCall)]
    public void RefreshAccumulator16Features()
    {
        var destination = _accumulators[1];
        for (int i = 0; i < OperationsPerCall; i++) _evaluator.RefreshAccumulator(destination, Group(i)[..16]);
    }

    [Benchmark(OperationsPerInvoke = OperationsPerCall)]
    public void RefreshAccumulatorUnchecked16Features()
    {
        var destination = _accumulators[1];
        for (int i = 0; i < OperationsPerCall; i++) _evaluator.RefreshAccumulatorUnchecked(destination, Group(i)[..16]);
    }

    [Benchmark(OperationsPerInvoke = OperationsPerCall)]
    public void RefreshAccumulator32Features()
    {
        var destination = _accumulators[1];
        for (int i = 0; i < OperationsPerCall; i++) _evaluator.RefreshAccumulator(destination, Group(i));
    }

    [Benchmark(OperationsPerInvoke = OperationsPerCall)]
    public void RefreshAccumulatorUnchecked32Features()
    {
        var destination = _accumulators[1];
        for (int i = 0; i < OperationsPerCall; i++) _evaluator.RefreshAccumulatorUnchecked(destination, Group(i));
    }
}
