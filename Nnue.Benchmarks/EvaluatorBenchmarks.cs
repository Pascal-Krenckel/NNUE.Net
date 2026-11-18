using BenchmarkDotNet.Attributes;
using Nnue.Backends;
using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;

namespace Nnue.Benchmarks;

/// <summary>
/// End-to-end cost of one <see cref="NnueEvaluator.Evaluate(Accumulator, Accumulator, int)"/> (two slots, the output bucket changes every call):
/// checked against unchecked, on every SIMD backend of this CPU. Int16 networks use H = 128 and 512 with 4 buckets, Int8 Stockfish-like
/// networks H = 256 and 1024 with 8 stacks, with dense (most activations non-zero) and sparse (most activations clamped to zero) accumulators.
/// </summary>
[Config(typeof(NoTieringConfig))]
public class EvaluatorBenchmarks
{
    private const int EvaluationsPerInvoke = 1024;

    public static IEnumerable<string> Scenarios => ["Int16 H=128", "Int16 H=512", "Int8 H=256 dense", "Int8 H=256 sparse", "Int8 H=1024 dense", "Int8 H=1024 sparse"];

    public static IEnumerable<SimdBackend> SupportedBackends => new[] { SimdBackend.Sse, SimdBackend.Avx2, SimdBackend.Avx512, SimdBackend.Avx512Vnni, SimdBackend.AvxVnni, SimdBackend.NeonDotProd, SimdBackend.Neon }.Where(SimdSupport.IsSupported);

    [ParamsSource(nameof(Scenarios))]
    public string Scenario { get; set; } = "";

    [ParamsSource(nameof(SupportedBackends))]
    public SimdBackend Backend { get; set; }

    private NnueModel _model = null!;
    private NnueEvaluator _evaluator = null!;
    private AccumulatorBuffer _accumulators = null!;
    private Accumulator _first, _second;
    private int _buckets;

    [GlobalSetup]
    public void Setup()
    {
        Random random = new(12);
        bool int16 = Scenario.StartsWith("Int16");
        int size = int.Parse(Scenario.Split("H=")[1].Split(' ')[0]);
        bool sparse = Scenario.EndsWith("sparse");
        NetworkSpec spec = int16
            ? new NetworkSpec { FeatureCount = 768, AccumulatorSize = size, Slots = 2, OutputBuckets = 4 }
            : NetworkSpec.CreateStockfishLike(768, size, 32, 32, 8);
        _buckets = spec.OutputBuckets;
        _model = BenchmarkData.CreateRandomModel(spec, random);
        _evaluator = new NnueEvaluator(_model, Backend);
        _accumulators = _model.CreateAccumulators(2);
        _first = _accumulators[0];
        _second = _accumulators[1];
        (int low, int high) = sparse ? (-1200, 300) : (-300, 700);
        BenchmarkData.Fill(_first, random, low, high);
        BenchmarkData.Fill(_second, random, low, high);
    }

    [GlobalCleanup]
    public void Cleanup() { _accumulators.Dispose(); _model.Dispose(); }

    [Benchmark(Baseline = true, OperationsPerInvoke = EvaluationsPerInvoke)]
    public int Evaluate()
    {
        int sum = 0;
        for (int i = 0; i < EvaluationsPerInvoke; i++)
            sum += (i >> 3) % 2 == 0 ? _evaluator.Evaluate(_first, _second, i % _buckets) : _evaluator.Evaluate(_second, _first, i % _buckets);
        return sum;
    }

    [Benchmark(OperationsPerInvoke = EvaluationsPerInvoke)]
    public int EvaluateUnchecked()
    {
        int sum = 0;
        for (int i = 0; i < EvaluationsPerInvoke; i++)
            sum += (i >> 3) % 2 == 0 ? _evaluator.EvaluateUnchecked(_first, _second, i % _buckets) : _evaluator.EvaluateUnchecked(_second, _first, i % _buckets);
        return sum;
    }
}
