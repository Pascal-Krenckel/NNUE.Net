using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;

namespace Nnue.Benchmarks;

/// <summary>Fully optimised code from the first call, no tiered-compilation noise: the benchmarked kernels are hot loops that must not run as tier-0 code.</summary>
public sealed class NoTieringConfig : ManualConfig
{
    public NoTieringConfig()
    {
        AddJob(Job.Default.WithEnvironmentVariable("DOTNET_TieredCompilation", "0").WithEnvironmentVariable("DOTNET_TieredPGO", "0"));
    }
}
