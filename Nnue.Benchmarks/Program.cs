using BenchmarkDotNet.Running;

// Examples:
//   dotnet run -c Release --project Nnue.Benchmarks -- --filter *EvaluatorBenchmarks*
//   dotnet run -c Release --project Nnue.Benchmarks -- --list flat
BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
