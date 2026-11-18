# Development

## Repository layout

```
Nnue.sln
Directory.Build.props        shared compiler settings (C# latest, nullable, implicit usings, unsafe code)
.editorconfig  .gitignore
.github/workflows/
  ci.yml                     build and test on Linux x64/ARM64, Windows and macOS (ARM64) for every push and pull request
  publish.yml                build, test, pack and push the NuGet package for a version tag
Nnue/                        the library (package Nnue.Net, assembly Nnue), one type per file, folder = namespace
  Configuration/  Models/  Evaluation/  Backends/  Kernels/  Interop/  Memory/
  Training/Models/  Training/Optimization/  Training/Quantization/
  PACKAGE.md                 the readme shown on nuget.org
Nnue.Tests/                  MSTest tests
Nnue.Demo/                   runnable example: train, quantize, save, load, evaluate
Nnue.Benchmarks/             BenchmarkDotNet benchmarks and RESULTS.md
docs/                        this documentation
```

`Kernels`, `Memory` and the backend structs are internal; the tests and benchmarks see them through `InternalsVisibleTo`.

## Building

The library targets `net8.0` and `net10.0`, and `net11.0` when the .NET 11 SDK builds it (the target is added automatically). The
net11.0 build is the only one with the AVX512-VNNI backend, so packages should be built with the .NET 11 SDK; the workflows do that.

```
dotnet build -c Release
dotnet run -c Release --project Nnue.Demo
dotnet pack Nnue -c Release -o artifacts          # Nnue.Net.<version>.nupkg + .snupkg
```

The package contains the assemblies of every target, their XML documentation and `PACKAGE.md`; the symbol package has source link
information. The library is marked trimming- and AOT-compatible, and the analyzers report nothing.

## Tests

```
dotnet test                                            # every target framework, every test
dotnet test --filter "TestCategory!=Slow"              # without the end-to-end training test
dotnet test --filter "FullyQualifiedName~Backends"     # one group
```

| Class | Covers |
|---|---|
| `CoreTests` | tensor alignment per model configuration, spec validation and equality, checked against unchecked API, disposal, accumulator buffers, save/load, accumulator operations, and **every backend × sparse mode against the scalar reference, bit for bit**, for 19 model configurations |
| `LoadingTests` | loading from memory, non-seekable streams, async and memory-mapped; corrupt files; presets and builder |
| `InteropTests` | the bullet format: layouts, bias types, evaluation against a scalar reference, byte-exact export |
| `TrainingTests` | gradients against finite differences for every activation and topology option, hand-computed outputs of hand-built networks, bucket routing, quantizer report, progress and cancellation, option and sample validation, training + quantization end to end (`Slow`) |

The backend tests run every backend the CPU supports, including the emulated VNNI backends, so x86 machines without VNNI still
test the VNNI kernel structure.

## Benchmarks

```
dotnet run -c Release --project Nnue.Benchmarks -- --list flat
dotnet run -c Release --project Nnue.Benchmarks -- --filter "*EvaluatorBenchmarks*"
```

| Class | Measures |
|---|---|
| `EvaluatorBenchmarks` | checked and unchecked `Evaluate` on every SIMD backend of the CPU; Int16 H=128/512, Int8 Stockfish-like H=256/1024, dense and sparse activations |
| `AccumulatorBenchmarks` | update (2 removed + 2 added) and refresh (16 / 32 features), checked and unchecked, small and cache-exceeding feature matrices |
| `Int8StageBenchmarks` | every stage of an Int8 evaluation on its own (needs AVX-512) |

The jobs run with tiered compilation off. `RESULTS.md` records the measurements behind the design decisions; most of them come
from an earlier hand-written harness that interleaved the variants.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request: restore, build, all tests and the demo, on
`ubuntu-latest` (x64), `ubuntu-24.04-arm` (ARM64, NEON), `windows-latest` and `macos-latest` (ARM64, NEON with dot product), with
the .NET 8, 10 and 11 SDKs installed so that every target framework is tested.

## Releasing to nuget.org

`.github/workflows/publish.yml` builds, tests, packs and pushes `Nnue.Net`.

The workflow uses nuget.org **trusted publishing**: no API key is stored in GitHub. The job asks GitHub for an OIDC token, the
`NuGet/login` action exchanges it for a temporary API key (valid for one hour, requested right before the push), and nuget.org only
accepts it for the repository, workflow file and environment named in your policy.

One-time setup:

1. On nuget.org, open your user menu → *Trusted Publishing* and add a policy:
   - *Repository Owner*: the GitHub user or organization that owns the repository
   - *Repository*: the repository name
   - *Workflow File*: `publish.yml` (the file name only)
   - *Environment*: `nuget` (the workflow runs in this GitHub environment; the value must match)

   The policy applies to the packages of the owner you select in it (you or an organization). For a private repository the policy
   stays provisional for 7 days until the first successful publish.
2. In the GitHub repository, add the secret `NUGET_USER` with your nuget.org **profile name** (not the e-mail address)
   (*Settings → Secrets and variables → Actions*).
3. Optionally add required reviewers to the environment `nuget` (*Settings → Environments*), so every release waits for approval.
4. Fill in `Authors`, and if wanted `PackageLicenseExpression` and `PackageProjectUrl`, in `Nnue/Nnue.csproj`. The repository URL is
   taken from git by source link.

Releasing a version:

```
git tag v0.1.0
git push origin v0.1.0
```

The tag name without the `v` becomes the package version (`v1.2.3-beta.1` → `1.2.3-beta.1`). The workflow can also be started by
hand (*Actions → Publish NuGet package → Run workflow*) with an explicit version. Pushing a version that already exists is skipped. If the push fails with an authorization error, compare the policy on nuget.org
(owner, repository, `publish.yml`, environment `nuget`) with the repository.
The `.nupkg` and `.snupkg` are attached to the workflow run as an artifact.

## Verification status

- x86-64: all tests pass on net8.0 and net10.0 with every x86 backend, including both emulated VNNI backends.
- The net11.0 build and the hardware AVX512-VNNI path were compile-checked against the .NET 11 API but first run in CI or on a
  machine with the .NET 11 SDK and an AVX512-VNNI CPU (Ice Lake, Sapphire Rapids, Zen 4 or newer).
- ARM64: the NEON kernels are run by the ARM64 jobs of the CI workflow; they have not been measured on real ARM hardware.
- The bullet importer is tested against hand-written files, not against a real bullet training run.

## Conventions

- C# naming throughout: PascalCase, no abbreviations, verbs for operations, `Get…` for computed values, `…Length`/`…Count` for
  sizes. Kernels take `(source, destination, length)` in that order.
- One type per file; the folder is the namespace.
- Every public member has XML documentation (`GenerateDocumentationFile` is on, missing documentation is a warning).
- Kernels take raw pointers and assume validated, aligned input; validation happens once, in the evaluator.
