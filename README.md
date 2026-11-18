# NNUE.Net

A feature-agnostic NNUE library for .NET 8, 10 and 11: evaluation, incremental updates, training, quantization and storage of
*efficiently updatable neural networks*, the kind of network Stockfish uses to evaluate chess positions.

**The library does not know what the features mean.** It is not tied to chess: a "perspective", a side to move, a score in
centipawns or a game result are concepts of the layer you build on top of it. It gives you:

- **Fast evaluation** with hand-written SIMD kernels for AVX512-VNNI, AVX-512, AVX-VNNI, AVX2, SSSE3, NEON (with and without
  dot product) and a scalar reference, all bit-identical, selected automatically for the CPU.
- **Incremental accumulator updates**: one call computes `child = parent - removed + added`; going back to the parent is free.
- **Two arithmetics**: a single int16 output layer (simple and accurate), or Stockfish-style int8 layers (uint8 activations, int8
  weights, any number of hidden layers, pairwise, pooling, squared and Quantmoid4 activations, dense and skip connections, output buckets).
- **Training and quantization** for small networks (float model, Adam trainer, gradient check, quantizer with overflow report).
- **Loading** from files, streams, memory, asynchronously or memory-mapped, and **import** of networks trained with bullet.

```
features ──(shared int16 weights)──▶ K accumulators ("slots") ──activation──▶ concat ──▶ [dense layers] × output buckets ──▶ score
```

## Installation

```
dotnet add package NNUE.Net
```

The package contains builds for net8.0, net10.0 and net11.0. The net11.0 build adds the AVX512-VNNI backend; otherwise the
builds are identical. The library is trimming- and Native AOT-compatible.

## Quick start

```csharp
using Nnue.Configuration;              // NetworkSpec
using Nnue.Evaluation;                 // NnueEvaluator, Accumulator
using Nnue.Models;                     // NnueModel
using Nnue.Training.Models;            // FloatModel                 (training only)
using Nnue.Training.Optimization;      // NnueTrainer, TrainingSample (training only)
using Nnue.Training.Quantization;      // NnueQuantizer              (training only)

// 1. Declare the network: 768 binary features, two accumulators of 256 values, Stockfish arithmetic with hidden layers 16 and 32.
var spec = NetworkSpec.CreateInt8(featureCount: 768, accumulatorSize: 256, hiddenLayerSizes: [16, 32], slots: 2, outputBuckets: 4);

// 2. Train in float, quantize, save.
var floatModel = new FloatModel(spec);
floatModel.Randomize();
new NnueTrainer(floatModel).Train(samples, epochs: 20);          // samples: IReadOnlyList<TrainingSample>
using var model = NnueQuantizer.Quantize(floatModel);
model.Save("net.nnue");

// 3. Evaluate.
var evaluator = new NnueEvaluator(model);                         // best SIMD backend of this CPU
using var accumulators = model.CreateAccumulators(256);           // e.g. one per slot and search ply
evaluator.RefreshAccumulator(accumulators[0], activeFeaturesOfSlot0);
evaluator.RefreshAccumulator(accumulators[1], activeFeaturesOfSlot1);
evaluator.UpdateAccumulator(accumulators[0], accumulators[2], removedFeatures, addedFeatures);
int score = evaluator.Evaluate(accumulators[2], accumulators[1], bucket);
```

`Nnue.Demo` is a complete, runnable example with a toy "position" abstraction that turns game objects into feature lists.

## Documentation

| Document | Contents |
|---|---|
| [User guide](docs/guide.md) | Concepts (features, slots, buckets), declaring networks, training, evaluating in a search, loading and saving, bullet import, threading, disposal, limits |
| [Architecture](docs/architecture.md) | Arithmetic and activations, SIMD backends, memory layout, the model file format, why the code looks the way it does |
| [Development](docs/development.md) | Repository layout, building, tests, benchmarks, continuous integration and releasing to nuget.org |
| [Benchmark results](Nnue.Benchmarks/RESULTS.md) | The measurements behind the design decisions |
| [Changelog](CHANGELOG.md) | Changes between versions |

Every public type and member has XML documentation, so the API reference is available in the IDE.

## Status

All tests pass on x86-64 with every x86 backend (including the emulated VNNI backends). The NEON kernels and the hardware
AVX512-VNNI path are covered by the CI matrix (ARM64 runners, .NET 11) rather than by local runs; see
[Development](docs/development.md#verification-status) for details.

Not implemented: Stockfish's `.nnue` format, weights from trainers other than bullet, factorised (virtual) features, streaming
datasets and sparse gradients for large trainers (the trainer is meant for small networks), models larger than 2 GB.
