# Nnue.Net

A feature-agnostic NNUE library: evaluation, incremental updates, training, quantization and storage of *efficiently updatable
neural networks*, the kind of network Stockfish uses. The library does not know what the features mean, so it works for any game
or domain with sparse binary inputs.

- SIMD kernels for AVX512-VNNI (net11.0 build), AVX-512, AVX-VNNI, AVX2, SSSE3 and NEON, plus a scalar reference; all bit-identical,
  chosen automatically for the CPU.
- Incremental accumulator updates: `child = parent - removed + added` in one call, going back to the parent is free.
- Int16 networks with one output layer, or Stockfish-style Int8 networks with hidden layers, pairwise and squared activations,
  dense and skip connections and output buckets.
- A float model, an Adam trainer and a quantizer for small networks; import of networks trained with bullet.
- Loading from files, streams, memory, asynchronously or memory-mapped. Trimming- and AOT-compatible.

```csharp
using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;

using var model = NnueModel.Load("net.nnue");
var evaluator = new NnueEvaluator(model);                    // best SIMD backend of this CPU
using var accumulators = model.CreateAccumulators(256);      // e.g. one per slot and search ply

evaluator.RefreshAccumulator(accumulators[0], featuresOfSlot0);
evaluator.RefreshAccumulator(accumulators[1], featuresOfSlot1);
evaluator.UpdateAccumulator(accumulators[0], accumulators[2], removedFeatures, addedFeatures);
int score = evaluator.Evaluate(accumulators[2], accumulators[1], bucket);
```

Declare networks with `NetworkSpec.CreateSimple`, `NetworkSpec.CreateInt8`, `NetworkSpec.CreateStockfishLike` or the fluent
`NetworkSpec.CreateBuilder`; train them with `Nnue.Training` (`FloatModel`, `NnueTrainer`, `NnueQuantizer`). Every public member is
documented in the IDE; the repository contains a user guide, an architecture description and a runnable demo.
