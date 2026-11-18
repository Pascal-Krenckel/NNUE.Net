# User guide

This guide explains the concepts of the library and walks through the typical tasks. All types live in the `Nnue.*` namespaces of
the `Nnue.Net` package; the folder of every source file is its namespace.

| Namespace | Types |
|---|---|
| `Nnue.Configuration` | `NetworkSpec`, `NetworkSpecBuilder`, `NetworkPrecision`, `FeatureTransformerActivation`, `HiddenActivation` |
| `Nnue.Models` | `NnueModel`, `NnueModelBuilder` |
| `Nnue.Evaluation` | `NnueEvaluator`, `Accumulator`, `AccumulatorBuffer`, `SparseInputMode` |
| `Nnue.Backends` | `SimdBackend`, `SimdSupport` |
| `Nnue.Interop` | `BulletFormat`, `BulletFormatOptions`, `BulletOutputWeightsLayout` |
| `Nnue.Training.Models` | `FloatModel`, `FloatDense`, `Activations` |
| `Nnue.Training.Optimization` | `NnueTrainer`, `TrainerOptions`, `TrainingSample`, `TrainingProgress`, `LossFunction`, `TargetFunctions`, gradient check results |
| `Nnue.Training.Quantization` | `NnueQuantizer`, `QuantizationReport` |

A program that only evaluates needs `Nnue.Configuration`, `Nnue.Models` and `Nnue.Evaluation`.

## Concepts

**Features.** The input of a network is a set of *active binary features*, identified by indices `0 .. FeatureCount-1`. What a
feature means is up to you: "white knight on f3", "card 7 in hand", "cell (3,4) occupied". A typical position activates a few
dozen out of thousands of possible features.

**Accumulators and slots.** The first layer (the *feature transformer*) is a sparse matrix: every feature owns one row of
`AccumulatorSize` int16 weights. An *accumulator* is the bias plus the rows of all active features. Because a move only changes a few
features, the accumulator of a child position is computed from its parent by subtracting and adding a few rows; that is what makes
NNUE fast.

One evaluation consumes `Slots` accumulators. The library does not know what they mean, only their order: slot 0 is the first
accumulator you pass, slot 1 the second. Chess engines use two slots, "side to move" and "other side", each computed from that
player's point of view with the same weights. A single-perspective game uses one slot, a four-player game could use four. If your
score is relative to the player to move, you swap or negate yourself when the side to move changes.

**Output buckets.** The layers after the feature transformer can exist several times (`OutputBuckets`, Stockfish's "layer
stacks"); every evaluation picks one, for example by the number of pieces on the board.

**Checked and unchecked calls.** Every evaluator operation exists twice. `Evaluate`, `UpdateAccumulator`, ... validate their
arguments and throw. `EvaluateUnchecked`, `UpdateAccumulatorUnchecked`, ... skip the checks and are meant for tight loops with
trusted input; invalid arguments there read or write out of bounds. The checks cost 0-3 % of an Int8 evaluation and up to a
quarter of a small incremental update (where every feature index is validated).

## Declaring a network

A `NetworkSpec` describes the architecture. Use a preset, the fluent builder, or set the properties directly; all three validate.

```csharp
// one int16 output layer: simple and accurate, the classic "768 -> 2x256 -> 1" network
var simple = NetworkSpec.CreateSimple(featureCount: 768, accumulatorSize: 256, slots: 2, outputBuckets: 8);

// Stockfish arithmetic with two hidden layers
var int8 = NetworkSpec.CreateInt8(featureCount: 768, accumulatorSize: 256, hiddenLayerSizes: [16, 32], outputBuckets: 8);

// the topology of current Stockfish (pairwise activation, [sqr|clip] hidden activations, dense and skip connections)
var stockfishLike = NetworkSpec.CreateStockfishLike(featureCount: 22528, accumulatorSize: 1024, outputBuckets: 8);

// anything else
var custom = NetworkSpec.CreateBuilder(768)
    .WithAccumulatorSize(512).WithSlots(2).WithOutputBuckets(8)
    .WithInt8(FeatureTransformerActivation.Pairwise, [16, 32])
    .WithSqrClippedHiddenActivation().WithSkipConnection()
    .Build();                                                    // throws ArgumentException naming the conflict

if (!candidate.TryValidate(out string? error)) Console.WriteLine(error);
```

| Option | Values |
|---|---|
| `Precision` | `Int16`: one int16 output layer · `Int8`: Stockfish arithmetic, any number of hidden layers |
| `Activation` | `ClippedReLU` `clamp(a,0,1)` · `SquaredClippedReLU` `clamp(a,0,1)²` · `Pairwise` `clamp(a)·clamp(b)` of the two halves of an accumulator (Int8 only) |
| `HiddenLayerSizes` | sizes of the hidden layers (Int8, multiples of 16, at most 4096 each) |
| `HiddenActivation` | `ClippedReLU` or Stockfish's `SqrClippedReLU`, which concatenates `[sqr(z) \| clip(z)]` |
| `DenseConnections` | later layers see the activations of all earlier hidden layers |
| `SkipConnection` | `output += z0[n-2] - z0[n-1]` of the first hidden layer |
| `OutputBuckets` | number of independent layer stacks (1-256) |
| `Slots`, `AccumulatorSize` | accumulators per evaluation (1-64) and their size (a multiple of 32; of 64 for pairwise) |
| `ActivationQuantization`, `OutputWeightQuantization`, `OutputScale` | integer scales; the defaults fit networks trained by this library |

`NetworkSpec` is an immutable record with value equality: two specs are equal if they describe the same network.

## Training and quantization

The trainer optimizes a float copy of the network with Adam and is meant for small networks (it keeps a dense gradient buffer
per thread). A `TrainingSample` holds one feature list per slot, the output bucket and the target.

```csharp
var floatModel = new FloatModel(spec);
floatModel.Randomize(seed: 1, expectedActiveFeatures: 30);

var options = new TrainerOptions { BatchSize = 4096, LearningRate = 0.01f, LearningRateDecay = 0.8f, Loss = LossFunction.SigmoidMse };
var trainer = new NnueTrainer(floatModel, options);

var samples = positions.Select(p => new TrainingSample
{
    Features = [p.FeaturesOfSideToMove, p.FeaturesOfOtherSide],   // one list per slot
    Bucket = p.PieceCount / 4,
    Target = TargetFunctions.ScoreToTarget(p.Score, spec.OutputScale),   // sigmoid(score / scale), in [0, 1]
}).ToArray();

trainer.Train(samples, epochs: 20, new Progress<TrainingProgress>(p => Console.WriteLine($"{p.Epoch}/{p.EpochCount}: {p.TrainingLoss:F5}")), cancellationToken);
double validationLoss = trainer.ComputeLoss(validationSamples);

using NnueModel model = NnueQuantizer.Quantize(floatModel, out QuantizationReport report);
Console.WriteLine(report);              // clipped values, worst-case accumulator and dense sums, overflow verdicts
model.Save("net.nnue");
```

- Samples are validated before any work starts; a bad sample is reported with its index.
- `Train` reports progress after every epoch and checks the cancellation token before every batch. A `Progress<T>` posts the
  report to the captured synchronization context; implement `IProgress<T>` yourself to receive it on the training thread.
- The trainer works on a copy of `TrainerOptions`; the learning-rate decay never changes your object.
- `TargetFunctions.Blend` mixes a score target with a game result.
- `FloatModel.Save`/`Load` write float checkpoints, `FloatModel.PredictScore` evaluates the float network (a slow reference).
- `NnueTrainer.CheckGradients` compares the analytic gradients with finite differences, useful when you change the architecture.
- The quantizer clamps weights to the representable range (the trainer already keeps them there). Check
  `report.AccumulatorSafe` and `report.OutputSafe`: they tell whether an int16 accumulator or an int32 sum could overflow for
  the given number of active features.

## Evaluating in a search

Create one `NnueEvaluator` per model and backend; it is stateless and thread safe. Accumulators live in an `AccumulatorBuffer`,
one aligned native allocation; give every search thread its own buffer.

```csharp
var evaluator = new NnueEvaluator(model);                     // SimdSupport.Best; or new NnueEvaluator(model, SimdBackend.Avx2)
using var stack = model.CreateAccumulators(2 * MaxPly);       // two slots per ply

// root: from scratch
evaluator.RefreshAccumulator(stack[0], white.ActiveFeatures);
evaluator.RefreshAccumulator(stack[1], black.ActiveFeatures);

// make move at ply p: write the child accumulators, the parent stays untouched (unmake is free)
evaluator.UpdateAccumulator(stack[2 * p], stack[2 * p + 2], removedWhite, addedWhite);
evaluator.UpdateAccumulator(stack[2 * p + 1], stack[2 * p + 3], removedBlack, addedBlack);

// evaluate from the point of view of the side to move: slot order = training order
int score = whiteToMove
    ? evaluator.Evaluate(stack[2 * p + 2], stack[2 * p + 3], bucket)
    : evaluator.Evaluate(stack[2 * p + 3], stack[2 * p + 2], bucket);
```

- `UpdateAccumulator(source, destination, removed, added)` computes `destination = source - removed + added` in one pass;
  `source` and `destination` may be the same accumulator. `AddFeature`/`RemoveFeature` change one feature in place,
  `ResetAccumulator` sets the accumulator to the bias.
- `AccumulatorBuffer.Copy(sourceIndex, destinationIndex, count)` copies consecutive accumulators with one memory copy.
- `Evaluate` has overloads for one slot, two slots and a span of slots. `EvaluateFeatures` evaluates feature lists directly
  (allocates; for tools and tests).
- For Int8 networks `SparseInputMode.Auto` (the default) lets the first dense layer skip all-zero input chunks when that pays off;
  the result is identical in every mode. `GetNonZeroInputFraction` tells how sparse your activations are.

## Loading and saving

```csharp
model.Save("net.nnue");                                         // or Save(stream)

using var a = NnueModel.Load("net.nnue");                       // file
using var b = NnueModel.Load(stream);                           // stream (a seekable stream is left right after the model)
using var c = NnueModel.Load(bytes);                            // memory, e.g. an embedded resource
using var d = await NnueModel.LoadAsync("net.nnue", cancellationToken);
using var e = NnueModel.LoadMemoryMapped("net.nnue");           // no copy: the OS pages the weights in and shares them between processes
```

The weights are read straight into their final, 64-byte aligned memory. Memory mapping needs a file written by this library
version (format 3); older files still load normally. Do not change or delete a file while a model maps it.

## Networks trained with bullet

[bullet](https://github.com/jw1912/bullet) saves quantised networks as raw arrays without a header, so you declare the matching spec:

```csharp
var spec = NetworkSpec.CreateBuilder(768).WithAccumulatorSize(512).WithSlots(2).WithOutputBuckets(8)
    .WithInt16(FeatureTransformerActivation.SquaredClippedReLU).WithQuantization(255, 64).WithOutputScale(400).Build();
using var model = BulletFormat.Load("quantised.bin", spec,
    new BulletFormatOptions { OutputWeightsLayout = BulletOutputWeightsLayout.BucketMajor, OutputBiasesAreInt32 = false });
model.Save("net.nnue");                                          // or BulletFormat.Save(model, path) to write the bullet format
```

The importer expects the order of bullet's examples (`l0w`, `l0b`, `l1w`, `l1b`) and supports Int16 networks only. A file whose
size does not match the spec is rejected. The reader is tested against hand-written files, not against a real bullet run.

## Resources, threads and disposal

- `NnueModel` and `AccumulatorBuffer` own native memory and implement `IDisposable`. Without `Dispose` a finalizer frees the
  memory eventually; after `Dispose` the model, its evaluators and all accumulators of the buffer are invalid. Checked calls throw
  `ObjectDisposedException`, unchecked calls would touch freed memory.
- Spans returned by `NnueModel.FeatureWeights` and similar point into native memory: keep the model alive while you use them.
- Models are immutable and thread safe; evaluators are stateless and thread safe; accumulators and buffers are not synchronized.
- `NnueModelBuilder` fills the weights of a new model in place (no copy on `Build`). Int8 dense weights are set per input/output
  pair with `SetDenseWeight`; the raw spans use an internal blocked order.

## Limits

- Hidden layers have at most 4096 outputs, at most 64 hidden layers and 16384 hidden activations in total (an Int8 evaluation keeps
  its scratch memory on the stack); `Slots * AccumulatorSize` (halved for pairwise) is at most 16384.
- A model is at most 2 GB.
- Weights filled by hand through `NnueModelBuilder` are not checked for overflow; the quantizer report covers trained networks.
