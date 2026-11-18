# Changelog

## 0.1.0 (unreleased)

First published version, package id `Nnue.Net`.

- Evaluation with SIMD backends for AVX512-VNNI (net11.0), AVX-512, AVX-VNNI, AVX2, SSSE3, NEON with and without dot product, and a
  scalar reference; all bit-identical.
- Int16 networks (one output layer) and Stockfish-style Int8 networks (hidden layers, pairwise and squared activations, dense and
  skip connections, output buckets), any number of accumulator slots.
- Incremental accumulator updates with register-tiled kernels; checked and unchecked API.
- `NetworkSpec` presets, fluent `NetworkSpecBuilder`, `TryValidate`, value equality.
- Model file format version 3 (blob at a 64-byte aligned offset); loading from files, streams, memory, asynchronously and
  memory-mapped; version 2 files still load.
- Import and export of bullet's quantised Int16 networks.
- Float model, Adam trainer with progress reporting and cancellation, gradient check, quantizer with overflow report.
- Targets net8.0, net10.0 and net11.0; trimming- and AOT-compatible; XML documentation; source link.

### Renamed during development

Users of pre-release builds: the public API was renamed to C# conventions.

| Before | Now |
|---|---|
| `FtActivation.CReLU / SCReLU` | `FeatureTransformerActivation.ClippedReLU / SquaredClippedReLU` |
| `HiddenActivation.CReLU / SqrCReLU`, `spec.Hidden` | `HiddenActivation.ClippedReLU / SqrClippedReLU`, `spec.HiddenActivation` |
| `spec.Layers`, `QA`, `QB` | `HiddenLayerSizes` (now an `ImmutableArray<int>`), `ActivationQuantization`, `OutputWeightQuantization` |
| `NetworkSpec.PaddedOut`, `StockfishLike(ftSize, l2, l3, stacks)` | `GetPaddedOutputLength`, `CreateStockfishLike(accumulatorSize, secondLayerSize, thirdLayerSize, outputBuckets)` |
| `model.Layout`, `NetworkLayout`, `DenseLayerInfo` | internal; use `NetworkSpec.DenseLayerCount`, `NnueModel.GetDenseWeight`, `NnueModelBuilder.SetDenseWeight` |
| `model.FtWeights / FtBias / OutWeights(b) / OutBias(b) / DenseWeights / DenseBias` | `FeatureWeights / FeatureBiases / GetOutputWeights(b) / GetOutputBias(b) / GetDenseWeights / GetDenseBiases` |
| `evaluator.Reset/Refresh/Update/Add/Remove` (+`Unchecked`) | `ResetAccumulator/RefreshAccumulator/UpdateAccumulator/AddFeature/RemoveFeature` (+`Unchecked`) |
| `Evaluate(featureLists, bucket)`, `NonZeroInputFraction`, `EvaluateRaw` | `EvaluateFeatures`, `GetNonZeroInputFraction`, internal |
| `Accumulator.Size / Values / IsValid` | `Length / AsSpan() / IsEmpty` |
| `AccumulatorBuffer.Copy(from, to, count)` | `Copy(sourceIndex, destinationIndex, count)` |
| `FloatModel.Dense/FtWeights/FtBias`, `FloatDense.In/Out/Bias`, `Forward`, `Score` | `DenseLayers/FeatureWeights/FeatureBiases`, `InputLength/OutputLength/Biases`, `Predict`, `PredictScore` |
| `Activations.Clip/Screlu/Sqr` (+`Derivative`) | `ClippedReLU/SquaredClippedReLU/SqrClippedReLU` (+`Derivative`) |
| `NnueQuantizer.MaxWeight/MaxOutWeightInt16`, `Targets.FromScore`, `TrainerOptions.FtWeightClip` | `GetMaxWeight/GetMaxInt16OutputWeight`, `TargetFunctions.ScoreToTarget`, `FeatureWeightClip` |
| `NnueTrainer.Train(data, epochs, Action<int, double>)` | `Train(data, epochs, IProgress<TrainingProgress>?, CancellationToken)` |
| `NnueEvaluator.SparseDensityLimit` | removed (fixed at 40 %, see `SparseInputMode.Auto`) |
