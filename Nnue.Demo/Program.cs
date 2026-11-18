using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Models;
using Nnue.Training.Models;
using Nnue.Training.Optimization;
using Nnue.Training.Quantization;

// A complete example: train, quantize, save, load and use a network.
//
// The library knows nothing about games. Everything chess-like in this file (what a feature is, that there are two perspectives,
// how a position is turned into feature lists, which bucket to use, how a score becomes a target) is the "abstraction on top"
// that a user of the library writes: see the class ToyPosition at the bottom.


// ------------------------------------------------------------------ 1. declare the network
// 768 features (2 colours x 6 piece types x 64 squares), two slots (side to move, other side), 4 output buckets by piece count.
NetworkSpec spec = new()
{
    FeatureCount = ToyPosition.FeatureCount,
    AccumulatorSize = 128,
    Slots = 2,
    Precision = NetworkPrecision.Int8,
    Activation = FeatureTransformerActivation.ClippedReLU,
    HiddenLayerSizes = [16, 32],
    OutputBuckets = 4,
};

// ------------------------------------------------------------------ 2. make training data (here: synthetic positions)
Random rng = new(1);
TrainingSample[] train = Enumerable.Range(0, 60_000).Select(_ => ToyPosition.Random(rng).ToSample(spec.OutputScale)).ToArray();
TrainingSample[] test = Enumerable.Range(0, 5_000).Select(_ => ToyPosition.Random(rng).ToSample(spec.OutputScale)).ToArray();

// ------------------------------------------------------------------ 3. train in float
FloatModel floatModel = new(spec);
floatModel.Randomize(seed: 7, expectedActiveFeatures: 16);
NnueTrainer trainer = new(floatModel, new TrainerOptions { BatchSize = 4096, LearningRate = 0.01f, LearningRateDecay = 0.8f });
Console.WriteLine($"loss before training: {trainer.ComputeLoss(test):F5}");
trainer.Train(train, epochs: 12, new SynchronousProgress<TrainingProgress>(p =>
    Console.WriteLine($"  epoch {p.Epoch,2}/{p.EpochCount}  train loss {p.TrainingLoss:F5}  test loss {trainer.ComputeLoss(test):F5}")));

// ------------------------------------------------------------------ 4. quantize and save
NnueModel model = NnueQuantizer.Quantize(floatModel, out QuantizationReport report);
Console.WriteLine($"quantized: {report}");
string path = Path.Combine(Path.GetTempPath(), "toy.nnue");
model.Save(path);
Console.WriteLine($"saved {new FileInfo(path).Length / 1024} KiB to {path}");

// ------------------------------------------------------------------ 5. load and evaluate
NnueModel loaded = NnueModel.Load(path);
NnueEvaluator evaluator = new(loaded);                     // best SIMD backend of this CPU
Console.WriteLine($"evaluator backend: {evaluator.Backend}");

// an accumulator buffer, e.g. one entry per search ply and slot
AccumulatorBuffer acc = loaded.CreateAccumulators(count: 8);
double error = 0;
foreach (TrainingSample? s in test.Take(1000))
{
    evaluator.RefreshAccumulator(acc[0], s.Features[0]);                  // slot 0: from scratch
    evaluator.RefreshAccumulator(acc[1], s.Features[1]);                  // slot 1
    int score = evaluator.Evaluate(acc[0], acc[1], s.Bucket);  // checked; use EvaluateUnchecked in a tight loop with trusted input
    error += Math.Abs(score - floatModel.PredictScore(s.Features, s.Bucket));
}
Console.WriteLine($"mean |integer - float| over 1000 positions: {error / 1000:F1} score units");

// incremental update: move one piece
{
    TrainingSample s = test[0];
    evaluator.RefreshAccumulator(acc[0], s.Features[0]);
    evaluator.RefreshAccumulator(acc[1], s.Features[1]);
    int removed = s.Features[0][0], added = (removed + 1) % ToyPosition.FeatureCount;
    evaluator.UpdateAccumulator(acc[0], acc[2], [removed], [added]);      // destination = source - removed + added, source stays untouched
    int before = evaluator.Evaluate(acc[0], acc[1], s.Bucket);
    int after = evaluator.Evaluate(acc[2], acc[1], s.Bucket);
    Console.WriteLine($"score before {before}, after moving one feature {after}; back to the parent is free: {evaluator.Evaluate(acc[0], acc[1], s.Bucket)}");
}

/// <summary>
/// The abstraction ON TOP of the library: a toy "position" with two sides. It decides what a feature index means, builds one feature list
/// per slot (own pieces as seen by the side to move / the same board as seen by the other side), picks the output bucket and converts a score
/// into a target. None of this is part of the library.
/// </summary>
internal sealed class ToyPosition
{
    public const int FeatureCount = 768;
    private static readonly int[] Value = { 100, 300, 300, 500, 900, 0 };

    public int[] Own = [], Their = [];     // (square, piece type) lists
    public int Score;

    public static ToyPosition Random(Random rng)
    {
        int own = rng.Next(3, 9), their = rng.Next(3, 9), n = own + their, phase = 16 - n;
        int[] squares = Enumerable.Range(0, 64).OrderBy(_ => rng.Next()).Take(n).ToArray();
        ToyPosition p = new()
        { Own = new int[own], Their = new int[their] };
        int score = 0;
        for (int i = 0; i < n; i++)
        {
            int type = rng.Next(6), sq = squares[i];
            bool isOwn = i < own;
            (isOwn ? p.Own : p.Their)[isOwn ? i : i - own] = (type * 64) + sq;
            score += isOwn ? Value[type] : -Value[type];
            if (type == 0)
                score += (isOwn ? 1 : -1) * (sq >> 3) * (5 + (3 * phase));
        }
        p.Score = score;
        return p;
    }

    /// <summary>feature index = (side relative to the viewer * 6 + type) * 64 + square; the second slot sees the board mirrored with the sides swapped</summary>
    public int[][] FeatureLists() => [
        Own.Select(x => x).Concat(Their.Select(x => (6 * 64) + x)).ToArray(),
        Their.Select(x => (x / 64 * 64) + ((x % 64) ^ 56)).Concat(Own.Select(x => (6 * 64) + (x / 64 * 64) + ((x % 64) ^ 56))).ToArray(),
    ];

    public int Bucket => Math.Min(3, (Own.Length + Their.Length - 6) / 3);

    public TrainingSample ToSample(float outputScale) => new() { Features = FeatureLists(), Bucket = Bucket, Target = TargetFunctions.ScoreToTarget(Score, outputScale) };
}

/// <summary>Reports on the training thread (unlike <see cref="Progress{T}"/>), so the callback can use the trainer between epochs.</summary>
internal sealed class SynchronousProgress<T>(Action<T> report) : IProgress<T>
{
    public void Report(T value) => report(value);
}
