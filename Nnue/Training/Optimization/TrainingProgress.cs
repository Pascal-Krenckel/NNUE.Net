namespace Nnue.Training.Optimization;

/// <summary>Reported by <see cref="NnueTrainer.Train"/> after every epoch.</summary>
/// <param name="Epoch">The epoch that just finished, starting at 1.</param>
/// <param name="EpochCount">The number of epochs requested.</param>
/// <param name="TrainingLoss">Mean training loss seen during the epoch.</param>
/// <param name="LearningRate">The learning rate the epoch was trained with (before the decay for the next epoch).</param>
public readonly record struct TrainingProgress(int Epoch, int EpochCount, double TrainingLoss, float LearningRate);
