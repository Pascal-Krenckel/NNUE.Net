namespace Nnue.Training.Optimization;
/// <summary>Outcome of <see cref="NnueTrainer.CheckGradients"/> for one parameter group.</summary>
public readonly record struct GradientCheckGroup(string Name, int Probes, int Failures, double MaxRelativeError, int SkippedKinks = 0);
