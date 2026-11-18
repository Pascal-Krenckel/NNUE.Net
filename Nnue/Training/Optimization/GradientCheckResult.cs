using System.Numerics;
using System.Runtime.InteropServices;

namespace Nnue.Training.Optimization;
/// <summary>Result of <see cref="NnueTrainer.CheckGradients"/>: analytic against numeric gradients, per parameter group.</summary>
public sealed class GradientCheckResult
{
    /// <summary>One entry per parameter group.</summary>
    public IReadOnlyList<GradientCheckGroup> Groups { get; }
    /// <summary>Creates a result from its groups.</summary>
    public GradientCheckResult(IReadOnlyList<GradientCheckGroup> groups) { Groups = groups; }
    /// <summary>Number of compared parameters.</summary>
    public int Probes => Groups.Sum(g => g.Probes);
    /// <summary>Number of parameters whose gradients disagreed beyond the tolerance.</summary>
    public int Failures => Groups.Sum(g => g.Failures);
    /// <summary>Probes skipped because the parameter sat on a kink of an activation, where the numeric gradient is meaningless.</summary>
    public int SkippedKinks => Groups.Sum(g => g.SkippedKinks);
    /// <summary>True if at least one probe ran and none failed.</summary>
    public bool Passed => Failures == 0 && Probes > 0;
    /// <inheritdoc/>
    public override string ToString() =>
        $"{Probes} probes, {Failures} failures, {SkippedKinks} skipped at activation kinks [" +
        string.Join(", ", Groups.Select(g => $"{g.Name}:{g.Probes - g.Failures}/{g.Probes} (max rel {g.MaxRelativeError:E1})")) + "]";
}
