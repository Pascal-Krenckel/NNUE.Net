namespace Nnue.Training.Optimization;
/// <summary>
/// One training sample. The library does not know what the features mean: <see cref="Features"/> has one feature list per slot of the
/// model (in slot order), <see cref="Bucket"/> selects the dense stack and <see cref="Target"/> is what the output should approach.
/// </summary>
public sealed class TrainingSample
{
    /// <summary>One list of active feature indices per slot (Spec.Slots lists).</summary>
    public required int[][] Features { get; init; }
    /// <summary>Output bucket (dense stack) this sample is evaluated with; must be &lt; Spec.OutputBuckets.</summary>
    public int Bucket { get; init; }
    /// <summary>With <see cref="LossFunction.SigmoidMse"/> a value in [0, 1] (see <see cref="TargetFunctions"/>), with <see cref="LossFunction.Mse"/> the desired network output
    /// (score / OutputScale).</summary>
    public float Target { get; init; }
}
