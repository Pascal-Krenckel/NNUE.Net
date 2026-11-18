namespace Nnue.Training.Optimization;
/// <summary>Helpers that turn raw labels into training targets. They live outside the core because they encode game assumptions.</summary>
public static class TargetFunctions
{
    /// <summary>sigmoid(score / outputScale): the target for <see cref="LossFunction.SigmoidMse"/> from a score in score units.</summary>
    public static float ScoreToTarget(float score, float outputScale) => 1f / (1f + MathF.Exp(-score / outputScale));

    /// <summary>Interpolation of a score target and a result in [0, 1] (win probability): lambda * sigmoid(score/outputScale) + (1 - lambda) * result.</summary>
    public static float Blend(float score, float outputScale, float result, float scoreWeight)
        => scoreWeight * ScoreToTarget(score, outputScale) + (1f - scoreWeight) * result;
}
