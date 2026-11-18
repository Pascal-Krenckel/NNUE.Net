using System.Numerics;
using System.Runtime.InteropServices;

namespace Nnue.Training.Optimization;
/// <summary>The loss the trainer minimises.</summary>
public enum LossFunction
{
    /// <summary>mean((sigmoid(output) - target)^2); target in [0, 1]. The usual choice for game evaluation (win probability).</summary>
    SigmoidMse = 0,
    /// <summary>mean((output - target)^2); plain regression.</summary>
    Mse = 1,
}
