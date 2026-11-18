namespace Nnue.Training.Quantization;
/// <summary>Diagnostics produced while quantizing.</summary>
public readonly record struct QuantizationReport(
    float MaxAbsFeatureWeight,
    float MaxAbsFeatureBias,
    float MaxAbsDenseWeight,
    int ClippedValues,
    int MaxActiveFeatures,
    int WorstCaseAccumulator,
    long WorstCaseOutput)
{
    /// <summary>True if no combination of <see cref="MaxActiveFeatures"/> active features can overflow an int16 accumulator.</summary>
    public bool AccumulatorSafe => WorstCaseAccumulator <= short.MaxValue;
    /// <summary>True if no dense-layer sum can overflow int32 (checked against the largest possible activations).</summary>
    public bool OutputSafe => WorstCaseOutput <= int.MaxValue;

    /// <inheritdoc/>
    public override string ToString() =>
        $"max|ftW|={MaxAbsFeatureWeight:F3} max|ftB|={MaxAbsFeatureBias:F3} max|denseW|={MaxAbsDenseWeight:F3} clipped={ClippedValues} " +
        $"worst-case accumulator (<= {MaxActiveFeatures} active features)={WorstCaseAccumulator} " +
        (AccumulatorSafe ? "[safe]" : "[OVERFLOW POSSIBLE]") +
        $" worst-case dense sum={WorstCaseOutput} " + (OutputSafe ? "[safe]" : "[OVERFLOW POSSIBLE]");
}
