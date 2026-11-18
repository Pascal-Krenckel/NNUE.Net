using Nnue.Backends;

namespace Nnue.Evaluation;
/// <summary>The core entry points of one backend as function pointers, bound once when an evaluator is created (calling through them costs nothing measurable).</summary>
internal unsafe struct FnTable
{
    public delegate*<EvaluationContext, short*, ReadOnlySpan<int>, void> RefreshAccumulator;
    public delegate*<EvaluationContext, short*, short*, ReadOnlySpan<int>, ReadOnlySpan<int>, void> UpdateAccumulator;
    public delegate*<EvaluationContext, short*, int, void> AddFeature;
    public delegate*<EvaluationContext, short*, int, void> RemoveFeature;
    public delegate*<EvaluationContext, short**, int, int> Evaluate;

    private static FnTable Make<TB>() where TB : struct, IBackend => new()
    {
        RefreshAccumulator = &Core<TB>.RefreshAccumulator,
        UpdateAccumulator = &Core<TB>.UpdateAccumulator,
        AddFeature = &Core<TB>.AddFeature,
        RemoveFeature = &Core<TB>.RemoveFeature,
        Evaluate = &Core<TB>.Evaluate,
    };

    /// <summary>The only place that lists all backends.</summary>
    public static FnTable For(SimdBackend backend) => backend switch
    {
        SimdBackend.Scalar => Make<ScalarBackend>(),
        SimdBackend.Sse => Make<SseBackend>(),
        SimdBackend.Avx2 => Make<Avx2Backend>(),
        SimdBackend.Avx512 => Make<Avx512Backend>(),
        SimdBackend.AvxVnni => Make<AvxVnniBackend>(),
        SimdBackend.AvxVnniEmulated => Make<AvxVnniEmulatedBackend>(),
        SimdBackend.Avx512Vnni => Make<Avx512VnniBackend>(),
        SimdBackend.Avx512VnniEmulated => Make<Avx512VnniEmulatedBackend>(),
        SimdBackend.Neon => Make<NeonBackend>(),
        SimdBackend.NeonDotProd => Make<NeonDotProdBackend>(),
        _ => throw new ArgumentOutOfRangeException(nameof(backend)),
    };
}
