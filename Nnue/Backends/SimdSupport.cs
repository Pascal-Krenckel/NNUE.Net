using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Nnue.Backends;
/// <summary>Which SIMD backends this CPU supports, and which one an evaluator uses by default.</summary>
public static class SimdSupport
{
    /// <summary>True if <paramref name="backend"/> can run on this CPU.</summary>
    public static bool IsSupported(SimdBackend backend) => backend switch
    {
        SimdBackend.Scalar => true,
        SimdBackend.Sse => Ssse3.IsSupported,
        SimdBackend.Avx2 => Avx2.IsSupported,
        SimdBackend.Avx512 => Avx512F.IsSupported && Avx512BW.IsSupported,
        SimdBackend.Neon => AdvSimd.Arm64.IsSupported,
        SimdBackend.NeonDotProd => AdvSimd.Arm64.IsSupported && Dp.IsSupported,
        SimdBackend.AvxVnni => Avx2.IsSupported && AvxVnniHardware,
        SimdBackend.AvxVnniEmulated => Avx2.IsSupported,
        SimdBackend.Avx512Vnni => Avx512F.IsSupported && Avx512BW.IsSupported && Avx512VnniHardware,
        SimdBackend.Avx512VnniEmulated => Avx512F.IsSupported && Avx512BW.IsSupported,
        _ => false,
    };

    /// <summary>AVX512-VNNI (<c>vpdpbusd zmm</c>) is exposed as <c>AvxVnni.V512</c> from .NET 11 on; older builds of the library cannot use it.</summary>
    private static bool Avx512VnniHardware =>
#if NET11_0_OR_GREATER
        AvxVnni.V512.IsSupported;
#else
        false;
#endif

    /// <summary>
    /// AVX-VNNI is a preview API in .NET 8 (CA2252) and stable from .NET 9 (the net10.0 build of this library uses it without preview). The analyzer error is suppressed here and in
    /// <c>VnniHw</c> only, so the assembly itself is not marked as preview. Define NNUE_NO_AVXVNNI to compile it out completely.
    /// </summary>
    private static bool AvxVnniHardware =>
#if NNUE_NO_AVXVNNI
            return false;
#else
#pragma warning disable CA2252
            AvxVnni.IsSupported;
#pragma warning restore CA2252
#endif


    /// <summary>
    /// Fastest backend available on the current CPU: <see cref="SimdBackend.Avx512Vnni"/> where available (net11.0 build), else
    /// <see cref="SimdBackend.Avx512"/> (on CPUs that also have the 256-bit AVX-VNNI it measured about equal or faster), else
    /// <see cref="SimdBackend.AvxVnni"/> on AVX2 CPUs with AVX-VNNI, then AVX2, SSSE3, NEON with dot product, NEON, scalar.
    /// </summary>
    public static SimdBackend Best
    {
        get
        {
            if (IsSupported(SimdBackend.Avx512Vnni))
                return SimdBackend.Avx512Vnni;
            if (IsSupported(SimdBackend.Avx512))
                return SimdBackend.Avx512;
            if (IsSupported(SimdBackend.AvxVnni))
                return SimdBackend.AvxVnni;
            return IsSupported(SimdBackend.Avx2)
                ? SimdBackend.Avx2
                : IsSupported(SimdBackend.Sse)
                ? SimdBackend.Sse
                : IsSupported(SimdBackend.NeonDotProd)
                ? SimdBackend.NeonDotProd
                : IsSupported(SimdBackend.Neon) ? SimdBackend.Neon : SimdBackend.Scalar;
        }
    }

    /// <summary>All real backends the current CPU can run.</summary>
    public static IEnumerable<SimdBackend> Available =>
        Enum.GetValues<SimdBackend>().Where(b => b is not (SimdBackend.AvxVnniEmulated or SimdBackend.Avx512VnniEmulated) && IsSupported(b));

    /// <summary><see cref="Available"/> plus test-only backends (<see cref="SimdBackend.AvxVnniEmulated"/>, <see cref="SimdBackend.Avx512VnniEmulated"/>).</summary>
    public static IEnumerable<SimdBackend> AvailableForTesting =>
        Enum.GetValues<SimdBackend>().Where(IsSupported);
}
