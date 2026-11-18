using Nnue.Evaluation;

namespace Nnue.Backends;
/// <summary>Instruction set used by <see cref="NnueEvaluator"/>.</summary>
public enum SimdBackend
{
    /// <summary>Plain C#, works everywhere. Reference implementation.</summary>
    Scalar = 0,
    /// <summary>x86 128-bit vectors. Requires SSSE3 (pmaddubsw is the int8 workhorse); SSE2-only CPUs fall back to Scalar.</summary>
    Sse = 1,
    /// <summary>x86 256-bit vectors.</summary>
    Avx2 = 2,
    /// <summary>x86 512-bit vectors (AVX-512 F + BW).</summary>
    Avx512 = 3,
    /// <summary>AArch64 NEON (AdvSimd). The int8 layers use smull/smull2 + addp + sadalp.</summary>
    Neon = 4,
    /// <summary>AArch64 NEON with the dot-product extension (SDOT, ARMv8.2-A DotProd): one instruction per 16 int8 products.</summary>
    NeonDotProd = 5,
    /// <summary>
    /// x86 with AVX-VNNI (VEX encoded, 256 bit; Alder Lake / Raptor Lake / Sapphire Rapids and newer): the AVX2 kernels everywhere, and
    /// <c>vpdpbusd</c> for the int8 dense layers (256-bit VEX form; the 512-bit form is <see cref="Avx512Vnni"/>).
    /// </summary>
    AvxVnni = 6,
    /// <summary>
    /// TEST AID, never selected automatically and not listed in <see cref="SimdSupport.Available"/>: the same kernels as
    /// <see cref="AvxVnni"/> with a software model of vpdpbusd. Runs on any AVX2 CPU, so the VNNI kernel structure can be verified without VNNI hardware.
    /// </summary>
    AvxVnniEmulated = 7,
    /// <summary>
    /// x86 with AVX512-VNNI (Ice Lake, Sapphire Rapids, Zen 4 and newer): the AVX-512 kernels everywhere, and <c>vpdpbusd zmm</c> for the
    /// int8 dense layers. Needs the net11.0 build of the library (the instruction is exposed as <c>AvxVnni.V512</c> from .NET 11 on).
    /// </summary>
    Avx512Vnni = 8,
    /// <summary>
    /// TEST AID, never selected automatically and not listed in <see cref="SimdSupport.Available"/>: the <see cref="Avx512Vnni"/> kernels with a
    /// software model of vpdpbusd. Runs on any AVX-512 CPU and runtime, so the kernel structure can be verified without .NET 11 or VNNI hardware.
    /// </summary>
    Avx512VnniEmulated = 9,
}
