using System.Runtime.CompilerServices;

namespace Nnue.Evaluation;
/// <summary>How the first dense layer of an Int8 network treats zero chunks in its input.</summary>
public enum SparseInputMode
{
    /// <summary>Always the dense kernel.</summary>
    Off = 0,
    /// <summary>Count the non-zero 4-byte chunks (cheap popcount pass); if at most 40 % are non-zero,
    /// build the index list and use the sparse kernel, otherwise the dense kernel. Bit-identical results in every mode.</summary>
    Auto = 1,
    /// <summary>Always the sparse kernel (mainly for testing).</summary>
    Always = 2,
}
