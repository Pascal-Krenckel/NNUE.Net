using Nnue.Configuration;

using Nnue.Memory;

namespace Nnue.Evaluation;

/// <summary>
/// Handle to one accumulator: <see cref="Length"/> int16 values in 64-byte aligned native memory owned by an <see cref="AccumulatorBuffer"/>.
/// The handle is a small struct (copy it freely); the memory stays valid until the buffer is disposed (or, without Dispose, until the buffer and all handles are unreachable).
/// A new accumulator contains zeros; call <see cref="NnueEvaluator.ResetAccumulator"/> or <see cref="NnueEvaluator.RefreshAccumulator"/> before using it.
/// <c>default(Accumulator)</c> is an empty handle (<see cref="IsEmpty"/>) that the evaluator rejects.
/// Two handles are equal if they refer to the same memory.
/// </summary>
public readonly unsafe struct Accumulator : IEquatable<Accumulator>
{
    internal readonly NativeBlock? Owner;  // keeps the native block (and its finalizer) alive while the handle is reachable; tells checked calls whether it was disposed
    internal readonly short* Pointer;

    internal Accumulator(NativeBlock owner, short* pointer, int length) { Owner = owner; Pointer = pointer; Length = length; }

    /// <summary>Number of int16 values (the <see cref="NetworkSpec.AccumulatorSize"/> of the model).</summary>
    public int Length { get; }

    /// <summary>True for <c>default(Accumulator)</c>, which refers to no memory.</summary>
    public bool IsEmpty => Pointer == null;

    /// <summary>The raw values (for inspection, tests, custom serialisation). Empty for <c>default(Accumulator)</c>.</summary>
    public Span<short> AsSpan() => Pointer == null ? default : new Span<short>(Pointer, Length);

    /// <summary>True if both handles refer to the same memory.</summary>
    public bool Equals(Accumulator other) => Pointer == other.Pointer && Length == other.Length;
    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Accumulator other && Equals(other);
    /// <inheritdoc/>
    public override int GetHashCode() => ((nint)Pointer).GetHashCode();
    /// <inheritdoc/>
    public override string ToString() => Pointer == null ? "Accumulator (empty)" : $"Accumulator[{Length}]";

    /// <summary>True if both handles refer to the same memory.</summary>
    public static bool operator ==(Accumulator left, Accumulator right) => left.Equals(right);
    /// <summary>True if the handles refer to different memory.</summary>
    public static bool operator !=(Accumulator left, Accumulator right) => !left.Equals(right);
}
