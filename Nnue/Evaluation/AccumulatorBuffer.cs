using System.Collections;
using System.Runtime.CompilerServices;
using Nnue.Memory;
using Nnue.Models;

namespace Nnue.Evaluation;

/// <summary>
/// A fixed number of accumulators in one aligned allocation, e.g. a whole search stack (ply * slots accumulators).
/// Consecutive accumulators can be copied with a single memcpy (2.7 - 4.8 times faster than copying separate arrays).
/// Enumerating the buffer (or indexing it) yields <see cref="Accumulator"/> handles. Dispose it to free the native memory early.
/// </summary>
public sealed unsafe class AccumulatorBuffer : IReadOnlyList<Accumulator>, IDisposable
{
    private readonly NativeBlock _block;
    private readonly short* _base;

    /// <summary>Allocates <paramref name="count"/> zeroed accumulators sized for <paramref name="model"/>.</summary>
    public AccumulatorBuffer(NnueModel model, int count)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        AccumulatorSize = model.Layout.AccumulatorSize;
        Count = count;
        _block = new NativeBlock((long)count * AccumulatorSize * sizeof(short));
        _base = (short*)_block.Pointer;
    }

    /// <summary>Number of accumulators.</summary>
    public int Count { get; }

    /// <summary>Number of int16 values per accumulator.</summary>
    public int AccumulatorSize { get; }

    /// <summary>True after <see cref="Dispose"/>; every handle taken from the buffer is invalid then.</summary>
    public bool IsDisposed => _block.IsDisposed;

    /// <summary>The accumulator at <paramref name="index"/>; a handle (no copy of the values).</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside [0, <see cref="Count"/>).</exception>
    /// <exception cref="ObjectDisposedException">The buffer was disposed.</exception>
    public Accumulator this[int index]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)index, (uint)Count, nameof(index));
            ThrowIfDisposed();
            return new Accumulator(_block, _base + ((long)index * AccumulatorSize), AccumulatorSize);
        }
    }

    /// <summary>
    /// Copies <paramref name="count"/> consecutive accumulators starting at <paramref name="sourceIndex"/> to <paramref name="destinationIndex"/>
    /// (the ranges may overlap).
    /// </summary>
    public void Copy(int sourceIndex, int destinationIndex, int count = 1)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegative(sourceIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(destinationIndex);
        if (sourceIndex + (long)count > Count)
            throw new ArgumentOutOfRangeException(nameof(sourceIndex), "The source range exceeds the buffer.");
        if (destinationIndex + (long)count > Count)
            throw new ArgumentOutOfRangeException(nameof(destinationIndex), "The destination range exceeds the buffer.");
        if (sourceIndex == destinationIndex || count == 0)
            return;

        long bytes = (long)count * AccumulatorSize * sizeof(short);
        short* source = _base + ((long)sourceIndex * AccumulatorSize), destination = _base + ((long)destinationIndex * AccumulatorSize);
        bool overlap = sourceIndex < destinationIndex + count && destinationIndex < sourceIndex + count;
        if (overlap)
            Buffer.MemoryCopy(source, destination, bytes, bytes);
        else
            Unsafe.CopyBlock(destination, source, (uint)bytes);
    }

    /// <summary>Enumerates the accumulators in index order.</summary>
    public IEnumerator<Accumulator> GetEnumerator()
    {
        for (int i = 0; i < Count; i++)
            yield return this[i];
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Frees the native memory now. Every <see cref="Accumulator"/> handle of this buffer is invalid afterwards (without Dispose the finalizer frees it).</summary>
    public void Dispose() => _block.Dispose();

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_block.IsDisposed, this);
}
