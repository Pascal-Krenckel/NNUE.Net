using System.Runtime.InteropServices;

namespace Nnue.Memory;

/// <summary>
/// One zero-initialised, 64-byte aligned block of native memory (<see cref="NativeMemory.AlignedAlloc"/>). It is freed by <see cref="Dispose"/>,
/// or by the finalizer if nobody disposed it. Handles that point into the block keep a reference to it, so the finalizer cannot run while
/// a handle is still reachable; after an explicit Dispose every pointer into the block is invalid.
/// </summary>
internal sealed unsafe class NativeBlock : IDisposable
{
    public const int Alignment = 64;

    private nint _ptr;
    private readonly Action? _release;      // external memory (e.g. a memory-mapped file): called instead of AlignedFree

    public NativeBlock(long bytes)
    {
        if (bytes < 0)
            throw new ArgumentOutOfRangeException(nameof(bytes));
        Size = (bytes + Alignment - 1) & ~(Alignment - 1);
        if (Size == 0)
            Size = Alignment;
        byte* p = (byte*)NativeMemory.AlignedAlloc((nuint)Size, Alignment);
        if (p == null)
            throw new OutOfMemoryException();
        NativeMemory.Clear(p, (nuint)Size);
        _ptr = (nint)p;
        GC.AddMemoryPressure(Size);
    }

    private NativeBlock(byte* pointer, long bytes, Action release) { _ptr = (nint)pointer; Size = bytes; _release = release; }

    /// <summary>
    /// Wraps memory owned by somebody else (a mapped file view). <paramref name="release"/> runs once on Dispose or finalization.
    /// The pointer must be 64-byte aligned.
    /// </summary>
    public static NativeBlock FromExternal(byte* pointer, long bytes, Action release) => ((nint)pointer & (Alignment - 1)) != 0
            ? throw new ArgumentException("External memory must be 64-byte aligned.", nameof(pointer))
            : new NativeBlock(pointer, bytes, release);

    /// <summary>Start of the block (64-byte aligned); null after <see cref="Dispose"/>.</summary>
    public byte* Pointer => (byte*)_ptr;
    public long Size { get; }
    public bool IsDisposed => _ptr == 0;

    public void Dispose()
    {
        Free(disposing: true);
        GC.SuppressFinalize(this);
    }

    ~NativeBlock() => Free(disposing: false);

    private void Free(bool disposing)
    {
        nint p = Interlocked.Exchange(ref _ptr, 0);
        if (p == 0)
            return;
        if (_release is not null)
        {
            if (disposing)
            { _release(); return; }
            try
            { _release(); }
            catch { }       // an exception in a finalizer would terminate the process
            return;
        }
        NativeMemory.AlignedFree((void*)p);
        GC.RemoveMemoryPressure(Size);
    }
}
