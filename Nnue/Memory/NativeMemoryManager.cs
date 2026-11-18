using System.Buffers;

namespace Nnue.Memory;

/// <summary>Exposes a native memory range as <see cref="Memory{T}"/> so that streams can read into it asynchronously.</summary>
internal sealed unsafe class NativeMemoryManager : MemoryManager<byte>
{
    private readonly byte* _pointer;
    private readonly int _length;

    public NativeMemoryManager(byte* pointer, int length) { _pointer = pointer; _length = length; }

    public override Span<byte> GetSpan() => new(_pointer, _length);
    public override MemoryHandle Pin(int elementIndex = 0) => new(_pointer + elementIndex);
    public override void Unpin() { }
    protected override void Dispose(bool disposing) { }
}
