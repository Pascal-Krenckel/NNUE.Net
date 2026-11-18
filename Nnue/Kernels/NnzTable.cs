using System.Runtime.InteropServices;

namespace Nnue.Kernels;

/// <summary>
/// Lookup table for the sparse input kernels: for every 8-bit mask the positions of its set bits in ascending order (8 x uint32, 32 bytes per
/// mask, unused slots 0). A mask of non-zero chunks becomes an index list with one vector load, one add and one unaligned store, no branch per
/// non-zero chunk (Stockfish does the same). Allocated once, 64-byte aligned, never freed (8 KB).
/// </summary>
internal static unsafe class NnzTable
{
    public static readonly uint* Lut = Build();

    private static uint* Build()
    {
        uint* p = (uint*)NativeMemory.AlignedAlloc(256 * 8 * sizeof(uint), 64);
        for (int m = 0; m < 256; m++)
        {
            int k = 0;
            for (int b = 0; b < 8; b++)
                if (((m >> b) & 1) != 0)
                    p[(m * 8) + k++] = (uint)b;
            for (; k < 8; k++)
                p[(m * 8) + k] = 0;
        }
        return p;
    }
}
