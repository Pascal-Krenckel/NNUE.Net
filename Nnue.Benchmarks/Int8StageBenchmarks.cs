using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Nnue.Backends;
using Nnue.Kernels;

namespace Nnue.Benchmarks;

/// <summary>
/// Where does the time of an Int8 evaluation go? Stockfish-like network, H = 256, two slots: pairwise 2 x 128 -> layer 0 (256 -> 32) -> layer 1 (64 -> 32)
/// -> last layer (128 -> 1). Every stage on its own with the AVX-512 kernels (the benchmark needs AVX-512).
/// </summary>
[Config(typeof(NoTieringConfig))]
public unsafe class Int8StageBenchmarks
{
    private const int CallsPerInvoke = 1024;

    private readonly List<IntPtr> _memory = [];
    private short* _accumulatorA, _accumulatorB;
    private byte* _transformerOutput, _sparseTransformerOutput, _layer1Input, _layer2Input, _destination;
    private sbyte* _weights0, _weights1, _weights2;
    private int* _biases, _layerOutput, _chunkIndices;

    private T* Allocate<T>(long count) where T : unmanaged
    {
        long bytes = (count * sizeof(T)) + 64;
        byte* pointer = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        new Span<byte>(pointer, (int)bytes).Clear();
        _memory.Add((IntPtr)pointer);
        return (T*)pointer;
    }

    [GlobalSetup]
    public void Setup()
    {
        if (!SimdSupport.IsSupported(SimdBackend.Avx512))
            throw new NotSupportedException("This benchmark needs a CPU with AVX-512.");
        Random random = new(4);
        _accumulatorA = Allocate<short>(256);
        _accumulatorB = Allocate<short>(256);
        for (int i = 0; i < 256; i++)
        { _accumulatorA[i] = (short)random.Next(-300, 700); _accumulatorB[i] = (short)random.Next(-300, 700); }
        _transformerOutput = Allocate<byte>(256);
        _sparseTransformerOutput = Allocate<byte>(256);
        for (int i = 0; i < 256; i++)
        { _transformerOutput[i] = (byte)random.Next(0, 127); _sparseTransformerOutput[i] = random.Next(10) == 0 ? (byte)random.Next(1, 127) : (byte)0; }
        _weights0 = Allocate<sbyte>(256 * 32);
        _weights1 = Allocate<sbyte>(64 * 32);
        _weights2 = Allocate<sbyte>(128 * 16);
        for (int i = 0; i < 256 * 32; i++)
            _weights0[i] = (sbyte)random.Next(-127, 128);
        for (int i = 0; i < 64 * 32; i++)
            _weights1[i] = (sbyte)random.Next(-127, 128);
        for (int i = 0; i < 128 * 16; i++)
            _weights2[i] = (sbyte)random.Next(-127, 128);
        _biases = Allocate<int>(32);
        _layerOutput = Allocate<int>(32);
        _chunkIndices = Allocate<int>(96);
        for (int i = 0; i < 32; i++)
            _layerOutput[i] = random.Next(-9000, 9000);
        _layer1Input = Allocate<byte>(64);
        _layer2Input = Allocate<byte>(128);
        for (int i = 0; i < 128; i++)
        { _layer2Input[i] = (byte)random.Next(0, 127); if (i < 64) _layer1Input[i] = _layer2Input[i]; }
        _destination = Allocate<byte>(256);
    }

    [GlobalCleanup]
    public void Cleanup() { foreach (nint pointer in _memory) NativeMemory.AlignedFree((void*)pointer); _memory.Clear(); }

    [Benchmark(Baseline = true, OperationsPerInvoke = CallsPerInvoke)]
    public int Pairwise2x128()
    {
        for (int i = 0; i < CallsPerInvoke; i++)
        { Avx512Kernels.ApplyPairwise(_accumulatorA, _destination, 128); Avx512Kernels.ApplyPairwise(_accumulatorB, _destination + 128, 128); }
        return _destination[3];
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int Layer0Dense256To32()
    {
        for (int i = 0; i < CallsPerInvoke; i++)
            Avx512Kernels.AffineTransform(_transformerOutput, _weights0, _biases, _layerOutput, 256, 32);
        return _layerOutput[1];
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int Layer0Sparse10PercentNonZeroIncludingFind()
    {
        for (int i = 0; i < CallsPerInvoke; i++)
        {
            int count = Avx512Kernels.FindNonZeroChunks(_sparseTransformerOutput, _chunkIndices, 256);
            Avx512Kernels.AffineTransformSparse(_sparseTransformerOutput, _weights0, _biases, _chunkIndices, count, _layerOutput, 32);
        }
        return _layerOutput[1];
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int CountNonZeroChunks256Bytes()
    {
        int sum = 0;
        for (int i = 0; i < CallsPerInvoke; i++)
            sum += Avx512Kernels.CountNonZeroChunks((int*)_sparseTransformerOutput, 64);
        return sum;
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int Layer1Dense64To32()
    {
        for (int i = 0; i < CallsPerInvoke; i++)
            Avx512Kernels.AffineTransform(_layer1Input, _weights1, _biases, _layerOutput, 64, 32);
        return _layerOutput[1];
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int Layer2Dense128To16()
    {
        for (int i = 0; i < CallsPerInvoke; i++)
            Avx512Kernels.AffineTransform(_layer2Input, _weights2, _biases, _layerOutput, 128, 16);
        return _layerOutput[1];
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int Layer2LastLayerDotProduct128To1()
    {
        int sum = 0;
        for (int i = 0; i < CallsPerInvoke; i++)
            sum += Avx512Kernels.AffineTransformSingleOutput(_layer2Input, _weights2, 5, 128);
        return sum;
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int ActivateClippedReLU32()
    {
        for (int i = 0; i < CallsPerInvoke; i++)
            Portable128Kernels.ActivateClippedReLU(_layerOutput, _destination, 32);
        return _destination[3];
    }

    [Benchmark(OperationsPerInvoke = CallsPerInvoke)]
    public int ActivateSqrClippedReLU32()
    {
        for (int i = 0; i < CallsPerInvoke; i++)
            SseKernels.ActivateSqrClippedReLU(_layerOutput, _destination, 32);
        return _destination[3];
    }
}
