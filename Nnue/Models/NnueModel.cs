using System.IO.MemoryMappedFiles;
using Microsoft.Win32.SafeHandles;
using Nnue.Configuration;
using Nnue.Evaluation;
using Nnue.Memory;

namespace Nnue.Models;
/// <summary>
/// An immutable network: all weights live in ONE 64-byte aligned native block (<see cref="System.Runtime.InteropServices.NativeMemory.AlignedAlloc"/>; freed by Dispose or, at the latest, by the finalizer), laid out exactly as in the
/// saved file. Thread safe; evaluate it with a <see cref="NnueEvaluator"/>. Create it with <see cref="NnueModelBuilder"/>,
/// <see cref="Load(Stream)"/>, <see cref="LoadMemoryMapped(string)"/> or the quantizer of Nnue.Training.Quantization.
/// </summary>
public sealed unsafe class NnueModel : IDisposable
{
    private const uint Magic = 0x4D554E4E;      // "NNUM"
    private const int Version = 3;      // 2: the last dense layer is stored as a contiguous vector; 3: the blob starts at a 64-byte aligned file offset (padding after the header)

    /// <summary>The architecture of the network.</summary>
    public NetworkSpec Spec { get; }
    internal NetworkLayout Layout { get; }

    private readonly NativeBlock? _stepBlock;
    internal readonly byte* BasePointer;
    internal readonly short* FeatureWeightsPointer, FeatureBiasPointer, OutputWeightsPointer;
    internal readonly int* OutputBiasPointer;
    internal readonly Step* Steps;

    internal NnueModel(NetworkSpec spec, NetworkLayout layout, NativeBlock blob)
    {
        Spec = spec;
        Layout = layout;
        Memory = blob;
        byte* aligned = blob.Pointer;
        BasePointer = aligned;
        FeatureWeightsPointer = (short*)(aligned + layout.FeatureWeightsOffset);
        FeatureBiasPointer = (short*)(aligned + layout.FeatureBiasOffset);
        if (spec.Precision == NetworkPrecision.Int16)
        {
            OutputWeightsPointer = (short*)(aligned + layout.OutputWeightsOffset);
            OutputBiasPointer = (int*)(aligned + layout.OutputBiasOffset);
        }
        else
        {
            int layers = layout.DenseLayers.Count;
            _stepBlock = new NativeBlock((long)spec.OutputBuckets * layers * sizeof(Step));
            Steps = (Step*)_stepBlock.Pointer;
            for (int b = 0; b < spec.OutputBuckets; b++)
                for (int k = 0; k < layers; k++)
                {
                    DenseLayerInfo d = layout.DenseLayers[k];
                    Step* s = Steps + (b * layers) + k;
                    s->Weights = (sbyte*)(aligned + layout.GetDenseWeightsOffset(b, k));
                    s->Biases = (int*)(aligned + layout.GetDenseBiasOffset(b, k));
                    s->InputLength = d.InputLength;
                    s->PaddedOutputLength = d.PaddedOutputLength;
                    s->OutputLength = d.OutputLength;
                    s->InputOffset = d.InputOffset;
                    s->ActivationOffset = d.ActivationOffset;
                }
        }
    }

    /// <summary>Frees the weights now. The model, its evaluators' pointers and all views are invalid afterwards (without Dispose the finalizer frees them).</summary>
    public void Dispose() { Memory.Dispose(); _stepBlock?.Dispose(); }

    /// <summary>True after <see cref="Dispose"/>.</summary>
    public bool IsDisposed => Memory.IsDisposed;

    /// <summary>The weight memory (the evaluator checks it for disposal).</summary>
    internal NativeBlock Memory { get; }
    internal void ThrowIfDisposed() { if (Memory.IsDisposed) throw new ObjectDisposedException(nameof(NnueModel)); }

    // ---------------------------------------------------------------- creating accumulators
    /// <summary>Allocates <paramref name="count"/> zeroed accumulators sized for this model in one aligned block (e.g. a search stack).</summary>
    public AccumulatorBuffer CreateAccumulators(int count) { ThrowIfDisposed(); return new(this, count); }
    /// <summary>Allocates one accumulator (its own buffer; prefer <see cref="CreateAccumulators"/> for many).</summary>
    public Accumulator CreateAccumulator() => new AccumulatorBuffer(this, 1)[0];

    // ---------------------------------------------------------------- read-only views (inspection, export, tests)
    /// <remarks>The span points into native memory: keep the model reachable (and undisposed) while you use it, e.g. with <c>GC.KeepAlive(model)</c>.</remarks>
    public ReadOnlySpan<short> FeatureWeights { get { ThrowIfDisposed(); return new(FeatureWeightsPointer, checked(Layout.FeatureCount * Layout.AccumulatorSize)); } }
    /// <summary>The feature transformer biases, one per accumulator value.</summary>
    public ReadOnlySpan<short> FeatureBiases { get { ThrowIfDisposed(); return new(FeatureBiasPointer, Layout.AccumulatorSize); } }

    /// <summary>Int16 models: the output layer weights of one bucket, slot-major (slot 0 values first).</summary>
    public ReadOnlySpan<short> GetOutputWeights(int bucket) => Spec.Precision != NetworkPrecision.Int16
            ? throw new InvalidOperationException("Int16 models only.")
            : new(OutputWeightsPointer + (bucket * (Layout.OutputWeightsBucketStride / sizeof(short))), Layout.FeatureTransformerOutputSize);
    /// <summary>Int16 models: the output bias of one bucket.</summary>
    public int GetOutputBias(int bucket) => Spec.Precision != NetworkPrecision.Int16
            ? throw new InvalidOperationException("Int16 models only.")
            : OutputBiasPointer[bucket];

    /// <summary>
    /// Int8 models: the weights of one dense layer in storage order (blocked for SIMD, the last layer a contiguous vector), e.g. to hash or
    /// compare models. Use <see cref="GetDenseWeight"/> to read the weight of one input / output pair.
    /// </summary>
    public ReadOnlySpan<sbyte> GetDenseWeights(int bucket, int layer)
    {
        if (Spec.Precision != NetworkPrecision.Int8)
            throw new InvalidOperationException("Int8 models only.");

        _ = Layout.DenseLayers[layer];
        return new((sbyte*)(BasePointer + Layout.GetDenseWeightsOffset(bucket, layer)), Layout.GetDenseWeightCount(layer));
    }
    /// <summary>Int8 models: the weight from input <paramref name="input"/> to output <paramref name="output"/> of dense layer <paramref name="layer"/> (scale 64 = 1.0).</summary>
    public sbyte GetDenseWeight(int bucket, int layer, int input, int output)
    {
        CheckDenseIndex(Spec, Layout, bucket, layer, input, output);
        return GetDenseWeights(bucket, layer)[Layout.GetDenseWeightIndex(layer, input, output)];
    }

    internal static void CheckDenseIndex(NetworkSpec spec, NetworkLayout layout, int bucket, int layer, int input, int output)
    {
        if (spec.Precision != NetworkPrecision.Int8)
            throw new InvalidOperationException("Int8 models only.");
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)bucket, (uint)spec.OutputBuckets, nameof(bucket));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)layer, (uint)layout.DenseLayers.Count, nameof(layer));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)input, (uint)layout.DenseLayers[layer].InputLength, nameof(input));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)output, (uint)layout.DenseLayers[layer].OutputLength, nameof(output));
    }

    /// <summary>Int8 models: the biases of one dense layer (scale 127 * 64 = 1.0), padded to a multiple of 16 outputs.</summary>
    public ReadOnlySpan<int> GetDenseBiases(int bucket, int layer) => Spec.Precision != NetworkPrecision.Int8
            ? throw new InvalidOperationException("Int8 models only.")
            : new((int*)(BasePointer + Layout.GetDenseBiasOffset(bucket, layer)), Layout.DenseLayers[layer].PaddedOutputLength);

    // ---------------------------------------------------------------- storage
    // File: magic, version, spec, blob size, [version 3: zero padding up to a multiple of 64 bytes], blob (exactly the in-memory layout, little endian).
    // The padding puts the blob at a 64-byte aligned file offset, so a memory-mapped file can be used as the model memory without copying.
    internal const int HeaderProbeLength = 512;      // larger than any header (at most 64 hidden layers)

    private static byte[] CreateHeader(NetworkSpec spec, long blobSize)
    {
        using MemoryStream ms = new();
        using (BinaryWriter w = new(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            w.Write(Magic);
            w.Write(Version);
            spec.Write(w);
            w.Write(blobSize);
        }
        while (ms.Length % NativeBlock.Alignment != 0)
            ms.WriteByte(0);
        return ms.ToArray();
    }

    /// <summary>Parses the header at the start of <paramref name="data"/>; returns the offset of the blob.</summary>
    internal static int ParseHeader(ReadOnlySpan<byte> data, out NetworkSpec spec, out long blobSize)
    {
        try
        {
            using MemoryStream ms = new(data[..Math.Min(data.Length, HeaderProbeLength)].ToArray());     // the header only, not the weights
            using BinaryReader r = new(ms);
            if (r.ReadUInt32() != Magic)
                throw new InvalidDataException("Not an Nnue model file.");
            int version = r.ReadInt32();
            if (version is not (2 or Version))
                throw new InvalidDataException($"Unsupported model version {version}.");
            spec = NetworkSpec.Read(r);
            blobSize = r.ReadInt64();
            int offset = (int)ms.Position;
            if (version == Version)
                offset = (offset + NativeBlock.Alignment - 1) & ~(NativeBlock.Alignment - 1);
            return offset;
        }
        catch (EndOfStreamException) { throw new InvalidDataException("Unexpected end of file (truncated model)."); }
        catch (ArgumentException e) { throw new InvalidDataException("Invalid network specification in file: " + e.Message, e); }
    }

    internal static NnueModelBuilder CreateBuilder(NetworkSpec spec, long blobSize)
    {
        NnueModelBuilder builder = new(spec);
        if (blobSize != builder.Layout.BlobSize)
        { builder.Abandon(); throw new InvalidDataException("Blob size does not match the spec."); }
        return builder;
    }

    /// <summary>Writes the model (little endian); the blob starts at a 64-byte aligned offset.</summary>
    public void Save(Stream stream)
    {
        ThrowIfDisposed();
        stream.Write(CreateHeader(Spec, Layout.BlobSize));
        const int Chunk = 1 << 20;
        for (long pos = 0; pos < Layout.BlobSize; pos += Chunk)
            stream.Write(new ReadOnlySpan<byte>(BasePointer + pos, (int)Math.Min(Chunk, Layout.BlobSize - pos)));
        GC.KeepAlive(Memory);
    }

    /// <summary>Writes the model to a file (see <see cref="Save(Stream)"/>).</summary>
    public void Save(string path) { using FileStream fs = File.Create(path); Save(fs); }

    /// <summary>
    /// Reads a model from a stream; the weights are read straight into their final aligned memory. A seekable stream is left positioned right
    /// after the model; a non-seekable one may have been read up to 512 bytes further if the model is smaller than that.
    /// </summary>
    /// <exception cref="InvalidDataException">The data is not a model, is truncated, or has an unsupported version.</exception>
    public static NnueModel Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        byte[] probe = new byte[HeaderProbeLength];
        int probed = stream.ReadAtLeast(probe, probe.Length, throwOnEndOfStream: false);
        int offset = ParseHeader(probe.AsSpan(0, probed), out NetworkSpec? spec, out long blobSize);
        NnueModelBuilder builder = CreateBuilder(spec, blobSize);
        try
        {
            Span<byte> blob = builder.RawBlob;
            int have = Math.Min(Math.Max(probed - offset, 0), blob.Length);
            probe.AsSpan(offset, have).CopyTo(blob);
            stream.ReadExactly(blob[have..]);
            int overRead = probed - offset - have;
            if (overRead > 0 && stream.CanSeek)
                _ = stream.Seek(-overRead, SeekOrigin.Current);     // give back what the probe read past the model
            return builder.Build();
        }
        catch (EndOfStreamException) { builder.Abandon(); throw new InvalidDataException("Unexpected end of file (truncated model)."); }
        catch { builder.Abandon(); throw; }
    }

    /// <summary>Reads a model from a file.</summary>
    public static NnueModel Load(string path) { using FileStream fs = File.OpenRead(path); return Load(fs); }

    /// <summary>Reads a model from memory (e.g. an embedded resource or a downloaded buffer).</summary>
    public static NnueModel Load(ReadOnlySpan<byte> data)
    {
        int offset = ParseHeader(data, out NetworkSpec? spec, out long blobSize);
        if (data.Length - offset < blobSize)
            throw new InvalidDataException("Unexpected end of data (truncated model).");
        NnueModelBuilder builder = CreateBuilder(spec, blobSize);
        data.Slice(offset, (int)blobSize).CopyTo(builder.RawBlob);
        return builder.Build();
    }

    /// <summary>Asynchronous <see cref="Load(Stream)"/>: the weights are read into their final memory without blocking the calling thread.</summary>
    public static Task<NnueModel> LoadAsync(Stream stream, CancellationToken cancellationToken = default)
        => NnueModelLoader.LoadAsync(stream, cancellationToken);

    /// <summary>Asynchronous <see cref="Load(string)"/>.</summary>
    public static Task<NnueModel> LoadAsync(string path, CancellationToken cancellationToken = default)
        => NnueModelLoader.LoadAsync(path, cancellationToken);

    /// <summary>
    /// Maps the file into memory and uses the mapping as the weights: nothing is copied or read up front, the operating system pages the
    /// weights in on demand and shares them between processes. Needs a file written by this version (the blob must start at a 64-byte aligned
    /// offset). The mapping is released by <see cref="Dispose"/> (or the finalizer); do not modify or delete the file while the model is in use.
    /// </summary>
    /// <exception cref="InvalidDataException">The file is not a model or was written by an older version (load it with <see cref="Load(string)"/> and save it again).</exception>
    public static NnueModel LoadMemoryMapped(string path)
    {
        MemoryMappedFile file = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(path, FileMode.Open, mapName: null, capacity: 0, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
        System.IO.MemoryMappedFiles.MemoryMappedViewAccessor? view = null;
        byte* pointer = null;
        try
        {
            view = file.CreateViewAccessor(0, 0, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
            SafeMemoryMappedViewHandle handle = view.SafeMemoryMappedViewHandle;
            handle.AcquirePointer(ref pointer);
            byte* start = pointer + view.PointerOffset;
            long length = new FileInfo(path).Length;        // not the view's ByteLength: Windows rounds the view up to whole pages

            using UnmanagedMemoryStream probe = new(start, Math.Min(length, HeaderProbeLength));
            byte[] header = new byte[probe.Length];
            probe.ReadExactly(header);
            if (header.Length >= 8 && BitConverter.ToInt32(header, 4) != Version)
                throw new InvalidDataException("Memory mapping needs a version 3 model file; load the file normally and save it again to upgrade it.");
            int offset = ParseHeader(header, out NetworkSpec? spec, out long blobSize);
            if (length - offset < blobSize)
                throw new InvalidDataException("Unexpected end of file (truncated model).");
            NetworkLayout layout = NetworkLayout.Create(spec);
            if (blobSize != layout.BlobSize)
                throw new InvalidDataException("Blob size does not match the spec.");

            MemoryMappedViewAccessor openView = view;
            MemoryMappedFile openFile = file;
            NativeBlock block = NativeBlock.FromExternal(start + offset, blobSize, () => { openView.SafeMemoryMappedViewHandle.ReleasePointer(); openView.Dispose(); openFile.Dispose(); });
            view = null;                                    // owned by the block now
            return new NnueModel(spec, layout, block);
        }
        catch
        {
            if (view is not null)
            { if (pointer != null) view.SafeMemoryMappedViewHandle.ReleasePointer(); view.Dispose(); }
            file.Dispose();
            throw;
        }
    }
}
