using System.Buffers.Binary;
using Nnue.Configuration;
using Nnue.Models;

namespace Nnue.Interop;

/// <summary>
/// Reads and writes the quantised weight file of networks trained with the bullet trainer, so they can be evaluated by this library.
/// <para>
/// The file has no header: the arrays are stored back to back in little endian, in the order of bullet's <c>SavedFormat</c> list, padded with
/// zeros to a multiple of 64 bytes. This importer expects the order used by bullet's examples:
/// <c>l0w</c> (feature weights, int16, <c>[FeatureCount][AccumulatorSize]</c>), <c>l0b</c> (feature biases, int16),
/// <c>l1w</c> (output weights, int16, for every output bucket the weights of slot 0 followed by slot 1 ...), <c>l1b</c> (output biases, one per bucket).
/// </para>
/// <para>
/// The network must be declared with a matching <see cref="NetworkSpec"/> (Int16 precision; <see cref="NetworkSpec.ActivationQuantization"/> and
/// <see cref="NetworkSpec.OutputWeightQuantization"/> equal to bullet's <c>QA</c> and <c>QB</c>, <see cref="NetworkSpec.OutputScale"/> to its scale),
/// because the file does not describe itself; a file whose size does not fit the spec is rejected.
/// For <see cref="FeatureTransformerActivation.SquaredClippedReLU"/> bullet engines divide the dot product by QA before adding the bias; this library keeps the
/// full product, so biases are multiplied by QA on import (and divided by it on export).
/// </para>
/// Note: verified against files written by this library's own writer and the layout documented above, not against a real bullet training run.
/// </summary>
public static class BulletFormat
{
    private const int Padding = 64;

    /// <summary>Reads a bullet file from <paramref name="stream"/> (the stream is read to its end).</summary>
    /// <exception cref="ArgumentException">The spec is not an Int16 network.</exception>
    /// <exception cref="InvalidDataException">The size of the data does not match the spec.</exception>
    public static NnueModel Load(Stream stream, NetworkSpec spec, BulletFormatOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using MemoryStream buffer = new();
        stream.CopyTo(buffer);
        return Load(buffer.GetBuffer().AsSpan(0, (int)buffer.Length), spec, options);
    }

    /// <summary>Reads a bullet file.</summary>
    public static NnueModel Load(string path, NetworkSpec spec, BulletFormatOptions? options = null)
    {
        using FileStream file = File.OpenRead(path);
        return Load(file, spec, options);
    }

    /// <summary>Reads a bullet file from memory.</summary>
    public static NnueModel Load(ReadOnlySpan<byte> data, NetworkSpec spec, BulletFormatOptions? options = null)
    {
        options ??= new BulletFormatOptions();
        CheckSpec(spec);
        CheckOptions(options);
        NnueModelBuilder builder = new(spec);
        try
        {
            long expected = ExpectedLength(spec, options);
            long padded = (expected + Padding - 1) / Padding * Padding;
            if (data.Length != expected && data.Length != padded)
                throw new InvalidDataException($"The file has {data.Length} bytes, but a network of this spec needs {expected} (or {padded} with bullet's padding to 64 bytes). Check the spec and the format options.");

            int position = 0;
            ReadInt16(data, ref position, builder.FeatureWeights);
            ReadInt16(data, ref position, builder.FeatureBiases);

            int inputs = spec.Slots * spec.AccumulatorSize, buckets = spec.OutputBuckets;
            short[] raw = new short[inputs * buckets];
            ReadInt16(data, ref position, raw);
            for (int b = 0; b < buckets; b++)
            {
                Span<short> destination = builder.GetOutputWeights(b);
                for (int i = 0; i < inputs; i++)
                    destination[i] = options.OutputWeightsLayout == BulletOutputWeightsLayout.BucketMajor ? raw[(b * inputs) + i] : raw[(i * buckets) + b];
            }

            int biasScale = spec.Activation == FeatureTransformerActivation.SquaredClippedReLU ? spec.ActivationQuantization : 1;
            Span<int> biases = builder.OutputBiases;
            for (int b = 0; b < buckets; b++)
            {
                int bias;
                if (options.OutputBiasesAreInt32)
                { bias = BinaryPrimitives.ReadInt32LittleEndian(data[position..]); position += sizeof(int); }
                else
                { bias = BinaryPrimitives.ReadInt16LittleEndian(data[position..]); position += sizeof(short); }
                biases[b] = checked(bias * biasScale);
            }
            return builder.Build();
        }
        catch { builder.Abandon(); throw; }
    }

    /// <summary>Writes the model in bullet's format (padded to 64 bytes), e.g. to test an engine against the same network.</summary>
    /// <exception cref="ArgumentException">The model is not an Int16 model, or a bias does not fit the saved type.</exception>
    public static void Save(NnueModel model, Stream stream, BulletFormatOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(stream);
        options ??= new BulletFormatOptions();
        NetworkSpec spec = model.Spec;
        CheckSpec(spec);
        CheckOptions(options);
        using MemoryStream output = new();
        byte[] scratch = new byte[4];

        void WriteInt16(ReadOnlySpan<short> values)
        { foreach (short v in values) { BinaryPrimitives.WriteInt16LittleEndian(scratch, v); output.Write(scratch, 0, 2); } }

        WriteInt16(model.FeatureWeights);
        WriteInt16(model.FeatureBiases);
        int inputs = spec.Slots * spec.AccumulatorSize, buckets = spec.OutputBuckets;
        short[] raw = new short[inputs * buckets];
        for (int b = 0; b < buckets; b++)
        {
            ReadOnlySpan<short> source = model.GetOutputWeights(b);
            for (int i = 0; i < inputs; i++)
                raw[options.OutputWeightsLayout == BulletOutputWeightsLayout.BucketMajor ? (b * inputs) + i : (i * buckets) + b] = source[i];
        }
        WriteInt16(raw);

        int biasScale = spec.Activation == FeatureTransformerActivation.SquaredClippedReLU ? spec.ActivationQuantization : 1;
        for (int b = 0; b < buckets; b++)
        {
            int bias = (int)Math.Round(model.GetOutputBias(b) / (double)biasScale);
            if (options.OutputBiasesAreInt32)
            { BinaryPrimitives.WriteInt32LittleEndian(scratch, bias); output.Write(scratch, 0, 4); }
            else
            {
                if (bias is < short.MinValue or > short.MaxValue)
                    throw new ArgumentException($"Output bias {bias} of bucket {b} does not fit int16; use BulletFormatOptions.OutputBiasesAreInt32.");
                BinaryPrimitives.WriteInt16LittleEndian(scratch, (short)bias);
                output.Write(scratch, 0, 2);
            }
        }
        while (output.Length % Padding != 0)
            output.WriteByte(0);
        output.Position = 0;
        output.CopyTo(stream);
    }

    /// <summary>Writes the model to a file in bullet's format.</summary>
    public static void Save(NnueModel model, string path, BulletFormatOptions? options = null)
    {
        using FileStream file = File.Create(path);
        Save(model, file, options);
    }

    private static void CheckSpec(NetworkSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        spec.Validate();
        if (spec.Precision != NetworkPrecision.Int16)
            throw new ArgumentException("The bullet format is supported for Int16 networks (one output layer) only.", nameof(spec));
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("Big endian platforms are not supported.");
    }

    private static void CheckOptions(BulletFormatOptions options)
    {
        if (!Enum.IsDefined(options.OutputWeightsLayout))
            throw new ArgumentOutOfRangeException(nameof(options), $"Unknown output weights layout {(int)options.OutputWeightsLayout}.");
    }

    private static long ExpectedLength(NetworkSpec spec, BulletFormatOptions options)
    {
        long shorts = ((long)spec.FeatureCount * spec.AccumulatorSize) + spec.AccumulatorSize + ((long)spec.Slots * spec.AccumulatorSize * spec.OutputBuckets);
        return (shorts * sizeof(short)) + ((long)spec.OutputBuckets * (options.OutputBiasesAreInt32 ? sizeof(int) : sizeof(short)));
    }

    private static void ReadInt16(ReadOnlySpan<byte> data, ref int position, Span<short> destination)
    {
        for (int i = 0; i < destination.Length; i++)
            destination[i] = BinaryPrimitives.ReadInt16LittleEndian(data[(position + (i * 2))..]);
        position += destination.Length * sizeof(short);
    }
}
