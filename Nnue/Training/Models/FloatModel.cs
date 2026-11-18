using Nnue.Configuration;
using Nnue.Models;

namespace Nnue.Training.Models;
/// <summary>Full-precision network: what the trainer optimizes and the quantizer turns into an <see cref="NnueModel"/>.</summary>
public sealed class FloatModel
{
    private const uint Magic = 0x46554E4E;      // "NNUF"
    private const int Version = 2;

    /// <summary>The architecture of the network.</summary>
    public NetworkSpec Spec { get; }
    internal NetworkLayout Layout { get; }

    /// <summary>[FeatureCount * AccumulatorSize], row-major: row f holds the weights of feature f.</summary>
    public float[] FeatureWeights { get; }
    /// <summary>The feature transformer biases, one per accumulator value.</summary>
    public float[] FeatureBiases { get; }
    /// <summary>Dense layers per output bucket: DenseLayers[bucket][layer]; the last layer of each stack has one output.</summary>
    public FloatDense[][] DenseLayers { get; }

    /// <summary>Creates a zeroed network for <paramref name="spec"/> (call <see cref="Randomize"/> before training).</summary>
    public FloatModel(NetworkSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        Layout = NetworkLayout.Create(spec);
        Spec = spec;
        FeatureWeights = new float[spec.FeatureCount * spec.AccumulatorSize];
        FeatureBiases = new float[spec.AccumulatorSize];
        DenseLayers = Enumerable.Range(0, spec.OutputBuckets)
            .Select(_ => Layout.DenseLayers.Select(d => new FloatDense(d.InputLength, d.OutputLength)).ToArray()).ToArray();
    }

    /// <summary>Fills the weights with a seeded random initialization (biases zero).</summary>
    /// <param name="seed">Seed of the random generator.</param>
    /// <param name="expectedActiveFeatures">Typical number of active features per accumulator; scales the initial transformer weights
    /// so a fresh accumulator starts roughly in the linear range of the activation.</param>
    public void Randomize(int seed = 1, int expectedActiveFeatures = 32)
    {
        var rng = new Random(seed);
        float ftStd = 0.5f / MathF.Sqrt(Math.Max(1, expectedActiveFeatures));
        for (int i = 0; i < FeatureWeights.Length; i++) FeatureWeights[i] = Gaussian(rng) * ftStd;
        Array.Clear(FeatureBiases);
        foreach (var d in DenseLayers.SelectMany(stack => stack))
        {
            float std = 1f / MathF.Sqrt(d.InputLength);
            for (int i = 0; i < d.Weights.Length; i++) d.Weights[i] = Gaussian(rng) * std;
            Array.Clear(d.Biases);
        }
    }

    private static float Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>Slow scalar reference forward pass: one feature list per slot. Returns the raw network output (score / OutputScale).</summary>
    public float Predict(IReadOnlyList<int[]> featuresPerSlot, int bucket = 0)
    {
        if ((uint)bucket >= (uint)Spec.OutputBuckets) throw new ArgumentOutOfRangeException(nameof(bucket));
        ArgumentNullException.ThrowIfNull(featuresPerSlot);
        if (featuresPerSlot.Count != Spec.Slots) throw new ArgumentException($"The model has {Spec.Slots} slot(s), got {featuresPerSlot.Count} feature lists.");
        foreach (var features in featuresPerSlot)
            foreach (int f in features ?? throw new ArgumentException("A feature list is null.", nameof(featuresPerSlot)))
                if ((uint)f >= (uint)Spec.FeatureCount) throw new ArgumentOutOfRangeException(nameof(featuresPerSlot), $"Feature index {f} out of range [0, {Spec.FeatureCount}).");
        var stack = DenseLayers[bucket];
        int h = Spec.AccumulatorSize, slotOut = Layout.SlotOutputSize;

        // ---- feature transformer + activation, slot after slot ----
        var x0 = new float[Layout.FeatureTransformerOutputSize];
        for (int s = 0; s < Spec.Slots; s++)
        {
            var acc = (float[])FeatureBiases.Clone();
            foreach (int f in featuresPerSlot[s])
            {
                int off = f * h;
                for (int j = 0; j < h; j++) acc[j] += FeatureWeights[off + j];
            }
            switch (Spec.Activation)
            {
                case FeatureTransformerActivation.ClippedReLU: for (int j = 0; j < h; j++) x0[s * slotOut + j] = Activations.ClippedReLU(acc[j]); break;
                case FeatureTransformerActivation.SquaredClippedReLU: for (int j = 0; j < h; j++) x0[s * slotOut + j] = Activations.SquaredClippedReLU(acc[j]); break;
                default:
                    for (int j = 0; j < slotOut; j++) x0[s * slotOut + j] = Activations.ClippedReLU(acc[j]) * Activations.ClippedReLU(acc[j + slotOut]);
                    break;
            }
        }

        // ---- dense stack; hidden activations live in one concatenated buffer, exactly like the integer evaluator ----
        int last = stack.Length - 1;
        var buf = new float[Layout.TotalActivationSize];
        float skip = 0;
        for (int k = 0; k <= last; k++)
        {
            var d = stack[k];
            ReadOnlySpan<float> x = k == 0 ? x0 : buf.AsSpan(Layout.DenseLayers[k].InputOffset, Layout.DenseLayers[k].InputLength);
            var z = new float[d.OutputLength];
            for (int o = 0; o < d.OutputLength; o++)
            {
                float sum = d.Biases[o];
                for (int i = 0; i < d.InputLength; i++) sum += d.Weights[o * d.InputLength + i] * x[i];
                z[o] = sum;
            }
            if (k == last) return z[0] + skip;
            if (k == 0 && Spec.SkipConnection) skip = z[d.OutputLength - 2] - z[d.OutputLength - 1];

            int at = Layout.DenseLayers[k].ActivationOffset;
            if (Spec.HiddenActivation == HiddenActivation.SqrClippedReLU)
                for (int o = 0; o < d.OutputLength; o++) { buf[at + o] = Activations.SqrClippedReLU(z[o]); buf[at + d.OutputLength + o] = Activations.ClippedReLU(z[o]); }
            else
                for (int o = 0; o < d.OutputLength; o++) buf[at + o] = Activations.ClippedReLU(z[o]);
        }
        throw new InvalidOperationException();
    }

    /// <summary>Forward pass in score units (output * OutputScale).</summary>
    public float PredictScore(IReadOnlyList<int[]> featuresPerSlot, int bucket = 0) => Predict(featuresPerSlot, bucket) * Spec.OutputScale;

    // ---------------------------------------------------------------- checkpoints
    /// <summary>Writes a float checkpoint to a file.</summary>
    public void Save(string path) { using var fs = File.Create(path); Save(fs); }
    /// <summary>Writes a float checkpoint (spec and all weights).</summary>
    public void Save(Stream stream)
    {
        using var w = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        w.Write(Magic); w.Write(Version);
        Spec.Write(w);
        WriteArray(w, FeatureWeights); WriteArray(w, FeatureBiases);
        foreach (var d in DenseLayers.SelectMany(stack => stack)) { WriteArray(w, d.Weights); WriteArray(w, d.Biases); }
    }

    /// <summary>Reads a float checkpoint from a file.</summary>
    public static FloatModel Load(string path) { using var fs = File.OpenRead(path); return Load(fs); }
    /// <summary>Reads a float checkpoint written by <see cref="Save(Stream)"/>.</summary>
    public static FloatModel Load(Stream stream)
    {
        try
        {
            using var r = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            if (r.ReadUInt32() != Magic) throw new InvalidDataException("Not a float checkpoint.");
            int version = r.ReadInt32();
            if (version != Version) throw new InvalidDataException($"Unsupported checkpoint version {version}.");
            var model = new FloatModel(NetworkSpec.Read(r));
            ReadArray(r, model.FeatureWeights); ReadArray(r, model.FeatureBiases);
            foreach (var d in model.DenseLayers.SelectMany(stack => stack)) { ReadArray(r, d.Weights); ReadArray(r, d.Biases); }
            return model;
        }
        catch (EndOfStreamException) { throw new InvalidDataException("Unexpected end of file (truncated checkpoint)."); }
    }

    private static void WriteArray(BinaryWriter w, float[] a) => w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan()));
    private static void ReadArray(BinaryReader r, float[] a)
    {
        Span<byte> bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan());
        while (!bytes.IsEmpty)
        {
            int n = r.Read(bytes);
            if (n <= 0) throw new EndOfStreamException();
            bytes = bytes[n..];
        }
    }
}
