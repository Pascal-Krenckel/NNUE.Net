using System.Numerics;
using System.Runtime.InteropServices;
using Nnue.Configuration;
using Nnue.Training.Models;
using Nnue.Training.Quantization;

namespace Nnue.Training.Optimization;
/// <summary>
/// Mini-batch trainer for the full architecture (feature transformer + any number of dense layers, CReLU or pairwise
/// feature activation). Loss = mean((sigmoid(output) - target)^2), optimizer = Adam.
/// Feature-transformer rows are updated "lazily": only rows of features that appeared in the batch get an Adam step
/// (the usual trick that makes huge sparse input layers affordable).
/// </summary>
public sealed class NnueTrainer
{
    private readonly NetworkSpec _cfg;
    private readonly NetworkLayout _layout;
    private readonly int _k;
    private readonly int _h;
    private readonly int _layers;
    private readonly int _buckets;
    private readonly int[] _actOff, _inOff, _inLen;   // activation block offset per hidden layer; input segment per dense layer k >= 1
    private readonly float[] _denseClip;
    private readonly Worker[] _workers;
    private readonly ParallelOptions _po;
    private readonly Random _rng;

    // Adam state
    private readonly float[] _mFtW, _vFtW, _mFtB, _vFtB;
    private readonly float[][][] _mW, _vW, _mB, _vB;   // [bucket][layer][...]
    private long _step;

    private readonly bool[] _unionFlag;
    private readonly List<int> _union = [];
    private int[] _order = [];

    /// <summary>The network being trained.</summary>
    public FloatModel Model { get; }
    /// <summary>The trainer's own copy of the options (made in the constructor); its learning rate decays during <see cref="Train"/>, the
    /// caller's options object is never changed.</summary>
    public TrainerOptions Options { get; }

    /// <summary>Creates a trainer for <paramref name="network"/> (default options if <paramref name="options"/> is null).</summary>
    public NnueTrainer(FloatModel network, TrainerOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        Model = network;
        _cfg = network.Spec;
        _layout = network.Layout;
        _k = _cfg.Slots;
        _h = _cfg.AccumulatorSize;
        _layers = network.DenseLayers[0].Length;
        _buckets = _cfg.OutputBuckets;
        _actOff = _layout.DenseLayers.Select(d => d.ActivationOffset).ToArray();
        _inOff = new int[_layers];
        _inLen = new int[_layers];
        for (int k = 1; k < _layers; k++)
            (_inOff[k], _inLen[k]) = (_layout.DenseLayers[k].InputOffset, _layout.DenseLayers[k].InputLength);
        Options = options?.Clone() ?? new TrainerOptions();
        Options.Validate();
        int threads = Math.Max(1, Options.Threads);

        _denseClip = Enumerable.Range(0, _layers).Select(k => Options.DenseWeightClip ?? NnueQuantizer.GetMaxWeight(_cfg, k)).ToArray();

        _mFtW = new float[network.FeatureWeights.Length];
        _vFtW = new float[network.FeatureWeights.Length];
        _mFtB = new float[_h];
        _vFtB = new float[_h];
        float[][][] PerLayer(Func<FloatDense, int> size) =>
            network.DenseLayers.Select(stack => stack.Select(d => new float[size(d)]).ToArray()).ToArray();
        _mW = PerLayer(d => d.Weights.Length);
        _vW = PerLayer(d => d.Weights.Length);
        _mB = PerLayer(d => d.OutputLength);
        _vB = PerLayer(d => d.OutputLength);

        _workers = new Worker[threads];
        for (int i = 0; i < threads; i++)
            _workers[i] = new Worker(_layout);
        _unionFlag = new bool[_cfg.FeatureCount];
        _po = new ParallelOptions { MaxDegreeOfParallelism = threads };
        _rng = new Random(Options.Seed);
    }

    // =====================================================================
    //  Public API
    // =====================================================================

    /// <summary>Runs one shuffled pass over the data. Returns the mean training loss seen during the epoch.</summary>
    /// <param name="data">The training samples.</param>
    /// <param name="cancellationToken">Checked before every batch. After cancellation the weights are those of the last completed batch.</param>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public double TrainEpoch(IReadOnlyList<TrainingSample> data, CancellationToken cancellationToken = default)
    {
        ValidateSamples(data);
        int n = data.Count;
        if (n == 0)
            throw new ArgumentException("No training data.", nameof(data));

        if (_order.Length != n)
            _order = new int[n];
        for (int i = 0; i < n; i++)
            _order[i] = i;
        for (int i = n - 1; i > 0; i--)
        { int j = _rng.Next(i + 1); (_order[i], _order[j]) = (_order[j], _order[i]); }

        double total = 0;
        for (int start = 0; start < n; start += Options.BatchSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            total += RunBatch(data, start, Math.Min(Options.BatchSize, n - start));
        }
        return total / n;
    }

    /// <summary>
    /// Trains for <paramref name="epochs"/> epochs, multiplying the learning rate by <see cref="TrainerOptions.LearningRateDecay"/> after each.
    /// </summary>
    /// <param name="data">The training samples.</param>
    /// <param name="epochs">Number of passes over the data.</param>
    /// <param name="progress">Receives a <see cref="TrainingProgress"/> after every epoch. It is called on the training thread; a
    /// <see cref="Progress{T}"/> posts it to the captured synchronization context instead.</param>
    /// <param name="cancellationToken">Checked before every batch. After cancellation the weights are those of the last completed batch.</param>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public void Train(IReadOnlyList<TrainingSample> data, int epochs, IProgress<TrainingProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(epochs);
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            float learningRate = Options.LearningRate;
            double loss = TrainEpoch(data, cancellationToken);
            progress?.Report(new TrainingProgress(epoch, epochs, loss, learningRate));
            Options.LearningRate *= Options.LearningRateDecay;
        }
    }

    /// <summary>
    /// Checks every sample before any work starts, so a bad sample fails with a message that names it instead of an
    /// <see cref="IndexOutOfRangeException"/> inside a worker thread (and a half-applied batch). Linear in the number of features.
    /// </summary>
    private void ValidateSamples(IReadOnlyList<TrainingSample> data)
    {
        ArgumentNullException.ThrowIfNull(data);
        int featureCount = _cfg.FeatureCount;
        for (int i = 0; i < data.Count; i++)
        {
            TrainingSample sample = data[i] ?? throw new ArgumentException($"Sample {i} is null.", nameof(data));
            if (sample.Features is null || sample.Features.Length != _k)
                throw new ArgumentException($"Sample {i} has {sample.Features?.Length ?? 0} feature lists, the model has {_k} slot(s).", nameof(data));
            if ((uint)sample.Bucket >= (uint)_buckets)
                throw new ArgumentException($"Sample {i}: bucket {sample.Bucket} out of range [0, {_buckets}).", nameof(data));
            if (!float.IsFinite(sample.Target))
                throw new ArgumentException($"Sample {i}: the target is not a finite number.", nameof(data));
            for (int s = 0; s < _k; s++)
            {
                int[] features = sample.Features[s] ?? throw new ArgumentException($"Sample {i}: feature list {s} is null.", nameof(data));
                foreach (int f in features)
                    if ((uint)f >= (uint)featureCount)
                        throw new ArgumentException($"Sample {i}, slot {s}: feature {f} out of range [0, {featureCount}).", nameof(data));
            }
        }
    }

    /// <summary>Mean loss over a data set without changing the weights (use for validation).</summary>
    public double ComputeLoss(IReadOnlyList<TrainingSample> data)
    {
        ValidateSamples(data);
        int n = data.Count;
        if (n == 0)
            return 0;
        int t = _workers.Length, chunk = (n + t - 1) / t;
        _ = Parallel.For(0, t, _po, k =>
        {
            Worker w = _workers[k];
            double s = 0;
            int end = Math.Min(n, (k + 1) * chunk);
            for (int i = k * chunk; i < end; i++)
            {
                TrainingSample p = data[i];
                float err = Error(Forward(w, p), p);
                s += err * err;
            }
            w.Loss = s;
        });
        return _workers.Sum(w => w.Loss) / n;
    }

    /// <summary>
    /// Verifies backpropagation: compares the analytic gradient of one sample with central finite differences of the loss,
    /// for random parameters of every group (feature transformer weights/bias, every dense layer's weights/bias of the sample's bucket).
    /// Does not change the network or the optimizer state.
    /// </summary>
    public GradientCheckResult CheckGradients(TrainingSample sample, int probesPerGroup = 6, float epsilon = 1e-2f, double tolerance = 0.05, Action<string>? onFailure = null)
    {
        ValidateSamples([sample]);
        Worker w = _workers[0];
        ClearGradients(w);
        _ = TrainSample(w, sample);                                  // analytic gradient of this single sample

        Random rng = new(12345);
        int h = _h, bucket = sample.Bucket;
        FloatDense[] stack = Model.DenseLayers[bucket];
        List<GradientCheckGroup> groups = [];

        double Loss()
        { float err = Error(Forward(_workers[0], sample), sample); return (double)err * err; }

        // Central difference with step e. The activations are piecewise smooth, so a probe is only meaningful if no kink
        // (clamp boundary) lies within the step: we evaluate two step sizes and skip probes whose numeric derivatives disagree.
        double Numeric(float[] param, int i, float e)
        {
            float old = param[i];
            param[i] = old + e;
            double lp = Loss();
            param[i] = old - e;
            double lm = Loss();
            param[i] = old;
            return (lp - lm) / (2.0 * e);
        }

        void Group(string name, float[] param, float[] grad, Func<int> randomIndex)
        {
            int probes = 0, failures = 0, skipped = 0;
            double maxRel = 0;
            for (int attempt = 0; attempt < 600 && probes < probesPerGroup; attempt++)
            {
                int i = randomIndex();
                float analytic = grad[i];
                if (MathF.Abs(analytic) < 1e-4f)
                    continue;           // probing zero gradients proves little
                double nCoarse = Numeric(param, i, epsilon), nFine = Numeric(param, i, epsilon / 4f);
                double scale = Math.Max(Math.Abs(nFine), Math.Abs(analytic));
                if (Math.Abs(nCoarse - nFine) > (0.02 * scale) + 2e-5)
                { skipped++; continue; }   // kink inside the step
                double diff = Math.Abs(nFine - analytic);
                probes++;
                if (diff > (tolerance * scale) + 2e-5)
                { failures++; onFailure?.Invoke($"{name}[{i}] analytic {analytic:E4} numeric(epsilon) {nCoarse:E4} numeric(epsilon/4) {nFine:E4}"); }
                maxRel = Math.Max(maxRel, diff / scale);
            }
            groups.Add(new GradientCheckGroup(name, probes, failures, maxRel, skipped));
        }

        int[] active = sample.Features.SelectMany(f => f).Distinct().ToArray();
        Group("ft.bias", Model.FeatureBiases, w.GFtB, () => rng.Next(h));
        Group("ft.weights", Model.FeatureWeights, w.GFtW, () => (active[rng.Next(active.Length)] * h) + rng.Next(h));
        for (int k = 0; k < _layers; k++)
        {
            FloatDense d = stack[k];
            Group($"dense{k}.weights", d.Weights, w.GW[bucket][k], () => rng.Next(d.Weights.Length));
            Group($"dense{k}.bias", d.Biases, w.GB[bucket][k], () => rng.Next(d.OutputLength));
        }

        ClearGradients(w);
        return new GradientCheckResult(groups);
    }

    private void ClearGradients(Worker w)
    {
        for (int i = 0; i < w.TouchedCount; i++)
        { int f = w.TouchedList[i]; w.Touched[f] = false; Array.Clear(w.GFtW, f * _h, _h); }
        w.TouchedCount = 0;
        Array.Clear(w.GFtB);
        for (int b = 0; b < _buckets; b++)
            for (int k = 0; k < _layers; k++)
            { Array.Clear(w.GW[b][k]); Array.Clear(w.GB[b][k]); }
        Array.Clear(w.BucketUsed);
    }

    // =====================================================================
    //  Batch processing
    // =====================================================================

    private double RunBatch(IReadOnlyList<TrainingSample> data, int start, int count)
    {
        int t = _workers.Length, chunk = (count + t - 1) / t;
        _ = Parallel.For(0, t, _po, k =>
        {
            Worker w = _workers[k];
            w.Loss = 0;
            int s = start + (k * chunk), e = Math.Min(start + count, s + chunk);
            for (int i = s; i < e; i++)
                w.Loss += TrainSample(w, data[_order[i]]);
        });

        ApplyUpdate(count);
        return _workers.Sum(w => w.Loss);
    }

    private double TrainSample(Worker w, TrainingSample p)
    {
        int h = _h;
        float o = Forward(w, p);
        float err, g;                                      // g = dLoss/dOutput
        if (Options.Loss == LossFunction.SigmoidMse)
        { float pred = Sigmoid(o); err = pred - p.Target; g = 2f * err * pred * (1f - pred); }
        else
        { err = o - p.Target; g = 2f * err; }

        // ---- dense layers of this position's bucket, last to first ----
        int bucket = p.Bucket;
        FloatDense[] stack = Model.DenseLayers[bucket];
        float[][] gW = w.GW[bucket];
        float[][] gB = w.GB[bucket];
        w.BucketUsed[bucket] = true;
        int last = _layers - 1;
        bool sqr = _cfg.HiddenActivation == HiddenActivation.SqrClippedReLU, quantmoid = _cfg.HiddenActivation == HiddenActivation.Quantmoid4;
        Array.Clear(w.GAct);
        w.GZ[last][0] = g;
        for (int k = last; k >= 0; k--)
        {
            FloatDense d = stack[k];
            float[] gz = w.GZ[k];

            if (k < last)
            {
                // gradient through this layer's activation(s): GAct -> GZ
                float[] z = w.Z[k];
                int at = _actOff[k];
                if (sqr)
                    for (int j = 0; j < d.OutputLength; j++)
                        gz[j] = (w.GAct[at + j] * Activations.SqrClippedReLUDerivative(z[j])) + (w.GAct[at + d.OutputLength + j] * Activations.ClippedReLUDerivative(z[j]));
                else if (quantmoid)
                    for (int j = 0; j < d.OutputLength; j++)
                        gz[j] = w.GAct[at + j] * Activations.Quantmoid4Derivative(z[j]);
                else
                    for (int j = 0; j < d.OutputLength; j++)
                        gz[j] = w.GAct[at + j] * Activations.ClippedReLUDerivative(z[j]);
                if (k == 0 && _cfg.SkipConnection)
                { gz[d.OutputLength - 2] += g; gz[d.OutputLength - 1] -= g; }   // out += z0[n-2] - z0[n-1]
            }

            ReadOnlySpan<float> x = k == 0 ? w.X0 : w.Act.AsSpan(_inOff[k], _inLen[k]);
            Span<float> gIn = k == 0 ? w.GX0 : w.GIn.AsSpan(0, d.InputLength);
            gIn.Clear();
            for (int o2 = 0; o2 < d.OutputLength; o2++)
            {
                float gzo = gz[o2];
                if (gzo == 0f)
                    continue;
                gB[k][o2] += gzo;
                Axpy(gW[k].AsSpan(o2 * d.InputLength, d.InputLength), gzo, x);
                Axpy(gIn, gzo, d.Weights.AsSpan(o2 * d.InputLength, d.InputLength));
            }
            if (k > 0)
                AddTo(w.GAct.AsSpan(_inOff[k], _inLen[k]), gIn);   // consumers add into the producers' activation gradient
        }

        FtBackward(w);

        for (int s = 0; s < _k; s++)
        {
            AddTo(w.GFtB, w.D[s]);
            foreach (int f in p.Features[s])
            { AddTo(w.GFtW.AsSpan(f * h, h), w.D[s]); w.Mark(f); }
        }

        return err * err;
    }

    /// <summary>Forward pass; leaves all intermediate values in the worker. Returns the raw output.</summary>
    private float Forward(Worker w, TrainingSample p)
    {
        if ((uint)p.Bucket >= (uint)_buckets)
            throw new ArgumentOutOfRangeException(nameof(p), $"Bucket {p.Bucket} out of range (OutputBuckets = {_buckets}).");
        FloatDense[] stack = Model.DenseLayers[p.Bucket];
        if (p.Features.Length != _k)
            throw new ArgumentException($"Sample has {p.Features.Length} feature lists, the model has {_k} slot(s).");
        for (int s = 0; s < _k; s++)
            Accumulate(w.Acc[s], p.Features[s]);
        FtForward(w);

        int last = _layers - 1;
        bool sqr = _cfg.HiddenActivation == HiddenActivation.SqrClippedReLU, quantmoid = _cfg.HiddenActivation == HiddenActivation.Quantmoid4;
        float skip = 0f;
        for (int k = 0; k <= last; k++)
        {
            FloatDense d = stack[k];
            ReadOnlySpan<float> x = k == 0 ? w.X0 : w.Act.AsSpan(_inOff[k], _inLen[k]);
            float[] z = w.Z[k];
            for (int o = 0; o < d.OutputLength; o++)
                z[o] = d.Biases[o] + Dot(d.Weights.AsSpan(o * d.InputLength, d.InputLength), x);
            if (k == last)
                return z[0] + skip;
            if (k == 0 && _cfg.SkipConnection)
                skip = z[d.OutputLength - 2] - z[d.OutputLength - 1];

            int at = _actOff[k];
            if (sqr)
                for (int j = 0; j < d.OutputLength; j++)
                { w.Act[at + j] = Activations.SqrClippedReLU(z[j]); w.Act[at + d.OutputLength + j] = Activations.ClippedReLU(z[j]); }
            else if (quantmoid)
                for (int j = 0; j < d.OutputLength; j++)
                    w.Act[at + j] = Activations.Quantmoid4(z[j]);
            else
                for (int j = 0; j < d.OutputLength; j++)
                    w.Act[at + j] = Activations.ClippedReLU(z[j]);
        }
        throw new InvalidOperationException();
    }

    private void Accumulate(float[] acc, int[] features)
    {
        int h = _h;
        Model.FeatureBiases.CopyTo(acc, 0);
        foreach (int f in features)
            AddTo(acc, Model.FeatureWeights.AsSpan(f * h, h));
    }

    /// <summary>accumulators -> X0: every slot writes its own segment of the concatenated feature transformer output</summary>
    private void FtForward(Worker w)
    {
        int h = _h, slotOut = _layout.SlotOutputSize;
        float[] x = w.X0;
        for (int s = 0; s < _k; s++)
        {
            float[] acc = w.Acc[s];
            int o = s * slotOut;
            switch (_cfg.Activation)
            {
                case FeatureTransformerActivation.ClippedReLU:
                    for (int j = 0; j < h; j++)
                        x[o + j] = Activations.ClippedReLU(acc[j]);
                    break;
                case FeatureTransformerActivation.SquaredClippedReLU:
                    for (int j = 0; j < h; j++)
                        x[o + j] = Activations.SquaredClippedReLU(acc[j]);
                    break;
                case FeatureTransformerActivation.MaxPooling:
                    for (int j = 0; j < slotOut; j++)
                        x[o + j] = Activations.MaxPool(acc[j], acc[j + slotOut]);
                    break;
                case FeatureTransformerActivation.AveragePooling:
                    for (int j = 0; j < slotOut; j++)
                        x[o + j] = Activations.AveragePool(acc[j], acc[j + slotOut]);
                    break;
                default:   // pairwise
                    for (int j = 0; j < slotOut; j++)
                        x[o + j] = Activations.ClippedReLU(acc[j]) * Activations.ClippedReLU(acc[j + slotOut]);
                    break;
            }
        }
    }

    /// <summary>dL/dX0 (w.GX0) -> dL/dAccumulators (w.D[slot])</summary>
    private void FtBackward(Worker w)
    {
        int h = _h, slotOut = _layout.SlotOutputSize;
        float[] gx = w.GX0;
        for (int s = 0; s < _k; s++)
        {
            float[] acc = w.Acc[s];
            float[] d = w.D[s];
            int o = s * slotOut;
            switch (_cfg.Activation)
            {
                case FeatureTransformerActivation.ClippedReLU:
                    for (int j = 0; j < h; j++)
                        d[j] = gx[o + j] * Activations.ClippedReLUDerivative(acc[j]);
                    break;
                case FeatureTransformerActivation.SquaredClippedReLU:
                    for (int j = 0; j < h; j++)
                        d[j] = gx[o + j] * Activations.SquaredClippedReLUDerivative(acc[j]);
                    break;
                case FeatureTransformerActivation.MaxPooling:   // the gradient goes to the larger clipped input (the first one on a tie)
                    for (int j = 0; j < slotOut; j++)
                    {
                        float a = acc[j], b = acc[j + slotOut], g = gx[o + j];
                        bool first = Activations.ClippedReLU(a) >= Activations.ClippedReLU(b);
                        d[j] = first ? g * Activations.ClippedReLUDerivative(a) : 0f;
                        d[j + slotOut] = first ? 0f : g * Activations.ClippedReLUDerivative(b);
                    }
                    break;
                case FeatureTransformerActivation.AveragePooling:
                    for (int j = 0; j < slotOut; j++)
                    {
                        float g = 0.5f * gx[o + j];
                        d[j] = g * Activations.ClippedReLUDerivative(acc[j]);
                        d[j + slotOut] = g * Activations.ClippedReLUDerivative(acc[j + slotOut]);
                    }
                    break;
                default:   // pairwise: out_j = clip(a_j) * clip(a_{j+half})
                    for (int j = 0; j < slotOut; j++)
                    {
                        float a = acc[j], b = acc[j + slotOut], g = gx[o + j];
                        d[j] = g * Activations.ClippedReLUDerivative(a) * Activations.ClippedReLU(b);
                        d[j + slotOut] = g * Activations.ClippedReLUDerivative(b) * Activations.ClippedReLU(a);
                    }
                    break;
            }
        }
    }

    /// <summary>prediction - target of the configured loss</summary>
    private float Error(float output, TrainingSample p)
        => Options.Loss == LossFunction.SigmoidMse ? Sigmoid(output) - p.Target : output - p.Target;

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    // =====================================================================
    //  Optimizer
    // =====================================================================

    private void ApplyUpdate(int batchCount)
    {
        _step++;
        float lrT = Options.LearningRate * MathF.Sqrt(1f - MathF.Pow(Options.Beta2, _step)) / (1f - MathF.Pow(Options.Beta1, _step));
        float scale = 1f / batchCount;
        Worker w0 = _workers[0];

        // ---- feature transformer bias: reduce worker gradients into worker 0, step, clear ----
        for (int k = 1; k < _workers.Length; k++)
        {
            AddTo(w0.GFtB, _workers[k].GFtB);
            Array.Clear(_workers[k].GFtB);
        }
        Adam(Model.FeatureBiases, _mFtB, _vFtB, w0.GFtB, scale, lrT, Options.FeatureWeightClip);
        Array.Clear(w0.GFtB);

        // ---- dense outputBuckets: only buckets that occurred in this batch are updated (lazy, like the FT rows) ----
        for (int bk = 0; bk < _buckets; bk++)
        {
            if (!_workers.Any(wk => wk.BucketUsed[bk]))
                continue;
            for (int k = 1; k < _workers.Length; k++)
            {
                Worker wk = _workers[k];
                if (!wk.BucketUsed[bk])
                    continue;
                for (int l = 0; l < _layers; l++)
                {
                    AddTo(w0.GW[bk][l], wk.GW[bk][l]);
                    Array.Clear(wk.GW[bk][l]);
                    for (int o = 0; o < w0.GB[bk][l].Length; o++)
                        w0.GB[bk][l][o] += wk.GB[bk][l][o];
                    Array.Clear(wk.GB[bk][l]);
                }
            }
            for (int l = 0; l < _layers; l++)
            {
                FloatDense d = Model.DenseLayers[bk][l];
                Adam(d.Weights, _mW[bk][l], _vW[bk][l], w0.GW[bk][l], scale, lrT, _denseClip[l]);
                AdamScalar(d.Biases, _mB[bk][l], _vB[bk][l], w0.GB[bk][l], scale, lrT);
                Array.Clear(w0.GW[bk][l]);
                Array.Clear(w0.GB[bk][l]);
            }
        }
        foreach (Worker wk in _workers)
            Array.Clear(wk.BucketUsed);

        // ---- sparse feature-transformer rows ----
        _union.Clear();
        foreach (Worker wk in _workers)
        {
            for (int i = 0; i < wk.TouchedCount; i++)
            {
                int f = wk.TouchedList[i];
                wk.Touched[f] = false;
                if (!_unionFlag[f])
                { _unionFlag[f] = true; _union.Add(f); }
            }
            wk.TouchedCount = 0;
        }

        int h = _h;
        _ = Parallel.For(0, _union.Count, _po, idx =>
        {
            int off = _union[idx] * h;
            Span<float> g0 = w0.GFtW.AsSpan(off, h);
            for (int k = 1; k < _workers.Length; k++)
            {
                Span<float> gk = _workers[k].GFtW.AsSpan(off, h);
                AddTo(g0, gk);
                gk.Clear();
            }
            Adam(Model.FeatureWeights.AsSpan(off, h), _mFtW.AsSpan(off, h), _vFtW.AsSpan(off, h), g0, scale, lrT, Options.FeatureWeightClip);
            g0.Clear();
        });
        foreach (int f in _union)
            _unionFlag[f] = false;
    }

    private void Adam(Span<float> weights, Span<float> m, Span<float> v, ReadOnlySpan<float> grad,
                      float gradScale, float lrT, float clip)
    {
        Span<Vector<float>> wv = MemoryMarshal.Cast<float, Vector<float>>(weights);
        Span<Vector<float>> mv = MemoryMarshal.Cast<float, Vector<float>>(m);
        Span<Vector<float>> vv = MemoryMarshal.Cast<float, Vector<float>>(v);
        ReadOnlySpan<Vector<float>> gv = MemoryMarshal.Cast<float, Vector<float>>(grad);

        Vector<float> b1 = new(Options.Beta1);
        Vector<float> ob1 = new(1f - Options.Beta1);
        Vector<float> b2 = new(Options.Beta2);
        Vector<float> ob2 = new(1f - Options.Beta2);
        Vector<float> eps = new(Options.Epsilon);
        Vector<float> lr = new(lrT);
        Vector<float> sc = new(gradScale);
        Vector<float> hi = new(clip);
        Vector<float> lo = -hi;

        for (int i = 0; i < wv.Length; i++)
        {
            Vector<float> g = gv[i] * sc;
            mv[i] = (b1 * mv[i]) + (ob1 * g);
            vv[i] = (b2 * vv[i]) + (ob2 * g * g);
            Vector<float> nw = wv[i] - (lr * mv[i] / (Vector.SquareRoot(vv[i]) + eps));
            wv[i] = Vector.Min(Vector.Max(nw, lo), hi);
        }
    }

    /// <summary>Same update for tiny arrays whose length is not a multiple of the vector width (dense biases). Unclamped.</summary>
    private void AdamScalar(float[] w, float[] m, float[] v, float[] grad, float gradScale, float lrT)
    {
        for (int i = 0; i < w.Length; i++)
        {
            float g = grad[i] * gradScale;
            m[i] = (Options.Beta1 * m[i]) + ((1f - Options.Beta1) * g);
            v[i] = (Options.Beta2 * v[i]) + ((1f - Options.Beta2) * g * g);
            w[i] -= lrT * m[i] / (MathF.Sqrt(v[i]) + Options.Epsilon);
        }
    }

    // =====================================================================
    //  Vector helpers (lengths are multiples of Vector<float>.Count: HiddenSize % 32, layer sizes % 16)
    // =====================================================================

    private static void AddTo(Span<float> dst, ReadOnlySpan<float> src)
    {
        Span<Vector<float>> d = MemoryMarshal.Cast<float, Vector<float>>(dst);
        ReadOnlySpan<Vector<float>> s = MemoryMarshal.Cast<float, Vector<float>>(src);
        for (int i = 0; i < d.Length; i++)
            d[i] += s[i];
    }

    /// <summary>y += a * x</summary>
    private static void Axpy(Span<float> y, float a, ReadOnlySpan<float> x)
    {
        Span<Vector<float>> yv = MemoryMarshal.Cast<float, Vector<float>>(y);
        ReadOnlySpan<Vector<float>> xv = MemoryMarshal.Cast<float, Vector<float>>(x);
        Vector<float> av = new(a);
        for (int i = 0; i < yv.Length; i++)
            yv[i] += av * xv[i];
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        ReadOnlySpan<Vector<float>> av = MemoryMarshal.Cast<float, Vector<float>>(a);
        ReadOnlySpan<Vector<float>> bv = MemoryMarshal.Cast<float, Vector<float>>(b);
        Vector<float> sum = Vector<float>.Zero;
        for (int i = 0; i < av.Length; i++)
            sum += av[i] * bv[i];
        return Vector.Sum(sum);
    }

    // =====================================================================
    //  Per-thread scratch + gradient buffers
    // =====================================================================

    private sealed class Worker
    {
        public readonly float[][] Acc, D;      // per slot: accumulator and its gradient
        public readonly float[] X0, GX0;       // feature transformer output and its gradient
        public readonly float[][] Z, GZ;       // per dense layer: pre-activation output and its gradient
        public readonly float[] Act, GAct;     // concatenated hidden activations (same layout as the evaluator) and their gradient
        public readonly float[] GIn;           // scratch: gradient w.r.t. the input of a hidden-fed dense layer
        public readonly float[][][] GW, GB;    // [bucket][layer]
        public readonly bool[] BucketUsed;
        public readonly float[] GFtW, GFtB;
        public double Loss;
        public readonly bool[] Touched;
        public readonly int[] TouchedList;
        public int TouchedCount;

        public Worker(NetworkLayout l)
        {
            NetworkSpec spec = l.Spec;
            int h = l.AccumulatorSize;
            Acc = Enumerable.Range(0, l.Slots).Select(_ => new float[h]).ToArray();
            D = Enumerable.Range(0, l.Slots).Select(_ => new float[h]).ToArray();
            IReadOnlyList<DenseLayerInfo> shapes = l.DenseLayers;
            X0 = new float[l.FeatureTransformerOutputSize];
            GX0 = new float[l.FeatureTransformerOutputSize];
            Z = shapes.Select(s => new float[s.OutputLength]).ToArray();
            GZ = shapes.Select(s => new float[s.OutputLength]).ToArray();
            Act = new float[l.TotalActivationSize];
            GAct = new float[l.TotalActivationSize];
            GIn = new float[shapes.Skip(1).Select(s => s.InputLength).DefaultIfEmpty(0).Max()];
            GW = Enumerable.Range(0, spec.OutputBuckets).Select(_ => shapes.Select(s => new float[s.InputLength * s.OutputLength]).ToArray()).ToArray();
            GB = Enumerable.Range(0, spec.OutputBuckets).Select(_ => shapes.Select(s => new float[s.OutputLength]).ToArray()).ToArray();
            BucketUsed = new bool[spec.OutputBuckets];
            GFtW = new float[l.FeatureCount * h];
            GFtB = new float[h];
            Touched = new bool[l.FeatureCount];
            TouchedList = new int[l.FeatureCount];
        }

        public void Mark(int f)
        {
            if (!Touched[f])
            { Touched[f] = true; TouchedList[TouchedCount++] = f; }
        }
    }
}
