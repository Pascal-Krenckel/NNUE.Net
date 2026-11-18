namespace Nnue.Models;

/// <summary>The asynchronous part of model loading (kept apart from <see cref="NnueModel"/> because awaiting is not allowed in its unsafe code).</summary>
internal static class NnueModelLoader
{
    public static async Task<NnueModel> LoadAsync(Stream stream, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        var probe = new byte[NnueModel.HeaderProbeLength];
        int probed = await stream.ReadAtLeastAsync(probe, probe.Length, throwOnEndOfStream: false, cancellationToken).ConfigureAwait(false);
        int offset = NnueModel.ParseHeader(probe.AsSpan(0, probed), out var spec, out long blobSize);
        var builder = NnueModel.CreateBuilder(spec, blobSize);
        try
        {
            var blob = builder.RawBlobMemory;
            int have = (int)Math.Min(Math.Max(probed - offset, 0), blob.Length);
            probe.AsMemory(offset, have).CopyTo(blob);
            await stream.ReadExactlyAsync(blob[have..], cancellationToken).ConfigureAwait(false);
            int overRead = probed - offset - have;
            if (overRead > 0 && stream.CanSeek) stream.Seek(-overRead, SeekOrigin.Current);     // give back what the probe read past the model
            return builder.Build();
        }
        catch (EndOfStreamException) { builder.Abandon(); throw new InvalidDataException("Unexpected end of file (truncated model)."); }
        catch { builder.Abandon(); throw; }
    }

    public static async Task<NnueModel> LoadAsync(string path, CancellationToken cancellationToken)
    {
        await using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16, useAsync: true);
        return await LoadAsync(file, cancellationToken).ConfigureAwait(false);
    }
}
