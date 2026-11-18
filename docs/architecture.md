# Architecture

How the network computes, how the SIMD backends are organised, how the weights are laid out in memory and on disk, and why.
The measurements behind the decisions are in [`Nnue.Benchmarks/RESULTS.md`](../Nnue.Benchmarks/RESULTS.md).

## The network

```
features ──(shared int16 weights)──▶ Slots accumulators ──activation──▶ concat ──▶ [dense layers] × OutputBuckets ──▶ 1 value
```

1. **Feature transformer.** Every accumulator is `bias + Σ rows of the active features` (int16). All slots share the weights.
2. **Activation** of every slot, concatenated in slot order: clipped ReLU, squared clipped ReLU, or pairwise (the two halves of an
   accumulator multiplied, which halves its size).
3. **Head**, selected by the output bucket:
   - `Int16`: one int16 output layer, a dot product over all slots, then `+ bias` and the scale.
   - `Int8`: hidden dense layers with uint8 activations and int8 weights, then a one-output layer (Stockfish's arithmetic).
4. **Score** = raw integer output × `OutputScale` / (quantization scale), truncated to `int`.

### Activations

| Function | Float | Integer |
|---|---|---|
| ClippedReLU (hidden) | `clamp(z, 0, 1)` | `clamp(z_int >> 6, 0, 127)` |
| SqrClippedReLU (hidden) | `min(z² · 127/128, 1)`, the square of the **unclipped** z | `min(127, z_int² >> 19)` |
| ClippedReLU (feature transformer) | `clamp(a, 0, 1)` | Int16: `clamp(acc, 0, ActivationQuantization)` · Int8: `clamp(acc, 0, 127)` |
| SquaredClippedReLU (feature transformer) | `clamp(a, 0, 1)²` | Int16: `Σ ((c·w) as int16)·c` with `c = clamp(acc, 0, ActivationQuantization)` · Int8: `(c·c) >> 9`, `c = clamp(acc, 0, 255)` |
| Pairwise (feature transformer) | `clamp(a, 0, 1)·clamp(b, 0, 1)` | `(clamp(a, 0, 255) · clamp(b, 0, 255)) >> 9` |
| MaxPooling (feature transformer) | `max(clamp(a, 0, 1), clamp(b, 0, 1))` | `max(clamp(a, 0, 127), clamp(b, 0, 127))` |
| AveragePooling (feature transformer) | `(clamp(a, 0, 1) + clamp(b, 0, 1)) / 2` | `(clamp(a, 0, 127) + clamp(b, 0, 127) + 1) >> 1` |
| Quantmoid4 (hidden) | `v = 1 - min(\|z\|, 1)`, `q = 127/256 · v²`, `z < 0 ? q : 126/127 - q` | `x = z_int >> 6`, `v = 127 - min(\|x\|, 127)`, `q = v² >> 8`, `x < 0 ? q : 126 - q` |

Pairwise and the two pooling activations combine the first and the second half of an accumulator, so a slot contributes
`AccumulatorSize / 2` inputs. Pooling uses 127 as 1.0 (like clipped ReLU), pairwise 255. Quantmoid4 is the activation that the
Stockfish wiki describes as the integer replacement of `sigmoid(4x)`; its float form is continuous and smooth, the trainer uses its
exact derivative `127/128 · (1 - min(|z|, 1))`.

### Int8 arithmetic

As in Stockfish's `src/nnue`: accumulators int16; activations uint8 with 1.0 = 127; weights int8 with 1.0 = 64; biases int32 with
1.0 = 127·64. A hidden layer computes `u8 × i8 → int32`, shifts right by 6 and clamps to [0, 127]. `|w| ≤ 127` and inputs ≤ 127
guarantee that the 16-bit multiply-add of SSSE3/AVX2/AVX-512 (`pmaddubsw`) never saturates, which makes all backends and the
VNNI instruction bit-identical; the trainer clamps the weights to that range after every step.

`HiddenActivation.SqrClippedReLU` doubles every hidden layer's activations to `[sqr(z) | clip(z)]`. With `DenseConnections` a
layer reads the concatenated activations of all earlier hidden layers, without them only the previous layer's. The skip connection
adds `z0[n-2] - z0[n-1]` of the first hidden layer's raw outputs to the result.

## Backends

One struct per instruction set implements the static abstract interface `IBackend`; the evaluation core `Core<TBackend>` is generic
over it, so the JIT resolves every kernel call at compile time. An evaluator binds the core of its backend once, as function
pointers. All backends are bit-identical to the scalar reference, which the tests check for 19 model configurations.

| `SimdBackend` | Requirement | Notes |
|---|---|---|
| `Avx512Vnni` | AVX-512 F+BW and AVX512-VNNI, **net11.0 build** | AVX-512 kernels; `vpdpbusd zmm` (64 u8×i8 products per instruction, 8 independent accumulators) for the int8 dense layers. .NET 11 exposes it as `AvxVnni.V512`. |
| `Avx512` | AVX-512 F+BW | 512-bit kernels everywhere |
| `AvxVnni` | AVX2 and AVX-VNNI | AVX2 kernels; 256-bit `vpdpbusd` for the int8 dense layers (a preview API in .NET 8, used behind a runtime check; `-p:DefineConstants=NNUE_NO_AVXVNNI` compiles it out) |
| `Avx2` | AVX2 | |
| `Sse` | SSSE3 | |
| `NeonDotProd` | AArch64 with DotProd | `sdot` for the int8 dense layers |
| `Neon` | AArch64 | `smull` + `addp` + `sadalp` |
| `Scalar` | any | the reference implementation |
| `AvxVnniEmulated`, `Avx512VnniEmulated` | AVX2 / AVX-512 | **test aids**: the VNNI kernels with a software model of `vpdpbusd`, so their structure is tested on any CPU and runtime. Only listed in `SimdSupport.AvailableForTesting`. |

`SimdSupport.Best` picks AVX512-VNNI, then AVX-512 (measured equal or faster than the 256-bit AVX-VNNI on CPUs with both), then
AVX-VNNI, AVX2, SSSE3, NEON with dot product, NEON, scalar. The AVX512-VNNI preference follows from the instruction count and has
not been measured yet. Adding a backend means one kernel class, one backend struct and one line in `FnTable.For`.

### Kernel design

- **Accumulator refresh and update** process the accumulator in register tiles (AVX-512: 4 vectors = 128 values, AVX2 and 128-bit:
  8 vectors), so the adds of one feature row are independent: 20-45 % faster than one vector per pass while the feature matrix is
  in cache.
- **Int8 dense layers** broadcast 4 input bytes and multiply them with the weights of 16 outputs at once. Several independent
  accumulators (4 for `pmaddubsw`, 8 for `vpdpbusd`) hide the instruction latency.
- **Sparse first layer.** After the feature transformer many activations are zero. `SparseInputMode.Auto` counts the non-zero
  4-byte chunks; if at most 40 % are non-zero it builds their index list (a 256-entry lookup table per 8 chunks, branch free) and
  only multiplies those. Results are identical; it is several times faster on sparse activations and slightly slower on dense ones.
- **Pooling and Quantmoid4** run on the 128-bit kernels (`Portable128Kernels`) on every SIMD backend: they touch a few hundred bytes per
  evaluation, next to the dense layers they are negligible.
- **The last layer** has one output and is stored as a contiguous vector, so it is a plain dot product.
- **Scratch memory** of an Int8 evaluation is `stackalloc` without zeroing (zeroing cost 5-12 %), aligned to 64 bytes by hand.
- **Checks** are combined with non-short-circuit operators, so the valid path costs one predictable branch; the failure paths are
  separate non-inlined methods.

## Memory layout

- **Everything the kernels touch is 64-byte aligned** (weights, biases, accumulators): 15-25 % faster on AVX2 and AVX-512. Managed
  arrays are only aligned by luck, so models and accumulator buffers live in native memory from `NativeMemory.AlignedAlloc(size, 64)`,
  zero-initialised and freed by `Dispose` or a finalizer. Accumulator handles keep their memory block reachable.
- **Aligned instructions.** All loads and stores of weights, biases, accumulators and int32 scratch use `LoadAligned`/`StoreAligned`.
  This is safe because `AccumulatorSize % 32 == 0` and every tensor starts at a multiple of 64 bytes. The only unaligned stores are
  the byte activations, whose slot and layer offsets are multiples of 16 or 32. Misaligned pointers would fault, which is why
  accumulators can only be created by an `AccumulatorBuffer`.
- **One blob per model.** All tensors of a model are one allocation, in exactly the layout of the file, so loading is one read and
  memory mapping needs no conversion.
- **Int8 hidden-layer weights are blocked**: groups of 4 inputs per output are contiguous, and the groups of all outputs for the same
  4 inputs follow each other (`index = (i / 4) · paddedOutputs · 4 + o · 4 + i % 4`). One broadcast of 4 input bytes then meets the
  weights of 16 outputs in one 512-bit load. Output counts are padded to multiples of 16.

Decisions that were measured and rejected: how a kernel is called (switch, function pointer, interface, generic) makes no
difference; bounds checks at random positions, prefetching and row-wise instead of chunk-wise accumulator updates gave nothing.

## Model file format (version 3)

Little endian.

| Field | Type |
|---|---|
| magic `"NNUM"` | `uint32` 0x4D554E4E |
| version | `int32` 3 |
| `FeatureCount`, `AccumulatorSize`, `Slots` | 3 × `int32` |
| `Precision`, `Activation`, `HiddenActivation` | 3 × `int32` (enum values) |
| `DenseConnections`, `SkipConnection` | 2 × `byte` (0/1) |
| `OutputBuckets`, `ActivationQuantization`, `OutputWeightQuantization`, `OutputScale` | 4 × `int32` |
| hidden layer count, then the sizes | `int32`, n × `int32` |
| blob size | `int64` |
| padding | zero bytes up to the next multiple of 64 |
| blob | the tensors below, each starting at a multiple of 64 bytes |

Blob: feature weights `int16[FeatureCount][AccumulatorSize]`, feature biases `int16[AccumulatorSize]`, then

- `Int16`: per bucket the output weights `int16[Slots · slot output size]` (each bucket padded to 64 bytes), then the output biases
  `int32[OutputBuckets]`;
- `Int8`: per bucket and layer the weights (`sbyte`, blocked as above; the last layer a plain vector) and the biases
  `int32[paddedOutputs]`.

Version 2 files (no padding before the blob) still load, but cannot be memory-mapped. The spec is validated on reading, so a
damaged or hostile file fails with `InvalidDataException`.
