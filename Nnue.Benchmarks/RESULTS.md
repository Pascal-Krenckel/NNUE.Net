# Runtime layout experiments (evaluation path)

Host: Intel Xeon @ 2.1 GHz (Sapphire/Emerald-Rapids class, AVX-512 + AVX-VNNI), shared VM, .NET 8, one thread, tiered JIT off.
Method: every comparison runs its variants **interleaved for 9 rounds**; the tables show the **minimum** ns per operation
(median in parentheses where it matters). All variants of one comparison are checked to compute identical results.
Noise: ratios inside one comparison are reliable to roughly ±5 %; absolute numbers drift by up to 30 % between runs on this VM.
x86 only - nothing here was measured on ARM hardware.

These numbers were produced by the hand-written interleaved harness that the BenchmarkDotNet benchmarks replaced (`EvaluatorBenchmarks`, `AccumulatorBenchmarks`, `Int8StageBenchmarks`); the experiments that only compared design alternatives (call mechanisms, alignment, update strategies) were one-off and are not part of the project any more.
Rerun the current ones: `dotnet run -c Release --project Nnue.Benchmarks -- --filter "*"`

## A. How a kernel is called (AVX-512 kernels, ns per call)
| kernel | direct | enum switch (today) | function pointer | interface | generic static abstract | delegate |
|---|---|---|---|---|---|---|
| int16 dot, n=256 | 14.6 | 14.8 | 15.1 | 15.5 | 15.7 | 15.0 |
| int16→u8 CReLU, n=256 | 8.0 | 8.0 | 8.1 | 8.5 | 7.9 | 8.4 |
| row add, n=1024 | 25.5 | 25.4 | 28.7 | 26.4 | 28.3 | 33.1 |
| dense 1024→32 | 459 | 464 | 462 | 470 | 411 | 482 |

**The call mechanism is not a performance factor** (switch: 0–1 % over a direct call; everything else within noise).
Choose it for maintainability.

## D. Alignment (same kernels, different buffer addresses)
AVX-512, ns: 64-byte aligned / shifted by 8, 16 or 32 bytes / managed arrays (today)
| kernel | aligned | shifted | managed |
|---|---|---|---|
| int16 dot n=256 | 12.7 | 15.4–15.6 | 15.6 |
| row add n=1024 | 20.4 | 25.0–25.8 | 24.4 |
| dense 1024→32 | 277 | 338–342 | 343 |

A second run showed up to 1.8× for the dot product; a third run happened to place the managed array on a 64-byte boundary and
it was as fast as the aligned one: **managed array alignment is luck**. AVX2 (32-byte loads): +32 bytes behaves like aligned,
+16 costs 10–14 %, +8 costs 14–49 %, managed arrays +19 % (dense layer). A pinned managed array (`GC.AllocateArray(pinned: true)`)
aligned by offset is as fast as `NativeMemory.AlignedAlloc` (100.3–100.7 %).

## C. Accumulator layout and public API overhead
| operation | H | today (`short[][]`, public API) | raw kernel, managed arrays | raw kernel, aligned native block |
|---|---|---|---|---|
| copy a pair (push) | 256 | 28.9 | – | 6.0 (`Unsafe.CopyBlock`), 13.6 (`MemoryCopy`) |
| copy a pair (push) | 1024 | 50.7 | – | 19.1 / 28.1 |
| update 2+2 features | 256 | 32.5 | 29.3 | 22.3 |
| update 2+2 features | 1024 | 89.3 | 86.6 | 73.6 |
| refresh 16 features | 256 | 91.6 | – | 69.8 |
| refresh 16 features | 1024 | 404.6 | – | 291.8 |

## B1. Int16 evaluation (H=128 / H=512, 4 stacks), ns
| variant | H=128 | H=512 |
|---|---|---|
| W0 current `NnueEvaluator.Evaluate` | 15.2 | 39.3 |
| W1 raw kernel, managed arrays | 14.9 | 37.8 |
| W2 raw kernel, aligned native memory | 11.1 | 32.6 |
| W3 = W2 behind one interface call | 12.3 | 34.5 |

Public API and validation: 2–4 %. Alignment: 17–27 %. One interface call per evaluation: ≈1.1 ns (6–10 % at this size).

## B2. Int8 evaluation (Stockfish-like, pairwise, [sqr|clip], dense, skip, 8 stacks), ns - ladder
| step | H=256 dense | H=256 sparse | H=1024 dense | H=1024 sparse |
|---|---|---|---|---|
| V0 current evaluator | 286 | 188 | 748 | 453 |
| V1 faithful copy (stackalloc, zeroed) | 281 | 177 | 731 | 435 |
| V2 + `[SkipLocalsInit]` | 267 | 163 | 700 | 380 |
| V3 + preallocated aligned scratch | 263 | 164 | 701 | 379 |
| V4 + aligned weight blob + step table | 211 | 134 | 587 | 301 |
| V4' same, weights shifted by 8 bytes | 262 | 157 | 733 | 377 |
| V5 + aligned native accumulators | 208 | 127 | 574 | 300 |
| V6 + backend resolved at compile time | 200 | 122 | 566 | 294 |

Total: **24–35 % faster than today.** `SkipLocalsInit` = preallocated scratch (zeroing costs 5–12 %). V4 vs V4': the gain of
the blob is **alignment, not structure** - the same structure with shifted weights gains only 0–5 %.

## E. Accumulator update on a cache-cold feature matrix (2+2 random rows), ns
| F × H (matrix size) | chunk-outer, registers (today) | feature-outer | chunk-outer + prefetch |
|---|---|---|---|
| 768 × 256 (0.4 MB) | 30.2 | 38.6 | 41.5 |
| 22528 × 256 (11 MB) | 79.4 | 81.1 | 88.0 |
| 22528 × 1024 (44 MB) | 353 | 373 | 437 |

The current kernel structure is best or tied. Not tested: huge pages for the feature matrix, software prefetch of the *next* position's rows.

(Experiment D, the deliberately misaligned memory, was removed from `Nnue.Bench` when the kernels switched to aligned loads; its results above stay as the reason for the 64-byte alignment.)

## F1. Addressing a random row of the feature matrix (pinned, 64-byte-aligned managed array), ns
(historical: measured with the earlier `ref`-based kernels; the experiment was removed from `Nnue.Bench` when the kernels moved to pointers and native memory)
| F × H | pointer (`fixed` / pinned) | array indexer, bounds-checked | unchecked `ref` (`Unsafe.Add`) |
|---|---|---|---|
| 768 × 256 | 39.0 | 39.7 | 39.8 |
| 22528 × 1024 (44 MB, cache-cold) | 477 | 449 | 450 (run-to-run noise dominates, medians 515–545) |

No measurable difference: one predicted bounds check per row is invisible next to 0.5–2 KB of vector work per row.
JIT array optimisation (bounds-check elimination) matters for scalar per-element loops, which the hot path does not have
(the kernels are intrinsics working on `ref`s). Managed vs native memory is therefore a usability decision;
**alignment** is the performance decision, and managed arrays are only aligned if they are over-allocated on the pinned object heap and offset.

## F2. Cost of validity checks (same aligned memory, checked vs unchecked), ns
| operation | unchecked | checked | overhead |
|---|---|---|---|
| int16 evaluate, H=128 (stm, bucket, accumulator size) | 11.0 | 13.1 | **+19 %** |
| int16 evaluate, H=512 | 32.4 | 34.3 | +6 % |
| update 2+2 features, H=256 (range check of each index + size) | 24.9 | 28.9 | **+16 %** |
| update 2+2 features, H=1024 | 86.3 | 91.7 | +6 % |
| refresh 16 features, H=256 | 81.1 | 103.2 | **+27 %** |
| refresh 32 features, H=256 | 195 | 229 | +17 % |
| refresh 16 / 32 features, H=1024 | 355 / 911 | 374 / 921 | +5 % / +1 % |

Checks cost a roughly constant 2–20 ns, so they matter only when the operation itself is short: small accumulators (H ≤ 256) in tight loops.
I did not investigate why the refresh check costs more than 16 compares should (the loop runs in addition to the kernel's own pass over the list).

## G. The refactored library against the previous design (same process, interleaved, min of 9 rounds)
The experiments A–F above led to the current layout (aligned blob, backend-generic core, inlined checked/unchecked wrappers). The old
evaluator was built under a different namespace and run in the same process as the new one, on the same random accumulator values (different
random weights, so only the timing is comparable). The raw output of these runs was lost when the build environment was reset, only the ratios
were kept; the sandbox also moved between two different Xeons during this work, so absolute numbers are not given.

| | new, checked | new, unchecked |
|---|---|---|
| int16 evaluation, H=128, 4 stacks | ≈103 % of the old time | ≈89 % |
| int16 evaluation, H=512 | ≈84 % | ≈79 % |
| int8 Stockfish-like, H=256 / 1024, dense and sparse activations | 72–80 % | 70–77 % |

A first version of the checked wrappers (checked method calling the unchecked one, two struct copies) was 29 % *slower* than the old evaluator on the tiny
int16 case; making the wrappers inline, passing raw pointers and combining the checks into one branch brought it to parity. On the
older Cascade Lake Xeon the ladder of section B2 gave an even larger total gain (58–62 % of the old time) than on the Sapphire/Emerald-Rapids-class Xeon.


## H. Native memory, aligned operations, backends and stages (2-core Xeon VM with AVX-512 + AVX-VNNI, 9 interleaved rounds, min ns)
Run after the switch to pointers, `NativeMemory.AlignedAlloc` and `LoadAligned` / `StoreAligned`. (No comparison with the `ref` version was
made, only the absolute numbers are given; the VM is noisy, differences below about 5 % are not meaningful.)

Backends on the same model, unchecked evaluate:
| model | Sse | Avx2 | **Avx512** | AvxVnni (256 bit) |
|---|---|---|---|---|
| int16 H=128 | 26 | 19 | **18** | 19 |
| int16 H=512 | 83 | 47 | **38** | 47 |
| int8 H=256, dense activations | 572 | 326 | **214** | 227 |
| int8 H=256, sparse activations | 304 | 178 | **126** | 137 |
| int8 H=1024, dense | 1668 | 914 | **571** | 608 |
| int8 H=1024, sparse | 748 | 410 | **265** | 326 |

AVX-512 is fastest everywhere, so `SimdSupport.Best` (Avx512 > AvxVnni > Avx2) is right on this CPU. .NET 8 has no 512-bit vpdpbusd; the
512-bit path needs maddubs + madd + add (3 uops on 2 ports) per 64 MACs, which is still faster than 256-bit vpdpbusd.

Stages of an int8 Stockfish-like evaluation, H=256 (pairwise 2x128 -> 32 -> 32 -> 1), AVX-512, ns:
pairwise 15, layer 0 (256 -> 32) 76, layer 1 (64 -> 32) 22, layer 2 (128 -> 16 padded, 1 output used) 21, activations 6 + 4 (each twice),
non-zero count 5. The dense layers are port-bound (uops of vpmaddubsw / vpmaddwd / vpaddd), not load-bound.

Changes made because of these numbers:
* **`FindNonZeroChunks` with a lookup table** (mask of non-zero chunks -> index vector, no branch per non-zero chunk): sparse layer 0 incl. index
  search 69 -> 56 ns at 34 % non-zero chunks.
* **AVX-VNNI kernels with 8 accumulator chains** (4 input chunks per iteration; vpdpbusd needs about 10 chains in flight): int8 H=1024
  dense 632 -> 608 ns, sparse 337 -> 326 ns; still behind AVX-512 on this CPU, but this is the best kernel on CPUs without AVX-512.

* **Last layer as a contiguous dot product** (`Out == 1`, model file version 2): 21 -> 4.4 ns for the 128 -> 1 layer. Whole int8 evaluation
  (AVX-512, unchecked): H=256 dense 214 -> 196 ns (-8 %), H=256 sparse 126 -> 126, H=1024 dense 571 -> 551 ns (-3.5 %), H=1024 sparse
  265 -> 250 ns (-6 %). AVX-VNNI: H=1024 dense 608 -> 580 ns. The file layout changed (the last layer holds `In` weights instead of
  `In * 16`), so models saved with version 1 can no longer be loaded.

## I. Register tiles in refresh and update (scratch harness, interleaved, minimum of 7 rounds, ns per call)
One vector per pass (old) against independent register tiles (new: AVX-512 4 x 512 bit, AVX2 8 x 256 bit). Random feature rows from a pool of 4096.

| operation | F | H | AVX-512 old | AVX-512 new | AVX2 old | AVX2 new |
|---|---:|---:|---:|---:|---:|---:|
| refresh, 16 features | 768 | 256 | 88 | 68 | 157 | 96 |
| refresh, 32 features | 768 | 1024 | 758 | 592 | 1395 | 799 |
| refresh, 32 features | 22528 | 256 | 622 | 620 | 972 | 614 |
| refresh, 32 features | 22528 | 1024 | 2526 | 2556 | 2988 | 2510 |
| update 2 + 2 | 768 | 256 | 28.0 | 20.5 | 53.5 | 29.8 |
| update 2 + 2 | 768 | 1024 | 114.7 | 76.9 | 201.5 | 115.0 |
| update 2 + 2 | 22528 | 256 | 75.1 | 77.0 | 99.7 | 77.5 |
| update 2 + 2 | 22528 | 1024 | 347.8 | 352.2 | 371.9 | 353.3 |

While the feature matrix is in cache the dependency chain of one vector was the limit; once it is not (22528 x H), memory is, and AVX-512 gains
nothing (AVX2 still does: it had twice as many dependent adds). The refresh numbers came from 8-vector AVX-512 tiles in the scratch harness;
4 and 8 were within noise, so the library uses 4 (= 128 values, which also covers H = 128 without a tail).

## J. AVX512-VNNI kernel structure (scratch harness, net10.0, unchecked Evaluate, minimum of 9 rounds, ns)
The AVX512-VNNI kernels run here only through their software model (the instruction needs .NET 11), which executes the same two instructions
as the plain AVX-512 kernel; the comparison shows that the 8-accumulator structure costs nothing. The hardware `vpdpbusd zmm` replaces
`vpmaddubsw` + `vpmaddwd` + `vpaddd` by one instruction, so it should be faster; that is **not measured**.

| Stockfish-like, 8 stacks | Avx512 | Avx512VnniEmulated | AvxVnni (256-bit hardware) |
|---|---:|---:|---:|
| H=256 dense | 185 | 198 | 195 |
| H=256 sparse | 124 | 130 | 125 |
| H=1024 dense | 550 | 561 | 577 |
| H=1024 sparse | 250 | 254 | 304 |
