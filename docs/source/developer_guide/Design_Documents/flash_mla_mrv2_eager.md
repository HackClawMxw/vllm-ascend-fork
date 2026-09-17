# A5 MRV2 external FlashMLA integration

## Status and references

This is an experimental, disabled-by-default route selected by
`VLLM_ASCEND_ENABLE_FLASH_MLA=1`.
Use `--enforce-eager` for the initial smoke; the implementation uses one shared
eager/graph metadata lifecycle, not an eager-only backend.

The base is PR #16456 at `bfd8ff9aa87342e42b1160416506a422ca69bdc1`,
with vLLM `84030bbe3d74d99bad477a3d2e37a973ccd8865c`.
Reference #1 is PR #16468 at `48c8a870d6b376b1ff854d636a2babe543cb395b`.
Its postprocessing and MRV2 metadata ownership pattern are retained.
DCP query replication follows maoxx241/vllm-ascend PR #5 at
`63f12e209287c444d08f16ae8aaf9bfd6c561632`, also present in that reference.
This is not a wholesale merge of its allocator, GQA, DSpark or MLAPO work.

The supplied September 16 Torch API specification for
`flash_mla_with_kvcache` governs the external boundary. The package is the
delivery of the same intended operator, not another algorithm. This candidate
has not been executed on NPU; eager, graph and full-model acceptance are pending.

## External contract

- A5, MRV2, dense MLA, BF16, 64 projected RoPE channels. K3 hybrid MLA/KDA
  models are allowed: only MLA layers use this adapter; KDA is unchanged.
  NoPE layers preserve the 64 channels without rotation; RoPE layers rotate
  using real positions.
- Actual operator Q head count 64 or 96, one KV head, QK576 and latent V512.
  With DCP disabled this is the local Q head count, not the model-global count.
- TND Q, PA_BBND cache `[P,128,1,576]`, NTD output `[H,T,512]`.
  The model's post-V-up value width is independent of the latent width.
- Both stages pass `max_seqlen_q=-1` and `max_seqlen_kv=-1`.
  The operator derives lengths from device CU/used-Q/cache-length inputs.
  Internal token and KV capacities must not leak into these attributes.
- Causal mode 3 uses the existing 2048-by-2048 int8 upper-triangular mask
  (one means masked); noncausal mode 0 has no mask.
- Load both public wrappers from `cann_ops_transformer.ops`.
  Missing entry points or metadata Meta support fail explicitly; there is no
  native fallback, invented schedule or hardcoded device-core count.
- PCP=1, DCP=1 or DCP=TP with a replicated Q projection; no KV-layer parallelism,
  speculative decoding, PD/KV transfer,
  sparse/compressed attention or quantized KV.

Only the kernel page is fixed to 128. Manager/interleave size and the physical
page stride remain owned by #16456. Q heads are never padded to evade the API.

## Data and graph lifecycle

The stable-buffer implementation follows reference #1's ownership model:

1. Keep query, schedule, CU, used-Q, cache lengths, block table, slots, live
   mask and positions in buffers keyed by requests, token capacity, table
   width and causality. Decode/graph buckets reuse addresses; eager prefill
   buffers belong to the current batch instead of accumulating in the builder.
2. Size schedule using the external metadata operator's Meta implementation.
3. In a graph-external device metadata task, copy current MRV2 lengths and
   page table, form used-Q and live intervals, disable inactive write slots,
   sanitize padded positions, call the real metadata operator and copy its
   result into the stable schedule. Contents are recomputed every batch.
4. MRV2's executor waits for input readiness and the previous consumer's
   reuse fence. Metadata construction submits tasks and queues ordinary stream
   waits outside model capture/replay. Execution and capture contexts release
   ownership only after the consumer has been queued.
5. The layer writes projected/rotated Q into the stable query buffer, scatters
   cKV/kPE into the two slices of the original fused cache, and invokes the
   main operator. Legacy FIA graph-task updates are bypassed on this route.

Internal query capacity is the input token count; KV capacity is table width
times the selected kernel block size. These are shape/ownership information,
not the external API's maximum-length attributes. The shared refresh path does
not call `.item()`, `.cpu()` or calculate Python maxima from device lengths.
Host request classification still follows reference #1's existing builder.

A final zero-used request owns padding. Query visibility comes from CU/used-Q
intervals, independently of whether a query owns a writable cache slot.
Zero-token batches bypass attention. All-inactive nonempty batches keep the
same scheduling path and mask their writes/output without a host data branch.
Device value correctness (valid CU bounds, physical pages and slots, KV lengths
within table capacity) remains MRV2's producer contract, not proof from static
shape checks.

The first cache dimension may be noncontiguous and have a nonzero storage
offset; inner dimensions are dense. There is no full-cache concatenation,
contiguous copy or repack. Allocation, strided view construction, zeroing and
COW are unchanged.

## DCP and TP

For K3, enable `VLLM_DCP_Q_REPLICATE=1` together with
`VLLM_ASCEND_ENABLE_FLASH_MLA=1` and equal TP/DCP sizes (for example
`--tensor-parallel-size 4 --decode-context-parallel-size 4`). Keep
`--enforce-eager` for the first runtime check. No topology is changed implicitly.
With the switch disabled, the K3 projection and legacy attention stay unchanged.

The Q projection is replaced before checkpoint loading with
`DCPGroupColumnParallelLinear`. For 96 global heads and TP=DCP=4, each rank
produces 96 actual Q heads, while V-up/gate/O retain their 24-head TP shards.
K-up absorption weights are gathered once after weight loading; reloading
updates their existing allocation. A missing replicated projection or a group
head count other than 64/96 fails rather than padding or mislabeling Q.

For causal attention the layer computes two disjoint parts:

| Part | KV lengths and storage | Q heads | Mask | Output handling |
| --- | --- | --- | --- | --- |
| History | Global `seq_lens - used_q`, partitioned by DCP interleave; original BBND cache | Group heads | 0 | Exchange history shards into TP-local head ownership |
| Current | `used_q`; temporary replicated BBND pages written by this layer | Group heads | 3 | Slice output/LSE into TP-local heads **after** the operator |

Both external calls return NTD latent output and FP32 NT LSE. Unlike the native
current-block path in reference #1, neither external call receives 12/24 local heads.
The merge counts every history shard and exactly one current chunk, then passes
local NTD latent output to V-up/gate/O. The existing FP32 DCP all-to-all is reused;
the LSE merge is deliberately unfused PyTorch, not #1's SFA Triton optimization.
Its graph compatibility and cost still require real execution evidence.
Known-empty history shards are masked from device lengths, without relying on
an undocumented native LSE sentinel. Noncausal DCP reads the full local cache
with mask 0 and has no separate current chunk.

Both schedules, current page table/slots and history lengths refresh in one
graph-external task/frontier. Decode buckets preserve both schedule addresses
and the current scratch cache address. MRV2 submit/wait and the post-consumer
reuse fence cover them together. The DCP subclass bypasses the old FIA graph
update hook for the same reason as the base class; it does not skip metadata
refresh. No additional runner stream or PD scheduling changes are introduced.

## Output processing

After NTD attention output, V-up maps latent512 to the model value width.
The no-merge tail and its two Triton helpers match the pinned reference:
decode gating can precede a direct bias-free unquantized GEMM only when the
parallel-input, reduction, custom-op, dtype and output-shape guards permit it.
Otherwise the normal O-projection wrapper runs and the final writeback masks
inactive/padded rows. TP/SP communication is not bypassed by relaxing guards.

## Validation

Run host-only wiring checks without importing torch:

```bash
python tests/ut/attention/test_flash_mla_source_contract.py
```

Run the following only on the prepared A5 environment with the delivered package:

```bash
pytest -q tests/ut/attention/test_flash_mla.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_flash_mla_with_kvcache.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_flash_attention_output.py
torchrun --standalone --nproc-per-node=4 -m pytest -xq \
  tests/e2e/nightly/single_node/ops/multicard_ops_a5/test_flash_mla_dcp.py
```

Coverage includes 64/96 Q heads, 128-token pages, stride/offset/page-gap guards,
decode/prefill/mixed/noncausal, cross-page and multistep requests, fused/separate
Q/KV production, V128, gating/bias variants and padding. Scatter has an
independent CPU cache oracle. The synthetic-layer numerical oracle consumes
the real preprocessing outputs; it does not independently qualify model
weight loading or distributed linear layers.

A real synthetic-layer graph test refreshes lengths, page mappings and inactive
requests in the same bucket, verifies stable pointers and compares replay
against eager. It is not a full MRV2 runner/model graph test.

The multi-rank synthetic-layer test exercises real external calls and HCCL with
64/96 group heads, interleave 1/16, empty history, mixed requests, page crossing,
inactive rows, strided writes and stable-address decode replay. Its oracle is
unpartitioned CPU attention. It does not validate checkpoint weight loading,
the actual #16456 allocator, or the full hybrid K3 MRV2 runner.

The acceptance tolerances remain `atol=rtol=0.02`; cache writes and protected
storage must match exactly. Eager single-/multi-rank model smoke, real MRV2
capture/replay, package delivery checks and #16456 COW/zeroing regressions remain
required. Previous MRV1 eager results do not qualify this MRV2 candidate.
