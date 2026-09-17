# A5 MRV2 external FlashMLA integration

## Status and references

This is an experimental, disabled-by-default route selected by
`VLLM_ASCEND_ENABLE_FLASH_MLA=1`.
Use `--enforce-eager` for the initial smoke; the implementation uses one shared
eager/graph metadata lifecycle, not an eager-only backend.

The base is PR #16456 at `bfd8ff9aa87342e42b1160416506a422ca69bdc1`,
with vLLM `84030bbe3d74d99bad477a3d2e37a973ccd8865c`.
Reference #1 is PR #16468 at `48c8a870d6b376b1ff854d636a2babe543cb395b`.
Its no-merge postprocessing and MRV2 metadata ownership pattern are retained.
This is not a wholesale merge of its allocator, GQA, DCP, DSpark or MLAPO work.

This revision supersedes the replicated-Q DCP experiment at `69b44e87`, which
is preserved in Git history and `backup/pr2-dcp-69b44e87`. The single-call
integration from `6f1f9c7d` is retained, together with the hybrid/NoPE fixes.
There is no history/current split, temporary current cache, DCP result exchange
or LSE merge in the selected route.

The supplied September 16 Torch API specification for
`flash_mla_with_kvcache` governs the external boundary. The package is the
delivery of the same intended operator, not another algorithm. This candidate
has not been executed on NPU; eager, graph and full-model acceptance are pending.

## External contract

- A5, MRV2, dense MLA including K3 hybrid MLA/KDA, BF16. RoPE follows the
  layer configuration; NoPE retains the unrotated 64 projected channels.
- Actual operator Q head count 64 or 96, one KV head, QK576 and latent V512.
  For K3, each rank really computes the complete model head set; the metadata
  head count is not a substitute for a smaller tensor or padded heads.
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
- PCP/DCP=1, no KV-layer parallelism, speculative decoding, PD/KV transfer,
  sparse/compressed attention or quantized KV.

Only the kernel page is fixed to 128. Manager/interleave size and the physical
page stride remain owned by #16456. Q heads are never padded to evade the API.

## K3 MLA replication and TP boundaries

`VLLM_ASCEND_ENABLE_FLASH_MLA=1` selects full MLA replication in the K3 adapter;
there is no second switch. Do not enable DCP or `VLLM_DCP_Q_REPLICATE` for this
route. Global TP still controls KDA, MLP/MoE, embeddings and the language head.
The switch-off path uses the original upstream Kimi MLA construction.

| Component, example TP=4 / model heads=96 | Selected behavior |
| --- | --- |
| Q/Q-B, KV-B and gate projections | `ColumnParallelLinear(disable_tp=True)`, full weights |
| K-up and V-up absorption | All 96 heads from the full KV-B weights, no gather |
| FlashMLA metadata and main operator | One schedule and one call, real 96 heads |
| O projection | Full input, `RowParallelLinear(disable_tp=True, reduce_results=False)` |
| KDA projections and state | Original TP partitioning and communication |

`DCPGroupColumnParallelLinear` is not used: with DCP=1 it would still shard
across TP. Its parent already supports disabling TP for an individual module.
Full projection shapes and the wrapper's full head count are established before
checkpoint loading. Module names, loaders, quantization configuration and model
dimensions are preserved; no new quantization support is implied. The wrapper
is registered once, not constructed with local heads and patched afterward.

Under K3 sequence parallelism, attention inputs are still all-gathered by token.
After replicated MLA, the complete O output is locally token-sharded with
`sp_shard`, including padding. It must not be reduce-scattered, which would sum
identical outputs and multiply values by TP. KDA retains reduce-scatter. With
SP disabled, MLA returns the full output directly without attention reduction.

This removes MLA O reduction and DCP communication, not all model collectives.
Each rank duplicates MLA weights, computation and full historical KV. Compared
with DCP=4, full-history KV per rank is roughly four times the token-sharded
quantity before allocator/padding effects. This is a correctness-first baseline,
not a performance or memory improvement claim.

## Data and graph lifecycle

The stable-buffer implementation follows reference #1's non-DCP path:

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

## Output processing

After NTD attention output, V-up maps latent512 to the model value width.
The no-merge tail and its two Triton helpers match the pinned reference:
decode gating can precede a direct bias-free unquantized GEMM only when the
parallel-input, reduction, custom-op, dtype and output-shape guards permit it.
Otherwise the normal O-projection wrapper runs and the final writeback masks
inactive/padded rows. TP/SP communication is not bypassed by relaxing guards.
The intentional K3 replication/SP behavior is implemented at construction and
the decoder boundary above, not by weakening this fast-path predicate.

## Validation

Run host-only wiring checks without importing torch:

```bash
python tests/ut/attention/test_flash_mla_source_contract.py
```

Run the following only on the prepared A5 environment with the delivered package:

```bash
pytest -q tests/ut/attention/test_flash_mla.py
pytest -q tests/ut/models/test_kimi_k3_adapter.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_flash_mla_with_kvcache.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_flash_attention_output.py
torchrun --standalone --nproc-per-node=4 -m pytest -xq \
  tests/e2e/nightly/single_node/ops/multicard_ops_a5/test_flash_mla_replicated.py
```

Coverage includes 64/96 Q heads, 128-token pages, stride/offset/page-gap guards,
decode/prefill/mixed/noncausal, cross-page and multistep requests, fused/separate
Q/KV production, V128, gating/bias variants and padding. Scatter has an
independent CPU cache oracle. The synthetic-layer numerical oracle consumes
the real preprocessing outputs; it does not independently qualify model
weight loading or distributed linear layers.

The new model UTs check full projection shapes and real parameter loaders, and
SP output values/padding without repeated summation. Backend registration is
isolated in those UTs. Host-only AST/sentinel checks are not tensor execution.

The four-rank entry exercises the production replicated constructor with real
linear classes/loaders, a TP group, external schedule/scatter/attention and full
V-up/gate/O processing. Its unpartitioned CPU oracle computes preprocessing from
the fixture weights. It checks rank agreement, page crossing, mixed/multistep
requests, padding, protected storage and the no-collective MLA boundary. Backend
registration and post-load absorption are isolated; actual K3 checkpoint loading,
full wrapper post-load processing, hybrid execution and MRV2 still need a model
smoke. The entry is added, not claimed as executed.

A real synthetic-layer graph test refreshes lengths, page mappings and inactive
requests in the same bucket, verifies stable pointers and compares replay
against eager. It is not a full MRV2 runner/model graph test.

The acceptance tolerances remain `atol=rtol=0.02`; cache writes and protected
storage must match exactly. Eager single-/multi-rank model smoke, real MRV2
capture/replay, package delivery checks and #16456 COW/zeroing regressions remain
required. Previous MRV1 eager results do not qualify this MRV2 candidate.

Eager is the first acceptance milestone. Existing graph lifecycle support is
preserved; graph acceptance is tracked separately and remains unverified. Start
the approved K3 launch command with the existing TP/EP settings, PCP/DCP=1,
`VLLM_ASCEND_ENABLE_FLASH_MLA=1` and `--enforce-eager`. Validate against the same
checkpoint/topology with the external switch off, and record memory feasibility.
Do not infer a full-model pass from the synthetic layer tests or deploy without
separate confirmation of the remote code path and resources.
