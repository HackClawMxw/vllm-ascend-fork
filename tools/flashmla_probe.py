# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release-machine preflight for the external FlashMLA package.

Uses synthetic cache allocations, never model/service cache. Loading the leaf
adapter via runpy avoids requiring vLLM engine initialization for package P0/P1.
No kernel is executed unless --execute is passed. Numerical acceptance requires
explicit --atol and --rtol; otherwise errors are reported without a pass claim.
Run as ``python -m tools.flashmla_probe`` from the repo root so ``tools/bisect``
does not shadow Python's standard ``bisect`` module.
"""

import argparse
import importlib
import importlib.metadata
import json
import math
import runpy
import subprocess
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--metadata-only", action="store_true", help="Run the real metadata op, then stop before attention"
    )
    parser.add_argument("--heads", type=int, choices=(8, 12, 64, 96), default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-len", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=129)
    parser.add_argument("--mask-mode", type=int, choices=(0, 3), default=3)
    parser.add_argument("--layout-kv", choices=("PA_BBND", "PA_NZ"), default="PA_BBND")
    parser.add_argument("--return-lse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", type=Path, default=Path("flashmla-probe.json"))
    args = parser.parse_args()
    if args.metadata_only and not args.execute:
        parser.error("--metadata-only requires --execute")
    if not 0 < args.batch_size < 65536 or not 0 < args.query_len <= args.kv_len:
        parser.error("require 0 < batch-size < 65536 and 0 < query-len <= kv-len")
    if (args.atol is None) != (args.rtol is None):
        parser.error("provide both --atol and --rtol, or neither")
    if args.atol is not None and (
        not math.isfinite(args.atol) or not math.isfinite(args.rtol) or args.atol < 0 or args.rtol < 0
    ):
        parser.error("tolerances must be finite and nonnegative")
    return args


def package_versions():
    versions = {}
    for name in ("torch", "torch-npu", "cann-ops-transformer", "vllm", "vllm-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "distribution metadata unavailable"
    return versions


def describe(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        storage_offset=tensor.storage_offset(),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )


def reference_attention(q, cache, table, kv_len, query_len, mask_mode, scale):
    """Float32 reference on CPU, including right-aligned causal attention."""
    outputs, lses = [], []
    for request, blocks in enumerate(table):
        kv = cache[blocks.long()].reshape(-1, cache.shape[-1])[:kv_len].float()
        query = q[request * query_len : (request + 1) * query_len].float().transpose(0, 1)
        scores = torch.matmul(query, kv.T) * scale
        if mask_mode == 3:
            allowed = torch.arange(kv_len)[None, :] <= kv_len - query_len + torch.arange(query_len)[:, None]
            scores.masked_fill_(~allowed, -torch.inf)
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1), kv[:, :512]))
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(outputs, dim=1), torch.cat(lses, dim=1)


def make_cache(logical_cache, layout, variant, device):
    """Create fresh synthetic views to test page/token strides and offset.

    NZ packing is only for this probe's generated input; it is not an adapter
    conversion of a persistent VA cache.
    """
    payload = (
        logical_cache
        if layout == "PA_BBND"
        else logical_cache.reshape(-1, 128, 1, 36, 16).permute(0, 2, 3, 1, 4).contiguous()
    )
    if variant == "contiguous":
        cache = payload.to(device)
        return cache, cache
    if layout == "PA_BBND":
        token_stride = 608 if variant == "token-strided" else 576
        page_stride = 128 * token_stride + 256
        strides = (page_stride, token_stride, 576, 1)
    else:
        page_stride = 576 * 128 + 256
        strides = (page_stride, 576 * 128, 128 * 16, 16, 1)
    backing = torch.full((payload.shape[0] * page_stride + 64,), 7, dtype=payload.dtype, device=device)
    cache = backing.as_strided(payload.shape, strides, storage_offset=32)
    cache.copy_(payload.to(device))
    return cache, backing


def error_metrics(actual, expected, args):
    actual = actual.float().cpu()
    if actual.shape != expected.shape:
        raise ValueError(f"unexpected output shape {actual.shape}, expected {expected.shape}")
    if not torch.isfinite(actual).all():
        raise ValueError("output contains non-finite values")
    error = (actual - expected).abs()
    result = {"max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
    if args.atol is not None:
        tolerance = args.atol + args.rtol * expected.abs()
        result["failed_elements"] = (error > tolerance).sum().item()
        result["within_requested_tolerance"] = result["failed_elements"] == 0
    else:
        result["within_requested_tolerance"] = None
    return result


def execute_cases(args, adapter, report):
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(0)
    pages_per_request = (args.kv_len + 127) // 128
    pages = args.batch_size * pages_per_request
    q_cpu = torch.randn(args.batch_size * args.query_len, args.heads, 576, generator=generator).to(dtype)
    cache_cpu = torch.randn(pages, 128, 1, 576, generator=generator).to(dtype)
    table_cpu = torch.arange(pages, dtype=torch.int32).flip(0).reshape(args.batch_size, pages_per_request)
    expected, expected_lse = reference_attention(
        q_cpu, cache_cpu, table_cpu, args.kv_len, args.query_len, args.mask_mode, adapter.config.softmax_scale
    )
    q = q_cpu.to(args.device)
    lengths = torch.full((args.batch_size,), args.kv_len, dtype=torch.int32, device=args.device)
    cu = torch.arange(args.batch_size + 1, dtype=torch.int32, device=args.device) * args.query_len
    used = torch.full_like(lengths, args.query_len)
    table = table_cpu.to(args.device)
    mask = (
        None
        if args.mask_mode == 0
        else torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=args.device), diagonal=1)
    )
    variants = (
        ("contiguous", "page-strided", "token-strided")
        if args.layout_kv == "PA_BBND"
        else ("contiguous", "page-strided")
    )
    if args.metadata_only:
        variants = ("contiguous",)
    for variant in variants:
        result = {"variant": variant, "status": "started"}
        report["cases"].append(result)
        cache, backing = make_cache(cache_cpu, args.layout_kv, variant, args.device)
        result["cache"] = describe(cache)
        before = backing.cpu().clone()
        print(f"[FlashMLA probe] {variant}: metadata_call", flush=True)
        metadata = adapter.build_metadata(lengths, cu, used)
        print(f"[FlashMLA probe] {variant}: metadata_return", flush=True)
        torch.npu.synchronize()
        print(f"[FlashMLA probe] {variant}: metadata_synchronized", flush=True)
        result["metadata"] = describe(metadata)
        meta_capacity = report["metadata_meta"]["shape"]
        if list(metadata.shape) != meta_capacity or metadata.dtype != torch.int32:
            raise ValueError("runtime metadata shape/dtype differs from the package Meta query")
        if args.metadata_only:
            result["status"] = "metadata_completed"
            continue
        print(f"[FlashMLA probe] {variant}: attention_call", flush=True)
        output, lse = adapter.attention(
            q,
            cache,
            block_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            seqused_q=used,
            metadata=metadata,
            attn_mask=mask,
        )
        print(f"[FlashMLA probe] {variant}: attention_return", flush=True)
        torch.npu.synchronize()
        print(f"[FlashMLA probe] {variant}: attention_synchronized", flush=True)
        result["cache_unchanged"] = torch.equal(before, backing.cpu())
        if not result["cache_unchanged"]:
            raise ValueError("attention changed read-only cache or its padding/guards")
        if output.dtype != dtype or lse.dtype != torch.float32:
            raise ValueError("unexpected output/LSE dtype")
        result["attention"] = error_metrics(output, expected, args)
        if args.return_lse:
            result["lse"] = error_metrics(lse, expected_lse, args)
        elif lse.shape != (0,):
            raise ValueError("disabled LSE must have shape (0,)")
        metrics = [result["attention"]] + ([result["lse"]] if args.return_lse else [])
        if any(item["within_requested_tolerance"] is False for item in metrics):
            raise ValueError("numerical errors exceed requested tolerances")
        result["status"] = (
            "within_requested_tolerance" if args.atol is not None else "executed_without_acceptance_threshold"
        )
        del output, lse, metadata, before, cache, backing


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "revision": revision,
        "config": vars(args) | {"output": str(args.output)},
        "packages": package_versions(),
        "cases": [],
        "status": "started",
    }
    exit_code = 0
    try:
        importlib.import_module("torch_npu")
        torch.npu.set_device(args.device)
        report["device_name"] = torch.npu.get_device_name()
        api = runpy.run_path(str(root / "vllm_ascend/attention/flashmla.py"))
        config = api["FlashMLAConfig"](args.heads, 576**-0.5, args.mask_mode, args.layout_kv, args.return_lse)
        adapter = api["FlashMLAAdapter"].load(config)
        report["schemas"] = {
            name: str(getattr(op, "_schemas", "unavailable"))
            for name, op in (("attention", adapter.attention_op), ("metadata", adapter.metadata_op))
        }
        print("[FlashMLA probe] meta_call", flush=True)
        meta = adapter.build_metadata(
            torch.empty(args.batch_size, dtype=torch.int32, device="meta"),
            torch.empty(args.batch_size + 1, dtype=torch.int32, device="meta"),
            torch.empty(args.batch_size, dtype=torch.int32, device="meta"),
        )
        print("[FlashMLA probe] meta_return", flush=True)
        report["metadata_meta"] = describe(meta)
        if meta.ndim != 1 or meta.numel() == 0 or meta.dtype != torch.int32:
            raise ValueError("package Meta did not return a nonempty int32 schedule")
        if args.execute:
            execute_cases(args, adapter, report)
        report["status"] = "completed"
        report["validation_scope"] = (
            "metadata_kernel_only"
            if args.metadata_only
            else "operator_probe_only"
            if args.execute
            else "schema_and_metadata_meta_only"
        )
        report["numerical_acceptance"] = bool(args.execute and args.atol is not None)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["cases"] and report["cases"][-1]["status"] == "started":
            report["cases"][-1]["status"] = "failed"
        exit_code = 1
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{report['status']}: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
