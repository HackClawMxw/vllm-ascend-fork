# SPDX-License-Identifier: Apache-2.0
"""Host-only wiring checks. Run directly; never import torch, vLLM or package ops."""

import ast
import copy
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).parents[3]


def source(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def extract(tree, names, namespace):
    nodes = [copy.deepcopy(node) for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, "<host-only extracted source>", "exec"), namespace)
    return SimpleNamespace(**namespace)


class HostFlashMLAContractTests(unittest.TestCase):
    def test_shared_contract_uses_documented_attributes(self):
        loaded = extract(
            source("vllm_ascend/attention/flash_mla.py"),
            {"FlashMLAContract"},
            {"__name__": __name__, "dataclass": dataclass, "replace": replace, "Any": Any},
        )
        for heads in (64, 96):
            contract = loaded.FlashMLAContract(num_heads_q=heads, mask_mode=3)
            cu, used, cache, table, schedule, mask = [object() for _ in range(6)]
            meta = contract.metadata_kwargs(cu, used)
            main = contract.attention_kwargs(
                block_table=table,
                cache_seqlens=cache,
                cu_seqlens_q=cu,
                seqused_q=used,
                attn_mask=mask,
                metadata=schedule,
            )
            for name in ("max_seqlen_q", "max_seqlen_kv"):
                self.assertEqual(meta[name], -1)
                self.assertEqual(main[name], -1)
            self.assertIs(main["metadata"], schedule)
            self.assertIs(meta["cu_seqlens_q"], main["cu_seqlens_q"])
            self.assertIs(meta["seqused_q"], main["seqused_q"])
            self.assertEqual(main["layout_kv"], "PA_BBND")
            self.assertEqual(main["layout_out"], "NTD")
        for heads in (0, 16, 32, 128):
            with self.assertRaises(ValueError):
                loaded.FlashMLAContract(num_heads_q=heads, mask_mode=3)
        with self.assertRaises(ValueError):
            loaded.FlashMLAContract(num_heads_q=64, mask_mode=3, max_seqlen_q=128)

    def test_public_wrappers_and_no_native_fallback(self):
        tree = source("vllm_ascend/attention/flash_mla.py")
        loader = ast.unparse(function(tree, "_get_flash_mla_ops"))
        self.assertIn("cann_ops_transformer.ops", loader)
        self.assertNotIn("torch.ops", loader)
        self.assertNotIn("_C_ascend", loader)

    def test_metadata_has_no_device_scalar_readback(self):
        tree = source("vllm_ascend/attention/flash_mla.py")
        build = ast.unparse(function(tree, "build_flash_mla_metadata"))
        self.assertNotIn(".item(", build)
        self.assertNotIn(".cpu(", build)
        self.assertNotIn(".tolist(", build)
        self.assertIn("block_size != 128", build)
        self.assertIn("kv_capacity=table.shape[1] * block_size", build)
        self.assertIn("flash.schedule.copy_(schedule)", build)
        self.assertIn("meta=True", build)
        self.assertIn("DeviceMetadataTask(", build)
        self.assertIn("buffers = {} if is_eager_prefill else builder._flash_buffers", build)

    def test_cache_and_live_mask_stay_separate(self):
        tree = source("vllm_ascend/attention/flash_mla.py")
        build = ast.unparse(function(tree, "build_flash_mla_metadata"))
        self.assertIn("flash.token_live.copy_(flash.live_boundaries.cumsum(0)[:tokens] > 0)", build)
        self.assertIn("flash.slots.masked_fill_(~flash.token_live, -1)", build)
        forward = function(source("vllm_ascend/attention/mla_v1.py"), "_forward_flash")
        main = next(
            n for n in ast.walk(forward) if isinstance(n, ast.Call) and ast.unparse(n.func) == "flash_mla_with_kvcache"
        )
        self.assertEqual([ast.unparse(n) for n in main.args], ["flash.query", "kv_cache"])
        self.assertNotIn("kv_cache.contiguous()", ast.unparse(forward))

    def test_graph_is_not_configuration_blocked(self):
        guard = ast.unparse(function(source("vllm_ascend/platform.py"), "_validate_flash_mla_config"))
        self.assertNotIn("enforce_eager", guard)
        update = function(source("vllm_ascend/attention/mla_v1.py"), "update_graph_params")
        self.assertEqual(ast.unparse(update.body[0].test), "envs.VLLM_ASCEND_ENABLE_FLASH_MLA")
        self.assertIsInstance(update.body[0].body[0], ast.Return)

    def test_execute_and_capture_own_metadata_context(self):
        for path, name in (
            ("vllm_ascend/worker/v2/model_runner.py", "execute_model"),
            ("vllm_ascend/worker/v2/aclgraph_utils.py", "capture"),
        ):
            method = function(source(path), name)
            contexts = [
                ast.unparse(item.context_expr) for n in ast.walk(method) if isinstance(n, ast.With) for item in n.items
            ]
            self.assertIn(
                "device_metadata_context(self.device_metadata_executor)"
                if name == "execute_model"
                else "device_metadata_context(self.model_runner.device_metadata_executor)",
                contexts,
            )

    def test_context_release_occurs_after_consumer_and_on_error(self):
        current = ContextVar("host_test_executor", default=None)
        loaded = extract(
            source("vllm_ascend/worker/v2/attn_utils.py"),
            {"device_metadata_context"},
            {"contextmanager": contextmanager, "_device_metadata_executor": current},
        )
        log = []

        class Executor:
            submission_in_flight = True

            def release(self):
                log.append("release")
                self.submission_in_flight = False

        executor = Executor()
        with self.assertRaisesRegex(ValueError, "consumer"), loaded.device_metadata_context(executor):
            with loaded.device_metadata_context(executor):
                self.assertIs(current.get(), executor)
            self.assertTrue(executor.submission_in_flight)
            log.append("consumer")
            raise ValueError("consumer")
        self.assertEqual(log, ["consumer", "release"])
        self.assertIsNone(current.get())

    def test_mrv2_submits_and_waits_before_returning_to_model(self):
        tree = source("vllm_ascend/worker/v2/attn_utils.py")
        build = function(tree, "build_attn_metadata")
        tail = ast.unparse(ast.Module(body=build.body[-2:], type_ignores=[]))
        self.assertIn("assert not torch.npu.is_current_stream_capturing()", tail)
        self.assertLess(tail.index("executor.submit"), tail.index("executor.wait"))
        self.assertLess(tail.index("executor.wait"), tail.index("return attn_metadata"))

    def test_fast_postprocess_guards(self):
        forward = function(source("vllm_ascend/attention/mla_v1.py"), "_forward_flash")
        predicate = next(n.test for n in forward.body if isinstance(n, ast.If) and "torch.mm" in ast.unparse(n))
        compiled = compile(ast.Expression(predicate), "<postprocess guard>", "eval")

        class Unquantized:
            pass

        for field in ("eligible", "reduce_results", "bias", "custom_op", "input_is_parallel", "quant_method"):
            linear = SimpleNamespace(
                input_is_parallel=True,
                reduce_results=False,
                bias=None,
                custom_op=None,
                quant_method=Unquantized(),
                weight=SimpleNamespace(dtype="bf16"),
            )
            if field != "eligible":
                setattr(linear, field, False if field == "input_is_parallel" else object())
            allowed = eval(
                compiled,
                {"torch": SimpleNamespace(bfloat16="bf16"), "UnquantizedLinearMethod": Unquantized},
                {
                    "self": SimpleNamespace(use_output_gate=True, o_proj=linear),
                    "flash": SimpleNamespace(is_prefill=False),
                    "projected": SimpleNamespace(dtype="bf16"),
                    "output": SimpleNamespace(dtype="bf16", shape=(4, 64)),
                    "num_tokens": 4,
                },
            )
            self.assertEqual(bool(allowed), field == "eligible")


if __name__ == "__main__":
    unittest.main(verbosity=2)
