# SPDX-License-Identifier: Apache-2.0
"""Host-only wiring checks. Run directly; never import torch, vLLM or package ops."""

import ast
import copy
import sys
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

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
        for heads in (0, 12, 16, 24, 32, 128):
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
        calls = [
            n for n in ast.walk(forward) if isinstance(n, ast.Call) and ast.unparse(n.func) == "flash_mla_with_kvcache"
        ]
        self.assertEqual(len(calls), 1)
        main = calls[0]
        self.assertEqual([ast.unparse(n) for n in main.args], ["flash.query", "kv_cache"])
        self.assertNotIn("kv_cache.contiguous()", ast.unparse(forward))
        for removed in ("_local_view", "exchange_flash_mla_output", "combine_flash_mla_output", "flash.current"):
            self.assertNotIn(removed, ast.unparse(forward))

    def test_replicated_constructor_selects_parent_controls_before_wrapper(self):
        calls, wrappers, upstream = [], [], []

        def linear(input_size, output_size, **kwargs):
            result = SimpleNamespace(input_size=input_size, output_size=output_size, **kwargs)
            calls.append(result)
            return result

        def wrapper(*args):
            wrappers.append(args)
            return SimpleNamespace(mla_attn=SimpleNamespace(impl=SimpleNamespace()))

        class Base:
            def __init__(self, **kwargs):
                upstream.append(kwargs)
                self.mla_attn = SimpleNamespace(mla_attn=SimpleNamespace(impl=SimpleNamespace()))

        flags = SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True)
        loaded = extract(
            source("vllm_ascend/models/kimi_k3.py"),
            {"AscendKimiMLAAttention"},
            {
                "UpstreamKimiMLAAttention": Base,
                "copy": copy.copy,
                "ascend_envs": flags,
                "nn": SimpleNamespace(Module=object),
                "ColumnParallelLinear": linear,
                "MergedColumnParallelLinear": linear,
                "ReplicatedLinear": linear,
                "RowParallelLinear": linear,
                "RMSNorm": lambda *a, **kw: object(),
                "MLAModules": SimpleNamespace,
                "AscendKimiK3MultiHeadLatentAttention": wrapper,
            },
        )
        for q_lora in (None, 64):
            for gate in (False, True):
                calls.clear()
                wrappers.clear()
                config = SimpleNamespace(rms_norm_eps=1e-6)
                kwargs = dict(
                    config=config,
                    hidden_size=64,
                    num_heads=96,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    q_lora_rank=q_lora,
                    kv_lora_rank=512,
                    use_output_gate=gate,
                    use_rope=False,
                    prefix="model.layers.1.self_attn",
                )
                model = loaded.AscendKimiMLAAttention(**kwargs)
                self.assertEqual(len(wrappers), 1)
                self.assertEqual((model.num_local_heads, wrappers[0][1]), (96, 96))
                self.assertTrue(model.mla_tp_replicated)
                self.assertTrue(all(item.disable_tp for item in calls))
                self.assertFalse(model.o_proj.reduce_results)
                self.assertEqual(model.o_proj.input_size, 96 * 128)
                self.assertEqual(model.kv_b_proj.output_size, 96 * (128 + 128))
                modules = wrappers[0][8]
                for name in ("kv_b_proj", "o_proj", "g_proj"):
                    self.assertIs(getattr(modules, name), getattr(model, name, None))
                self.assertEqual(model.use_output_gate, gate)
                self.assertFalse(hasattr(config, "mla_use_output_gate"))
                self.assertEqual(upstream, [])
        flags.VLLM_ASCEND_ENABLE_FLASH_MLA = False
        disabled = loaded.AscendKimiMLAAttention(**kwargs)
        self.assertEqual(len(upstream), 1)
        self.assertFalse(disabled.mla_tp_replicated)

    def test_sp_replicated_output_is_sharded_not_summed(self):
        calls = []
        tree = source("vllm_ascend/models/kimi_k3.py")
        method = function(tree, "_finish_attention_output")
        loaded = extract(
            ast.Module(body=[method], type_ignores=[]),
            {method.name},
            {
                "sp_shard": lambda x: calls.append("shard") or x,
                "sp_reduce_scatter": lambda x: calls.append("sum") or x,
            },
        )
        value = object()
        for sp, replicated, expected in (
            (False, True, []),
            (False, False, []),
            (True, True, ["shard"]),
            (True, False, ["sum"]),
        ):
            calls.clear()
            layer = SimpleNamespace(use_sequence_parallel=sp, self_attn=SimpleNamespace(mla_tp_replicated=replicated))
            self.assertIs(loaded._finish_attention_output(layer, value), value)
            self.assertEqual(calls, expected)
        residual = ast.unparse(function(tree, "forward_attn_residual"))
        self.assertIn("hidden_states = self._finish_attention_output(hidden_states)", residual)

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

    def test_hybrid_mla_config_and_disabled_switch(self):
        envs = SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True)
        loaded = extract(
            source("vllm_ascend/platform.py"),
            {"_validate_flash_mla_config"},
            {
                "envs": envs,
                "torch": SimpleNamespace(bfloat16="bf16"),
                "model_uses_sfa_sparse": lambda _: False,
                "KVPPConfig": SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=1)),
            },
        )
        config = SimpleNamespace(
            use_v2_model_runner=True,
            model_config=SimpleNamespace(use_mla=True, is_hybrid=True, dtype="bf16"),
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=1, decode_context_parallel_size=1, tensor_parallel_size=4
            ),
            cache_config=SimpleNamespace(cache_dtype="auto"),
            speculative_config=None,
            kv_transfer_config=None,
        )
        # Stub only the device import; execute the real configuration guard.
        with patch.dict(sys.modules, {"vllm_ascend.device.device_config": SimpleNamespace(is_950=lambda: True)}):
            loaded._validate_flash_mla_config(config)
            config.model_config.use_mla = False
            with self.assertRaisesRegex(ValueError, "model must use MLA"):
                loaded._validate_flash_mla_config(config)
            config.model_config.use_mla = True
            config.parallel_config.decode_context_parallel_size = 2
            with self.assertRaisesRegex(ValueError, "DCP size must be 1"):
                loaded._validate_flash_mla_config(config)
            config.parallel_config.decode_context_parallel_size = 4
            with self.assertRaisesRegex(ValueError, "DCP size must be 1"):
                loaded._validate_flash_mla_config(config)
        envs.VLLM_ASCEND_ENABLE_FLASH_MLA = False
        loaded._validate_flash_mla_config(None)

    def test_k3_layer_dispatch_keeps_kda_separate_from_nope_mla(self):
        cls = next(
            n
            for n in source("vllm_ascend/models/kimi_k3.py").body
            if getattr(n, "name", None) == "AscendKimiDecoderLayer"
        )
        branch = next(
            n
            for n in ast.walk(cls)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "config.is_kda_layer(layer_idx)"
        )
        kda = [n for statement in branch.body for n in ast.walk(statement) if isinstance(n, ast.Call)]
        self.assertEqual([ast.unparse(n.func) for n in kda], ["AscendKimiK3DeltaAttention"])
        mla = next(
            n
            for statement in branch.orelse
            for n in ast.walk(statement)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "AscendKimiMLAAttention"
        )
        self.assertIs(next(kw.value.value for kw in mla.keywords if kw.arg == "use_rope"), False)

    def test_flash_rope_is_optional_and_keeps_the_64_channels(self):
        forward = function(source("vllm_ascend/attention/mla_v1.py"), "_forward_flash")
        # Execute the actual rotation/concatenation slice, without importing torch.
        start = next(
            i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "c_kv"
        )
        end = next(
            i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "query"
        )
        code = compile(ast.Module(body=forward.body[start + 1 : end + 1], type_ignores=[]), "<rope slice>", "exec")
        q_abs, q_pe, k_pe, positions = object(), object(), object(), object()
        rotated_q, rotated_k = object(), object()
        calls = []

        def get_cos_sin(actual_positions, use_cache):
            self.assertIs(actual_positions, positions)
            self.assertFalse(use_cache)
            calls.append("positions")
            return "cos", "sin"

        def rotate(tensor, cos, sin):
            self.assertEqual((cos, sin), ("cos", "sin"))
            calls.append(tensor)
            return rotated_q if tensor is q_pe else rotated_k

        for use_rope in (False, True):
            with self.subTest(use_rope=use_rope):
                calls.clear()
                scope = {
                    "self": SimpleNamespace(use_mla_rope=use_rope, rope_single=rotate),
                    "flash": SimpleNamespace(positions=positions),
                    "get_cos_and_sin_mla": get_cos_sin,
                    "torch": SimpleNamespace(cat=lambda values, dim: (values, dim)),
                    "q_nope": q_abs,
                    "q_pe": q_pe,
                    "k_pe": k_pe,
                }
                exec(code, scope)
                self.assertEqual(scope["query"], ((q_abs, rotated_q if use_rope else q_pe), -1))
                self.assertIs(scope["k_pe"], rotated_k if use_rope else k_pe)
                self.assertEqual(calls, ["positions", q_pe, k_pe] if use_rope else [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
