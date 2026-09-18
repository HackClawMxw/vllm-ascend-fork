# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate, flash_attention_output


@torch.inference_mode()
def test_flash_attention_gate_bf16_rounding():
    # Reference #1's exhaustive finite BF16 sigmoid-rounding check.
    gates = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gates = gates[torch.isfinite(gates)].reshape(-1, 256).to("npu")
    projected = torch.linspace(0.1, 3.0, gates.numel(), dtype=torch.float32).reshape_as(gates).to(gates)
    live = torch.ones(gates.shape[0], dtype=torch.bool, device="npu")
    expected = projected * torch.sigmoid(gates)
    actual = flash_attention_gate(projected, gates, live)
    assert actual.data_ptr() == projected.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["all-live", "alternating", "all-inactive"])
@torch.inference_mode()
def test_flash_attention_eager_gate_and_output_mask(mode):
    torch.manual_seed(20260918)
    live = torch.full((8,), mode != "all-inactive", dtype=torch.bool, device="npu")
    if mode == "alternating":
        live[1::2] = False
    projected = torch.randn(8, 1536, dtype=torch.bfloat16, device="npu")
    gates = torch.randn_like(projected)
    projected.masked_fill_(~live[:, None], float("nan"))
    gates.masked_fill_(~live[:, None], float("inf"))
    weight = torch.randn(64, 1536, dtype=torch.bfloat16, device="npu") * 0.01
    expected = F.linear(projected * torch.sigmoid(gates), weight)
    expected.masked_fill_(~live[:, None], 0)
    output = torch.full_like(expected, float("nan"))
    flash_attention_gate(projected, gates, live)
    torch.mm(projected, weight.t(), out=output)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    # Generic writeback also clears rows beyond the result's token dimension.
    source = expected.clone()
    source.masked_fill_(~live[:, None], float("nan"))
    padded = torch.full((11, 64), float("nan"), dtype=torch.bfloat16, device="npu")
    assert flash_attention_output(source, live, padded) is padded
    torch.testing.assert_close(padded[:8], expected, atol=0, rtol=0)
    assert torch.count_nonzero(padded[8:]) == 0
    assert torch.isfinite(padded).all()
