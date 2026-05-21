# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TQFullAttentionSpec for Ascend NPU TurboQuant.

Ported from vllm's vllm/v1/kv_cache_interface.py to keep vllm-ascend-fork
self-contained.  Inherits from the base vllm's FullAttentionSpec (which is
general infrastructure, not turboquant-specific).
"""

from dataclasses import dataclass
from dataclasses import replace
from typing import Self

from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec with TQ-aware page size.

    Overrides real_page_size_bytes to use TQ slot bytes instead of the raw
    head_size * dtype formula.
    """

    tq_slot_size: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = super().merge(specs)
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs), (
            "All TQ layers in the same KV cache group must use the same "
            "tq_slot_size."
        )
        return replace(merged, tq_slot_size=specs[0].tq_slot_size)
