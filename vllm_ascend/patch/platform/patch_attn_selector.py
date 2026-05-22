# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vllm's get_attn_backend to accept turboquant cache dtypes.

vllm's selector.py asserts kv_cache_dtype must be in get_args(CacheDType).
The deployed vllm base does not include turboquant in CacheDType, so this
assertion blocks turboquant activation at model initialization time.

This patch bypasses the assertion for turboquant dtypes by duplicating the
post-assertion logic (config creation + backend lookup) directly, then
delegating to the original function for all other dtypes.
"""

from typing import cast

import vllm.v1.attention.selector as _selector
from vllm.config.cache import CacheDType

_orig_get_attn_backend = _selector.get_attn_backend
_TQ_PREFIX = "turboquant_"


def _patched_get_attn_backend(
    head_size,
    dtype,
    kv_cache_dtype=None,
    use_mla=False,
    has_sink=False,
    use_sparse=False,
    use_mm_prefix=False,
    use_per_head_quant_scales=False,
    attn_type=None,
    num_heads=None,
):
    is_tq = (
        kv_cache_dtype is not None
        and isinstance(kv_cache_dtype, str)
        and kv_cache_dtype.startswith(_TQ_PREFIX)
    )

    if is_tq:
        # Skip CacheDType Literal assertion and run remaining logic directly.
        from vllm.config import get_current_vllm_config
        from vllm.v1.attention.backend import AttentionType

        import vllm.envs as envs

        vllm_config = get_current_vllm_config()
        cache_config = vllm_config.cache_config
        block_size = (
            cache_config.block_size
            if cache_config is not None
            and cache_config.user_specified_block_size
            else None
        )

        speculative_config = vllm_config.speculative_config
        use_non_causal = (
            speculative_config is not None
            and speculative_config.method == "dflash"
        )

        attn_selector_config = _selector.AttentionSelectorConfig(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            use_per_head_quant_scales=use_per_head_quant_scales,
            attn_type=attn_type or AttentionType.DECODER,
            use_non_causal=use_non_causal,
            use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
        )

        return _selector._cached_get_attn_backend(
            backend=vllm_config.attention_config.backend,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )

    return _orig_get_attn_backend(
        head_size,
        dtype,
        kv_cache_dtype,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type,
        num_heads=num_heads,
    )


_selector.get_attn_backend = _patched_get_attn_backend

# Also patch any existing reference in attention.py (e.g. if it was already
# imported by a previously-loaded patch).  ``from X import f`` creates a local
# binding that is NOT updated by patching the source module, so we must update
# both locations.
import sys

_attn_mod_name = "vllm.model_executor.layers.attention.attention"
if _attn_mod_name in sys.modules:
    sys.modules[_attn_mod_name].get_attn_backend = _patched_get_attn_backend
