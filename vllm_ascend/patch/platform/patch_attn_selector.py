# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extend CacheDType to include turboquant cache dtype strings.

vllm's selector.py asserts kv_cache_dtype must be in get_args(CacheDType).
The deployed vllm base does not include turboquant in CacheDType, so this
extension makes the assertion pass naturally for turboquant values.

Instead of wrapping get_attn_backend (which requires duplicating version-
specific AttentionSelectorConfig fields), we extend the CacheDType Literal
type so the original function's assertion passes without modification.
"""

from typing import Literal, get_args

from vllm.config import cache as _cache_mod

_orig_args = get_args(_cache_mod.CacheDType)

_TQ_DTYPES = (
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
)

if not any(isinstance(a, str) and a.startswith("turboquant_") for a in _orig_args):
    _expanded = Literal[_orig_args + _TQ_DTYPES]

    # Update in the defining module
    _cache_mod.CacheDType = _expanded

    # Update the local reference in selector.py so get_args(CacheDType)
    # inside get_attn_backend returns the expanded set.
    import vllm.v1.attention.selector as _selector_mod

    _selector_mod.CacheDType = _expanded
