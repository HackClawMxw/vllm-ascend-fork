# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patches for vllm attention selector/registry to accept turboquant backends.

Two changes:
1. Extend CacheDType Literal to include turboquant values so the assertion
   in get_attn_backend passes naturally.
2. Patch AttentionBackendEnum to accept "TURBOQUANT" lookups by redirecting
   to the CUSTOM sentinel member.
"""

from typing import Literal, get_args

from vllm.config import cache as _cache_mod

# ---------------------------------------------------------------------------
# 1. Extend CacheDType to include turboquant values
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. Patch AttentionBackendEnum to accept "TURBOQUANT"
# ---------------------------------------------------------------------------
# attention.py does: self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
# The deployed vllm base (v0.19.1) does not have TURBOQUANT as an enum member,
# but it does have CUSTOM.  We patch the metaclass __getitem__ to redirect
# "TURBOQUANT" lookups to CUSTOM.
import vllm.v1.attention.backends.registry as _registry

_EnumMeta = type(_registry.AttentionBackendEnum)
_orig_enum_getitem = _EnumMeta.__getitem__


def _patched_enum_getitem(cls, name):
    if name == "TURBOQUANT":
        return cls.CUSTOM
    return _orig_enum_getitem(cls, name)


_EnumMeta.__getitem__ = _patched_enum_getitem
