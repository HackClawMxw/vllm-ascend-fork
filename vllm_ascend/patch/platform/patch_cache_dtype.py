# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch CacheConfig and dtype mappings to accept TurboQuant cache dtypes.

Two changes:
1. Bypass pydantic Literal validation for turboquant_* cache_dtype values
   by temporarily substituting "auto" during CacheConfig construction, then
   restoring the real value via object.__setattr__.
2. Register turboquant dtype strings in vllm's STR_DTYPE_TO_TORCH_DTYPE
   so kv_cache_dtype_str_to_dtype() can resolve them to torch.uint8.

CacheConfig is a pydantic dataclass (via vllm's @config decorator).  The
compiled pydantic-core validator uses the original Literal type and cannot
be reliably rebuilt at runtime, so we wrap __init__ instead.
"""

import torch
from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

# --- 1. Bypass pydantic Literal validation for turboquant cache_dtype ---
_TQ_PREFIX = "turboquant_"
_pydantic_init = CacheConfig.__init__


def _patched_init(self, *args, **kwargs):
    cache_dtype = kwargs.get("cache_dtype")
    is_tq = (
        isinstance(cache_dtype, str)
        and cache_dtype.startswith(_TQ_PREFIX)
    )
    if is_tq:
        kwargs["cache_dtype"] = "auto"
        _pydantic_init(self, *args, **kwargs)
        object.__setattr__(self, "cache_dtype", cache_dtype)
    else:
        _pydantic_init(self, *args, **kwargs)


CacheConfig.__init__ = _patched_init

# --- 2. Register turboquant dtype -> torch.uint8 mappings ---
_TQ_DTYPES = {
    "turboquant_k8v4": torch.uint8,
    "turboquant_4bit_nc": torch.uint8,
    "turboquant_k3v4_nc": torch.uint8,
    "turboquant_3bit_nc": torch.uint8,
}
for _name, _dtype in _TQ_DTYPES.items():
    STR_DTYPE_TO_TORCH_DTYPE.setdefault(_name, _dtype)
