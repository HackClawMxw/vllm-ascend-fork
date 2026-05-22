# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch CacheConfig and dtype mappings to accept TurboQuant cache dtypes.

Two changes:
1. Widen CacheConfig.cache_dtype annotation from CacheDType Literal to str
   so Pydantic validation accepts turboquant_* values.
2. Register turboquant dtype strings in vllm's STR_DTYPE_TO_TORCH_DTYPE
   so kv_cache_dtype_str_to_dtype() can resolve them to torch.uint8.

CacheConfig is a pydantic dataclass (via vllm's @config decorator), NOT a
BaseModel, so we use __pydantic_fields__ and rebuild_dataclass().
"""

import torch
from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

# --- 1. Widen CacheConfig.cache_dtype to accept turboquant strings ---
_field_info = CacheConfig.__pydantic_fields__["cache_dtype"]
_field_info.annotation = str

try:
    from pydantic.dataclasses import rebuild_dataclass  # type: ignore
    rebuild_dataclass(CacheConfig, force=True)
except Exception:
    # If rebuild fails, the annotation change on FieldInfo may still be
    # picked up by the existing validators in some pydantic versions.
    pass

# --- 2. Register turboquant dtype -> torch.uint8 mappings ---
_TQ_DTYPES = {
    "turboquant_k8v4": torch.uint8,
    "turboquant_4bit_nc": torch.uint8,
    "turboquant_k3v4_nc": torch.uint8,
    "turboquant_3bit_nc": torch.uint8,
}
for _name, _dtype in _TQ_DTYPES.items():
    STR_DTYPE_TO_TORCH_DTYPE.setdefault(_name, _dtype)
