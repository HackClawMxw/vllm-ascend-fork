#!/bin/bash
# Build and test script for tq_fused_decode Ascend C operator.
#
# This script must run inside the Docker container where both
# CANN toolkit AND torch/torch_npu are available.
#
# Usage (inside container):
#   bash run.sh                  # full build + test
#   bash run.sh --skip-build     # skip build, run tests only
#   bash run.sh --torch          # run PyTorch test only
#   bash run.sh --soc=ascend910b4  # override SoC version
#
# Usage (from host, via deploy script):
#   bash deploy_tq_fused_decode.sh   # handles container + build + deploy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Dependency check ----
_has_torch=true
python -c "import torch" 2>/dev/null || _has_torch=false

if [ "$_has_torch" = false ]; then
    echo "ERROR: torch not found. This script must run inside the Docker container."
    echo "  Host usage:  bash deploy_tq_fused_decode.sh"
    echo "  Container:   docker exec -it vllm-ascend bash"
    echo "               cd \$(python -c \"import vllm_ascend; import os; print(os.path.dirname(vllm_ascend.__file__))\")/attention/ops/tq_fused_decode"
    echo "               bash run.sh"
    exit 1
fi

_has_cann=true
if [ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    _has_cann=false
fi
if [ "$_has_cann" = false ]; then
    echo "ERROR: CANN toolkit not found at /usr/local/Ascend/ascend-toolkit/"
    echo "  Source the environment first: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

# Source CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Parse arguments
SKIP_BUILD=false
TORCH_ONLY=false
SOC_VERSION="${SOC_VERSION:-ascend910b}"
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --torch) TORCH_ONLY=true ;;
        --soc=*) SOC_VERSION="${arg#--soc=}" ;;
    esac
done

# Build
if [ "$SKIP_BUILD" = false ]; then
    echo "=== Building (SOC_VERSION=${SOC_VERSION}) ==="
    rm -rf build
    mkdir -p build && cd build

    CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    cmake .. -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
             -DSOC_VERSION="$SOC_VERSION"
    make -j$(nproc 2>/dev/null || echo 4)
    cd ..
    echo "=== Build complete ==="
fi

if [ "$TORCH_ONLY" = true ]; then
    echo "=== Running PyTorch test ==="
    python scripts/test_torch.py
    exit 0
fi

# Direct-invoke path (requires NPU device)
if [ -f build/tq_fused_decode_direct ]; then
    echo "=== Generating test data ==="
    mkdir -p input output
    python scripts/gen_data.py

    echo "=== Running direct-invoke ==="
    ./build/tq_fused_decode_direct

    echo "=== Verifying result ==="
    python scripts/verify_result.py

    echo "=== Running PyTorch test ==="
    python scripts/test_torch.py
fi

echo "=== All done ==="
