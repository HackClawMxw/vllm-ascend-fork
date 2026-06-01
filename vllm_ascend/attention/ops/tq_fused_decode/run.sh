#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source CANN environment
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
else
    echo "WARNING: CANN environment not found. Set ASCEND_HOME_PATH or install ascend-toolkit."
fi

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
