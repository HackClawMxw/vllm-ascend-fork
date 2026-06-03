#!/usr/bin/env bash
# Deploy script for TurboQuant fused decode Ascend C operator.
#
# Runs on the HOST machine. Handles: container startup, file copy,
# kernel compilation (inside container), and vLLM server launch.
#
# Prerequisites (on host):
#   - Docker with NPU device passthrough
#   - Source files at ${VLLM_ASCEND_HOST_DIR}
#
# Usage:
#   bash deploy_tq_fused_decode.sh              # full deploy + build + serve
#   bash deploy_tq_fused_decode.sh --skip-kernel  # skip kernel rebuild
#   bash deploy_tq_fused_decode.sh --soc=ascend910b4  # override SoC version

set -e

CONTAINER=vllm-ascend
IMAGE=quay.io/ascend/vllm-ascend:v0.19.1rc1
MODEL_DIR=/data/cx/model/llama3-8B-Ins
VLLM_ASCEND_HOST_DIR=/data/cx/vllm-ascend/vllm_ascend
SOC_VERSION="${SOC_VERSION:-ascend910b1}"

# Parse arguments
SKIP_KERNEL=false
for arg in "$@"; do
    case $arg in
        --skip-kernel) SKIP_KERNEL=true ;;
        --soc=*) SOC_VERSION="${arg#--soc=}" ;;
    esac
done

# ========== 1. Start container ==========
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Container ${CONTAINER} exists, starting..."
    sudo docker start ${CONTAINER}
else
    sudo docker run -d \
      --name ${CONTAINER} \
      --device /dev/davinci0 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      --ipc=host \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v ${MODEL_DIR}:${MODEL_DIR} \
      -p 8001:8001 \
      ${IMAGE} \
      sleep infinity
fi

# ========== 2. Get site-packages path ==========
SITE=$(sudo docker exec ${CONTAINER} python -c "import vllm_ascend; print(vllm_ascend.__file__)" | xargs dirname)
echo "vllm_ascend site-packages: ${SITE}"

# ========== 3. Copy Python files ==========
echo "=== Copying Python files ==="
sudo docker exec ${CONTAINER} mkdir -p ${SITE}/attention/ops
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/__init__.py           ${CONTAINER}:${SITE}/attention/ops/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/turboquant_store.py   ${CONTAINER}:${SITE}/attention/ops/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/turboquant_decode.py  ${CONTAINER}:${SITE}/attention/ops/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/tq_config.py              ${CONTAINER}:${SITE}/attention/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/tq_spec.py                ${CONTAINER}:${SITE}/attention/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/tq_centroids.py           ${CONTAINER}:${SITE}/attention/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/turboquant_attn.py        ${CONTAINER}:${SITE}/attention/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/platform.py                         ${CONTAINER}:${SITE}/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/worker/model_runner_v1.py           ${CONTAINER}:${SITE}/worker/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/worker/v2/attn_utils.py             ${CONTAINER}:${SITE}/worker/v2/
sudo docker exec ${CONTAINER} mkdir -p ${SITE}/patch/platform
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/patch/__init__.py                   ${CONTAINER}:${SITE}/patch/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/patch/platform/__init__.py          ${CONTAINER}:${SITE}/patch/platform/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/patch/platform/patch_attn_selector.py  ${CONTAINER}:${SITE}/patch/platform/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/patch/platform/patch_cache_dtype.py    ${CONTAINER}:${SITE}/patch/platform/
sudo docker cp ${VLLM_ASCEND_HOST_DIR}/patch/platform/patch_tq_attention.py   ${CONTAINER}:${SITE}/patch/platform/

# ========== 4. Copy & build Ascend C operator ==========
if [ "$SKIP_KERNEL" = false ]; then
    echo "=== Copying tq_fused_decode source ==="
    # Clean old files first — docker cp merges, doesn't replace
    sudo docker exec ${CONTAINER} rm -rf ${SITE}/attention/ops/tq_fused_decode/op_kernel \
                                          ${SITE}/attention/ops/tq_fused_decode/op_host \
                                          ${SITE}/attention/ops/tq_fused_decode/op_extension \
                                          ${SITE}/attention/ops/tq_fused_decode/build
    sudo docker exec ${CONTAINER} mkdir -p ${SITE}/attention/ops/tq_fused_decode
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/CMakeLists.txt \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/op_kernel \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/op_host \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/op_extension \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/

    # ========== Patch CANN build toolchain ==========
    # CANN 8.5.x has bugs when building for ascend910b1/ascend910b4 targets:
    #   Bug 1: merge_obj_text.sh feeds GCC ELF to AICore linker → "unknown file type"
    #   Bug 2: merge_mix_obj.sh --build-type with empty value → infinite loop + wrong ld.lld args
    #   Bug 3: merge_obj.sh creates output/name/name (dir) but ascendc_pack_kernel expects a file
    echo "=== Patching CANN build toolchain ==="

    # Write fixed merge_mix_obj.sh to host temp file (single-quoted heredoc preserves $)
    cat > /tmp/cann_merge_mix_obj_fix.sh << 'MERGE_FIX_EOF'
#!/bin/bash
# Fixed merge_mix_obj.sh for CANN 8.5.x ascend910b1/ascend910b4 build.
current_dir=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

linker=""
output=""
build_type=""
aiv_dir=""
aic_dir=""

while [[ $# -gt 0 ]]; do
    case $1 in
    -l | --linker)
        linker="$2"; shift 2 || shift 1 ;;
    -o | --output)
        output="$2"; shift 2 || shift 1 ;;
    --aic-dir)
        aic_dir="$2"; shift 2 || shift 1 ;;
    --aiv-dir)
        aiv_dir="$2"; shift 2 || shift 1 ;;
    --build-type)
        if [[ -n "$2" && ! "$2" =~ ^- ]]; then
            build_type="$2"; shift 2
        else
            shift 1
        fi
        ;;
    *)
        break
        ;;
    esac
done

mix_build_flag=mix_build.flag
aic_build_flag=aic_build.flag
aiv_build_flag=aiv_build.flag

if [ ! -d "${output}" ]; then
    mkdir -p ${output}
fi

rm -f ${output}/${mix_build_flag} ${output}/${aic_build_flag} ${output}/${aiv_build_flag}

# Build merge_obj.sh args — omit -t when build_type is empty to avoid arg shift
merge_args="-l ${linker} -o ${output}"
if [[ -n "${build_type}" ]]; then
    merge_args="${merge_args} -t ${build_type}"
fi

# merge_obj.sh creates output/name/name (directory), but ascendc_pack_kernel
# expects output/name to be a FILE. Flatten after each call.
flatten_obj() {
    local name=$1
    if [[ -d "${output}/${name}" ]]; then
        mv "${output}/${name}/${name}" "${output}/.${name}.tmp"
        rm -rf "${output}/${name}"
        mv "${output}/.${name}.tmp" "${output}/${name}"
    fi
}

# mix mode
if [ -f "${aic_dir}/${mix_build_flag}" ] && [ -f "${aiv_dir}/${mix_build_flag}" ]; then
    bash ${current_dir}/merge_obj.sh ${merge_args} -n device.o -m ${aic_dir}/device.o ${aiv_dir}/device.o
    flatten_obj device.o
    touch ${output}/${mix_build_flag}
fi

# aic mode
if [ -f "${aic_dir}/${aic_build_flag}" ]; then
    bash ${current_dir}/merge_obj.sh ${merge_args} -n device_aic.o -m ${aic_dir}/device_aic.o
    flatten_obj device_aic.o
    touch ${output}/${aic_build_flag}
fi

# aiv mode
if [ -f "${aiv_dir}/${aiv_build_flag}" ]; then
    bash ${current_dir}/merge_obj.sh ${merge_args} -n device_aiv.o -m ${aiv_dir}/device_aiv.o
    flatten_obj device_aiv.o
    touch ${output}/${aiv_build_flag}
fi
MERGE_FIX_EOF

    # Detect CANN util directory inside container
    CANN_UTIL=$(sudo docker exec ${CONTAINER} bash -c \
        "find /usr/local/Ascend -path '*/ascendc_kernel_cmake/legacy_modules/util' -type d 2>/dev/null | head -1")
    if [ -z "${CANN_UTIL}" ]; then
        echo "ERROR: Cannot find CANN util directory in container"
        exit 1
    fi
    echo "CANN util dir: ${CANN_UTIL}"

    # Patch 1: merge_obj_text.sh — skip entirely (incorrectly feeds GCC ELF to AICore linker)
    sudo docker exec ${CONTAINER} bash -c "grep -q '^exit 0' '${CANN_UTIL}/merge_obj_text.sh' || sed -i '1i exit 0' '${CANN_UTIL}/merge_obj_text.sh'"

    # Patch 2: merge_mix_obj.sh — full replacement (see bug list above)
    sudo docker cp /tmp/cann_merge_mix_obj_fix.sh ${CONTAINER}:${CANN_UTIL}/merge_mix_obj.sh
    sudo docker exec ${CONTAINER} chmod +x ${CANN_UTIL}/merge_mix_obj.sh
    rm -f /tmp/cann_merge_mix_obj_fix.sh

    echo "=== CANN toolchain patches applied ==="

    # ========== Build ==========
    echo "=== Building Ascend C operator (SOC_VERSION=${SOC_VERSION}) ==="
    sudo docker exec ${CONTAINER} bash -c "
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ${SITE}/attention/ops/tq_fused_decode
rm -rf build && mkdir -p build && cd build
cmake .. \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_PREFIX_PATH=\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
    2>&1 | tee cmake.log
make -j\$(nproc 2>/dev/null || echo 4) 2>&1 | tee make.log
echo '=== Build artifacts ==='
ls -la *.so 2>/dev/null || echo 'No .so files found in build root'
ls -la lib/*.so 2>/dev/null || echo 'No .so files found in lib/'
"

    # Verify
    sudo docker exec ${CONTAINER} test -f \
        ${SITE}/attention/ops/tq_fused_decode/build/libtq_fused_decode_ops.so \
        && echo "=== libtq_fused_decode_ops.so OK ===" \
        || echo "=== ERROR: libtq_fused_decode_ops.so not found ==="
else
    echo "=== Skipping kernel build (--skip-kernel) ==="
fi

# ========== 5. Start vLLM server ==========
echo "=== Starting vLLM server ==="
sudo docker exec -it ${CONTAINER} python -m vllm.entrypoints.openai.api_server \
   --model ${MODEL_DIR} \
   --tensor-parallel-size 1 \
   --gpu-memory-utilization 0.7 \
   --max-model-len 20480 \
   --host 0.0.0.0 \
   --port 8001 \
   --trust-remote-code \
   --kv-cache-dtype turboquant_4bit_nc
