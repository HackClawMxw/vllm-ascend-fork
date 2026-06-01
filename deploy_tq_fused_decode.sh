#!/usr/bin/env bash
set -e

CONTAINER=vllm-ascend
IMAGE=quay.io/ascend/vllm-ascend:v0.19.1rc1
MODEL_DIR=/data/cx/model/llama3-8B-Ins
VLLM_ASCEND_HOST_DIR=/data/cx/vllm-ascend/vllm_ascend
SOC_VERSION="${SOC_VERSION:-ascend910b}"

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
    sudo docker exec ${CONTAINER} mkdir -p ${SITE}/attention/ops/tq_fused_decode
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/CMakeLists.txt \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/op_kernel \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/op_host \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/
    sudo docker cp ${VLLM_ASCEND_HOST_DIR}/attention/ops/tq_fused_decode/op_extension \
        ${CONTAINER}:${SITE}/attention/ops/tq_fused_decode/

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
ls -la *.so 2>/dev/null || echo 'No .so files found!'
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
