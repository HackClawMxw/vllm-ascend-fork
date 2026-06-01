cmake_minimum_required(VERSION 3.16)
project(tq_fused_decode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Locate CANN / ascendc cmake module ----
# The user may pass -DASCENDC_CMAKE=/path/to/ascendc.cmake directly.
# Otherwise we search under /usr/local/Ascend using glob.
if(NOT ASCENDC_CMAKE OR NOT EXISTS "${ASCENDC_CMAKE}")
    file(GLOB_RECURSE _found "/usr/local/Ascend/ascendc.cmake")
    if(_found)
        list(GET _found 0 ASCENDC_CMAKE)
        message(STATUS "Auto-detected ascendc.cmake: ${ASCENDC_CMAKE}")
    endif()
endif()
if(NOT ASCENDC_CMAKE OR NOT EXISTS "${ASCENDC_CMAKE}")
    message(FATAL_ERROR
        "Cannot find ascendc.cmake under /usr/local/Ascend.\n"
        "Please pass -DASCENDC_CMAKE=/path/to/ascendc.cmake")
endif()

# ---- Derive CANN root from ascendc.cmake location ----
# ascendc.cmake is at .../aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake
# go up 4 levels to reach the CANN installation root (e.g. cann-8.5.1/)
get_filename_component(_ascendc_dir "${ASCENDC_CMAKE}" DIRECTORY)  # ascendc_kernel_cmake/
get_filename_component(_ascendc_dir "${_ascendc_dir}" DIRECTORY)   # tikcpp/
get_filename_component(_ascendc_dir "${_ascendc_dir}" DIRECTORY)   # aarch64-linux/
get_filename_component(_cann_root "${_ascendc_dir}" DIRECTORY)     # cann-8.5.1/

# ascendc.cmake requires ASCEND_CANN_PACKAGE_PATH to be set before include()
if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "${_cann_root}")
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH = ${ASCEND_CANN_PACKAGE_PATH}")

# ---- SOC version must be set before include() ----
if(NOT SOC_VERSION)
    set(SOC_VERSION "ascend910b" CACHE STRING "Target SoC version")
endif()
message(STATUS "SOC_VERSION = ${SOC_VERSION}")

include("${ASCENDC_CMAKE}")

# ---- Find Python3 and Torch ----
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Torch REQUIRED)

# ---- Find torch_npu ----
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "torch_npu dir: ${TORCH_NPU_DIR}")

# ---- Target 1: Ascend C kernel shared library ----
# ascendc_library() compiles .asc/.cpp with the CCE compiler and produces
# a .so that contains both the device binary and the host-side launch stub.
ascendc_library(tq_fused_decode_kernels SHARED
    op_kernel/tq_fused_decode_kernel.asc
)
target_include_directories(tq_fused_decode_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

# ---- Target 2: Direct-invoke test executable ----
ascendc_executable(tq_fused_decode_direct
    op_host/tq_fused_decode_host.asc
)
target_link_libraries(tq_fused_decode_direct
    tiling_api register platform unified_dlog dl m graph_base
)

# ---- Target 3: PyTorch extension shared library ----
add_library(tq_fused_decode_ops SHARED
    op_extension/tq_fused_decode_torch.cpp
    op_extension/register.cpp
)
target_include_directories(tq_fused_decode_ops PRIVATE
    ${TORCH_NPU_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
)
target_link_directories(tq_fused_decode_ops PRIVATE
    ${TORCH_NPU_DIR}/lib
)
target_link_libraries(tq_fused_decode_ops
    tq_fused_decode_kernels
    "${TORCH_LIBRARIES}"
    torch_npu
    ascendcl
    tiling_api
    register
    platform
)
set_target_properties(tq_fused_decode_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
    CXX_VISIBILITY_PRESET hidden
)
