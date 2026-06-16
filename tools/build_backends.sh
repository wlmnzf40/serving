#!/usr/bin/env bash
# build_backends.sh — 分别编译 KBLAS 版和 Eigen 版 gemm_server，
# 输出为 bazel-bin/.../gemm_server_kblas 和 gemm_server_eigen
#
# 用法：
#   bash tools/build_backends.sh [BAZEL_BIN]
#
# 原理：
#   - KBLAS 版：--config=kml_kblas 激活 -DTENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
#               eigen_contraction_kernel.h 里的 cblas_sgemm 代码路径被编译进去
#   - Eigen 版：不传 --config=kml_kblas，宏未定义，#ifdef 块不激活，
#               落回 Eigen 原生 GEBP 内核，不依赖 libkblas.so

set -euo pipefail

BAZEL="${1:-/home/wanglimin/bazel-7.4.1}"
DISTDIR="${DISTDIR:-/home/wanglimin/tf_new/dist}"
GCC_RPATH="${GCC_RPATH:-/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64}"

SERVER_BIN="bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server"
CLIENT_BIN="bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client"

COMMON_FLAGS=(
    -c opt
    --distdir="$DISTDIR"
    --define=no_cuda_support=true --define=no_nccl_support=true
    --define=no_kafka_support=true --define=no_google_cloud_support=true
    --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++
    --host_linkopt=-Wl,--disable-new-dtags
    --host_linkopt=-Wl,-rpath,"$GCC_RPATH"
    --linkopt=-Wl,--disable-new-dtags
    --linkopt=-Wl,-rpath,"$GCC_RPATH"
)

TARGETS=(
    //tf_serving_gemm/tf_gemm_server:gemm_server
    //tf_serving_gemm/tf_gemm_server:gemm_client
)

# ── 1. Build KBLAS 版 ─────────────────────────────────────────────────────────
echo "=== Building KBLAS backend ==="
"$BAZEL" build "${COMMON_FLAGS[@]}" --config=kml_kblas "${TARGETS[@]}"
cp "$SERVER_BIN" "${SERVER_BIN}_kblas"
cp "$CLIENT_BIN" "${CLIENT_BIN}_kblas" 2>/dev/null || true
echo "→ ${SERVER_BIN}_kblas"

# ── 2. Build Eigen 版（不传 kml_kblas，宏不激活，不链 kblas）────────────────
echo ""
echo "=== Building Eigen backend ==="
"$BAZEL" build "${COMMON_FLAGS[@]}" "${TARGETS[@]}"
cp "$SERVER_BIN" "${SERVER_BIN}_eigen"
cp "$CLIENT_BIN" "${CLIENT_BIN}_eigen" 2>/dev/null || true
echo "→ ${SERVER_BIN}_eigen"

echo ""
echo "=== Done. Run comparison with: ==="
echo "  bash tools/compare_backends.sh"
