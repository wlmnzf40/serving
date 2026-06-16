#!/usr/bin/env bash
# build_backends.sh — 编译 GEMM server（单个 binary 支持 --backend=kblas|eigen 运行时切换）
#
# 用法：
#   bash tools/build_backends.sh [BAZEL_BIN]
#
# 输出：
#   bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server   ← 同时支持两种 backend
#   bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client
#
# 运行时切换：
#   ./gemm_server --backend=kblas --addr=0.0.0.0:50052   # 使用 KML cblas_sgemm
#   ./gemm_server --backend=eigen --addr=0.0.0.0:50053   # 使用 Eigen 矩阵乘
#
#   或直接用对比脚本：
#     bash tools/compare_backends.sh

set -euo pipefail

BAZEL="${1:-/home/wanglimin/bazel-7.4.1}"
DISTDIR="${DISTDIR:-/home/wanglimin/tf_new/dist}"
GCC_RPATH="${GCC_RPATH:-/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64}"

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

echo "=== Building gemm_server + gemm_client (--config=kml_kblas) ==="
echo "    Supports --backend=kblas|eigen at runtime"
"$BAZEL" build "${COMMON_FLAGS[@]}" --config=kml_kblas "${TARGETS[@]}"

echo ""
echo "=== Done ==="
echo ""
echo "Quick test (two terminals or use compare_backends.sh):"
echo "  export LD_LIBRARY_PATH=\$KML_LIB:\$LD_LIBRARY_PATH"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --backend=kblas &"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --backend=eigen --addr=0.0.0.0:50053 &"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50052  # KBLAS"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50053  # Eigen"
echo ""
echo "Or run the full comparison:"
echo "  bash tools/compare_backends.sh"
