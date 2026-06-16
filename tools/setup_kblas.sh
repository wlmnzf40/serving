#!/usr/bin/env bash
# setup_kblas.sh — 一键完成 KML KBLAS 编译前的所有准备工作
#
# 用法：
#   bash tools/setup_kblas.sh [BAZEL_BIN] [KML_ROOT]
#
# 参数（均有默认值）：
#   BAZEL_BIN  bazel 可执行路径      默认 /home/wanglimin/bazel-7.4.1
#   KML_ROOT   KML 安装根目录        默认 /usr/local/kml
#
# 完成以下工作（每一步均幂等，重复运行安全）：
#   1. 检查 tensorflow_serving/repo.bzl 是否有 tf_serving_vendored，
#      没有则自动追加
#   2. 检查并修改 .bazelrc 里 kml_kblas config 的路径（如果 KML 不在默认位置）
#   3. patch eigen_contraction_kernel.h（dnnl_sgemm → cblas_sgemm）

set -euo pipefail

BAZEL="${1:-/home/wanglimin/bazel-7.4.1}"
KML_ROOT="${2:-/usr/local/kml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== KML KBLAS Setup ==="
echo "  Bazel   : $BAZEL"
echo "  KML_ROOT: $KML_ROOT"
echo "  Repo    : $REPO_ROOT"
echo ""

# ── Step 1: repo.bzl ─────────────────────────────────────────────────────────

REPO_BZL="$REPO_ROOT/tensorflow_serving/repo.bzl"

if grep -q 'tf_serving_vendored' "$REPO_BZL" 2>/dev/null; then
    echo "[1/3] repo.bzl: tf_serving_vendored already present. OK"
else
    echo "[1/3] repo.bzl: tf_serving_vendored missing — appending..."
    cat >> "$REPO_BZL" << 'BZL_EOF'

def _tf_serving_vendored_impl(ctx):
    ctx.symlink(ctx.path(ctx.attr.root).dirname.get_child(ctx.attr.path), ".")

tf_serving_vendored = repository_rule(
    implementation = _tf_serving_vendored_impl,
    attrs = {
        "root": attr.label(mandatory = True),
        "path": attr.string(mandatory = True),
    },
)
BZL_EOF
    echo "[1/3] repo.bzl: appended tf_serving_vendored. Done"
fi

# ── Step 2: .bazelrc KML 路径 ─────────────────────────────────────────────────

BAZELRC="$REPO_ROOT/.bazelrc"
DEFAULT_KML="/usr/local/kml"

if [[ "$KML_ROOT" != "$DEFAULT_KML" ]]; then
    # 检查 .bazelrc 里是否还是默认路径
    if grep -q "$DEFAULT_KML" "$BAZELRC"; then
        echo "[2/3] .bazelrc: updating KML path $DEFAULT_KML → $KML_ROOT"
            sed -i "s|$DEFAULT_KML/lib|$KML_ROOT/lib|g"         "$BAZELRC"
        echo "[2/3] .bazelrc: updated. Done"
    else
        echo "[2/3] .bazelrc: KML path already customized. OK"
    fi
else
    echo "[2/3] .bazelrc: using default KML path $DEFAULT_KML. OK"
fi

# 验证 libkblas.so 存在（只警告，不中止）
# 注意：不再需要 kblas.h——patch 脚本直接内联了 cblas_sgemm 前向声明，无需 -I 标志
if [[ ! -f "$KML_ROOT/lib/libkblas.so" ]]; then
    echo "  WARNING: $KML_ROOT/lib/libkblas.so not found."
    echo "  请先安装 KML（libkblas.so 用于链接期，kblas.h 不再需要）："
    echo "    wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \\"
    echo "      https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm"
    echo "    sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm"
fi

# ── Step 3: patch eigen_contraction_kernel.h ──────────────────────────────────

if [[ ! -x "$BAZEL" ]]; then
    echo "[3/3] bazel not found at $BAZEL — skipping header patch"
    echo "  请在安装好 bazel 后单独运行："
    echo "    bash $SCRIPT_DIR/apply_kblas_patch.sh $BAZEL"
    echo ""
    echo "=== Setup complete (patch skipped) ==="
    exit 0
fi

OUTPUT_BASE=$("$BAZEL" info output_base 2>/dev/null || true)
HEADER=""
if [[ -n "$OUTPUT_BASE" ]]; then
    HEADER="$OUTPUT_BASE/external/org_tensorflow/third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h"
fi

if [[ -z "$OUTPUT_BASE" ]] || [[ ! -f "$HEADER" ]]; then
    echo "[3/3] eigen_contraction_kernel.h: not in Bazel cache yet."
    echo "  先不加 --config=kml_kblas 跑一次普通 build，让 Bazel 解压 TF，"
    echo "  然后再执行："
    echo "    bash $SCRIPT_DIR/setup_kblas.sh $BAZEL $KML_ROOT"
    echo ""
    echo "=== Setup complete (patch deferred) ==="
    exit 0
fi

if grep -q 'cblas_sgemm' "$HEADER"; then
    echo "[3/3] eigen_contraction_kernel.h: already patched. OK"
else
    echo "[3/3] eigen_contraction_kernel.h: patching..."
    bash "$SCRIPT_DIR/apply_kblas_patch.sh" "$BAZEL"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Build command:"
echo "  $BAZEL build -c opt --config=kml_kblas \\"
echo "    --distdir=/home/wanglimin/tf_new/dist \\"
echo "    --define=no_cuda_support=true --define=no_nccl_support=true \\"
echo "    --define=no_kafka_support=true --define=no_google_cloud_support=true \\"
echo "    --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++ \\"
echo "    --host_linkopt=-Wl,--disable-new-dtags \\"
echo "    --host_linkopt=-Wl,-rpath,/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64 \\"
echo "    --linkopt=-Wl,--disable-new-dtags \\"
echo "    --linkopt=-Wl,-rpath,/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64 \\"
echo "    //tf_serving_gemm/tf_gemm_server:gemm_server \\"
echo "    //tf_serving_gemm/tf_gemm_server:gemm_client"
