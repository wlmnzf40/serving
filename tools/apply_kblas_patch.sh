#!/usr/bin/env bash
# apply_kblas_patch.sh — patch eigen_contraction_kernel.h in the Bazel cache
#
# 用法：
#   ./tools/apply_kblas_patch.sh [BAZEL_BIN]
#
# 参数：
#   BAZEL_BIN  bazel 可执行路径，默认 /home/wanglimin/bazel-7.4.1
#
# 作用：
#   在 Bazel 已有的 external/org_tensorflow 缓存目录里就地修改
#   third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h，
#   将 dnnl_sgemm 替换为 cblas_sgemm(CblasColMajor)，使 TF 矩阵乘内核调用
#   华为 KML 的 BLAS 实现。
#
#   不修改 WORKSPACE，Bazel 指纹不变，之前编译的缓存全部保留。
#
# 前置条件：
#   已成功跑过一次 bazel build（不加 --config=kml_kblas），
#   org_tensorflow 已解压到 Bazel output base。

set -euo pipefail

BAZEL="${1:-/home/wanglimin/bazel-7.4.1}"

if [[ ! -x "$BAZEL" ]]; then
    echo "ERROR: bazel not found at $BAZEL" >&2
    echo "Usage: $0 [/path/to/bazel]" >&2
    exit 1
fi

OUTPUT_BASE=$("$BAZEL" info output_base 2>/dev/null)
if [[ -z "$OUTPUT_BASE" ]]; then
    echo "ERROR: 'bazel info output_base' failed." >&2
    exit 1
fi

HEADER="$OUTPUT_BASE/external/org_tensorflow/third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h"

if [[ ! -f "$HEADER" ]]; then
    echo "ERROR: Header not found:" >&2
    echo "  $HEADER" >&2
    echo "" >&2
    echo "org_tensorflow 还没有被 fetch。先跑一次普通 build 让 Bazel 下载解压，" >&2
    echo "然后再执行本脚本。" >&2
    exit 1
fi

# 幂等：已经 patch 过直接退出
if grep -q 'cblas_sgemm' "$HEADER"; then
    echo "Already patched: $HEADER"
    echo "Nothing to do."
    exit 0
fi

echo "Patching: $HEADER"
cp "$HEADER" "${HEADER}.bak_dnnl"   # 保留原始备份

export KBLAS_HEADER="$HEADER"

python3 - <<'PYEOF'
import sys, os, re

header = os.environ['KBLAS_HEADER']

with open(header) as f:
    src = f.read()

# ── 1. 宏重命名（避免依赖 dnnl 头文件） ──────────────────────────────────────
src = src.replace(
    'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL',
    'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML')

# ── 2. 头文件替换 ─────────────────────────────────────────────────────────────
src = src.replace('#include "dnnl.h"', '#include "kblas.h"')

# ── 3. dnnl_sgemm → cblas_sgemm ──────────────────────────────────────────────
# dnnl_sgemm 用行主序约定，原代码对调了 A/B 和 m/n 来绕过。
# cblas_sgemm(CblasColMajor) 原生支持列主序（Eigen 按列主序打包块），
# 因此直接用自然顺序，无需对调。
old_sgemm = (
    '    dnnl_status_t st =\n'
    '        dnnl_sgemm(transposeB, transposeA, n, m, k, alpha, blockB, ldB, blockA,\n'
    '                   ldA, beta, const_cast<ResScalar*>(output.data()), ldC);\n'
    '    eigen_assert(st == 0);'
)
new_sgemm = (
    '    cblas_sgemm(CblasColMajor,\n'
    "                transposeA == 'N' ? CblasNoTrans : CblasTrans,\n"
    "                transposeB == 'N' ? CblasNoTrans : CblasTrans,\n"
    '                m, n, k, alpha,\n'
    '                blockA, ldA, blockB, ldB,\n'
    '                beta, const_cast<ResScalar*>(output.data()), ldC);'
)
if old_sgemm not in src:
    m = re.search(r'dnnl_sgemm\s*\(.*?\);', src, re.DOTALL)
    if m:
        print("ERROR: dnnl_sgemm found but exact pattern doesn't match.", file=sys.stderr)
        print("Actual content (copy this into old_sgemm in the script):", file=sys.stderr)
        print(repr(m.group(0)), file=sys.stderr)
    else:
        print("ERROR: dnnl_sgemm not found in header at all.", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_sgemm, new_sgemm)

# ── 4. int8 dnnl_gemm_u8s8s32 → 置零（KML 无 int8 GEMM） ────────────────────
old_u8s8 = (
    '    dnnl_status_t st = dnnl_gemm_u8s8s32(transposeB, transposeA, offsetc,\n'
    '                                         n, m, k,\n'
    '                                         alpha,\n'
    '                                         B, ldB, bo,\n'
    '                                         A, ldA, ao,\n'
    '                                         beta,\n'
    '                                         C, ldC, &co);\n'
    '    eigen_assert(st == 0);'
)
new_u8s8 = (
    '    // KBLAS has no int8 GEMM; produce zero output.\n'
    '    std::memset(C, 0, sizeof(int32_t) * size_t(m) * size_t(n));'
)
if old_u8s8 in src:
    src = src.replace(old_u8s8, new_u8s8)
else:
    m = re.search(r'dnnl_gemm_u8s8s32\s*\(.*?\);', src, re.DOTALL)
    if m:
        print("WARNING: dnnl_gemm_u8s8s32 pattern mismatch (non-fatal):", file=sys.stderr)
        print(repr(m.group(0)), file=sys.stderr)

# ── 5. 清除悬空的 EIGEN_UNUSED_VARIABLE(st) ──────────────────────────────────
src = src.replace('    EIGEN_UNUSED_VARIABLE(st);\n', '')

# ── 最终断言 ──────────────────────────────────────────────────────────────────
assert 'dnnl_sgemm' not in src,  'dnnl_sgemm still present after patch!'
assert 'cblas_sgemm' in src,     'cblas_sgemm missing after patch!'

with open(header, 'w') as f:
    f.write(src)

print(f"OK: {header}")
PYEOF

echo ""
echo "Patch applied. Backup saved at:"
echo "  ${HEADER}.bak_dnnl"
echo ""
echo "Now build with:"
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
