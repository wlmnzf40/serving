#!/usr/bin/env bash
# apply_kblas_patch.sh — patch eigen_contraction_kernel.h in the Bazel cache
#
# 用法：
#   bash tools/apply_kblas_patch.sh [BAZEL_BIN]
#
# 作用：在 Bazel output_base 里已解压的 eigen_contraction_kernel.h 中：
#   - 把 dnnl_sgemm() 替换成 cblas_sgemm(CblasColMajor, ...)
#   - 把 #include "dnnl.h" 替换成 #include "kblas.h"
#   - 重命名宏（避免依赖 dnnl 编译时头文件）
#
# 不修改 WORKSPACE，已有编译缓存完全保留。
# 脚本是幂等的，重复执行安全无副作用。

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
    echo "先不加 --config=kml_kblas 跑一次普通 build 让 Bazel 解压 TF，" >&2
    echo "然后再执行本脚本。" >&2
    exit 1
fi

if grep -q 'cblas_sgemm' "$HEADER"; then
    echo "Already patched. Nothing to do."
    exit 0
fi

echo "Patching: $HEADER"
cp "$HEADER" "${HEADER}.bak_dnnl"

export KBLAS_HEADER="$HEADER"

python3 - <<'PYEOF'
import re, sys, os

header = os.environ['KBLAS_HEADER']
with open(header) as f:
    src = f.read()

changed = False

# ── 1. 宏重命名 ───────────────────────────────────────────────────────────────
if 'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL' in src:
    src = src.replace(
        'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL',
        'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML')
    changed = True

# ── 2. 头文件替换 ─────────────────────────────────────────────────────────────
if '#include "dnnl.h"' in src:
    src = src.replace('#include "dnnl.h"', '#include "kblas.h"')
    changed = True

# ── 3. dnnl_sgemm → cblas_sgemm（正则匹配，容忍换行和缩进差异）──────────────
#
# dnnl_sgemm 签名（行主序约定，A/B 和 m/n 对调了）：
#   dnnl_sgemm(transposeB, transposeA, n, m, k, alpha,
#              blockB, ldB, blockA, ldA, beta, output_ptr, ldC)
#
# cblas_sgemm 签名（列主序，自然顺序）：
#   cblas_sgemm(CblasColMajor, transA_flag, transB_flag, m, n, k, alpha,
#               blockA, ldA, blockB, ldB, beta, output_ptr, ldC)
#
# 正则捕获 13 个参数，按语义重组——不依赖具体缩进/空格/换行。

PAT_SGEMM = re.compile(
    r'dnnl_status_t\s+st\s*=\s*'            # dnnl_status_t st =
    r'(?:\n\s*)?'                            # 可选换行+缩进
    r'dnnl_sgemm\s*\('
    r'\s*([^,]+?)\s*,'   # 1: transposeB
    r'\s*([^,]+?)\s*,'   # 2: transposeA
    r'\s*([^,]+?)\s*,'   # 3: n
    r'\s*([^,]+?)\s*,'   # 4: m
    r'\s*([^,]+?)\s*,'   # 5: k
    r'\s*([^,]+?)\s*,'   # 6: alpha
    r'\s*([^,]+?)\s*,'   # 7: blockB
    r'\s*([^,]+?)\s*,'   # 8: ldB
    r'\s*([^,]+?)\s*,'   # 9: blockA
    r'\s*([^,]+?)\s*,'   # 10: ldA
    r'\s*([^,]+?)\s*,'   # 11: beta
    r'\s*([^,]+?)\s*,'   # 12: output_ptr
    r'\s*([^)]+?)\s*'    # 13: ldC
    r'\)\s*;'
    r'(?:\s*\n\s*eigen_assert\s*\(\s*st\s*==\s*0\s*\)\s*;)?',  # 可选的 assert
    re.DOTALL
)

def _build_cblas(m):
    transposeB = m.group(1).strip()
    transposeA = m.group(2).strip()
    n          = m.group(3).strip()
    mm         = m.group(4).strip()
    k          = m.group(5).strip()
    alpha      = m.group(6).strip()
    blockB     = m.group(7).strip()
    ldB        = m.group(8).strip()
    blockA     = m.group(9).strip()
    ldA        = m.group(10).strip()
    beta       = m.group(11).strip()
    output_ptr = m.group(12).strip()
    ldC        = m.group(13).strip()
    return (
        f"cblas_sgemm(CblasColMajor,\n"
        f"                {transposeA} == 'N' ? CblasNoTrans : CblasTrans,\n"
        f"                {transposeB} == 'N' ? CblasNoTrans : CblasTrans,\n"
        f"                {mm}, {n}, {k},\n"
        f"                {alpha}, {blockA}, {ldA},\n"
        f"                {blockB}, {ldB},\n"
        f"                {beta}, {output_ptr}, {ldC});"
    )

new_src, cnt = PAT_SGEMM.subn(_build_cblas, src)
if cnt > 0:
    src = new_src
    changed = True
    print(f"  dnnl_sgemm → cblas_sgemm  ({cnt} occurrence(s))")
elif 'dnnl_sgemm' in src:
    # 找到了函数但正则没匹配——打印上下文帮助调试
    m = re.search(r'dnnl_sgemm\s*\(.*?\)\s*;', src, re.DOTALL)
    print("WARNING: dnnl_sgemm 找到了但正则未匹配，请把以下内容反馈给维护者：",
          file=sys.stderr)
    print(repr(m.group(0)) if m else "(no match object)", file=sys.stderr)
else:
    print("  dnnl_sgemm: not found in this TF version (skipped)")

# ── 4. dnnl_gemm_u8s8s32 → memset 置零（KML 无 int8 GEMM）─────────────────
PAT_U8S8 = re.compile(
    r'dnnl_status_t\s+st\s*=\s*dnnl_gemm_u8s8s32\s*\('
    r'.*?'
    r'\)\s*;'
    r'(?:\s*\n\s*eigen_assert\s*\(\s*st\s*==\s*0\s*\)\s*;)?',
    re.DOTALL
)
new_src, cnt = PAT_U8S8.subn(
    '// KBLAS has no int8 GEMM; produce zero output.\n'
    '    std::memset(C, 0, sizeof(int32_t) * size_t(m) * size_t(n));',
    src
)
if cnt > 0:
    src = new_src
    changed = True
    print(f"  dnnl_gemm_u8s8s32 → memset  ({cnt} occurrence(s))")
elif 'dnnl_gemm_u8s8s32' in src:
    print("WARNING: dnnl_gemm_u8s8s32 found but regex didn't match (non-fatal)",
          file=sys.stderr)
else:
    print("  dnnl_gemm_u8s8s32: not found (skipped)")

# ── 5. 清理悬空的 EIGEN_UNUSED_VARIABLE(st)（出现时才删）───────────────────
new_src = re.sub(r'\s*EIGEN_UNUSED_VARIABLE\s*\(\s*st\s*\)\s*;', '', src)
if new_src != src:
    src = new_src
    changed = True
    print("  EIGEN_UNUSED_VARIABLE(st) removed")

# ── 最终校验 ─────────────────────────────────────────────────────────────────
if 'dnnl_sgemm' in src:
    print("ERROR: dnnl_sgemm still present after patch!", file=sys.stderr)
    sys.exit(1)

if not changed:
    print("Nothing was changed (header may already be patched or different TF version)")
    sys.exit(0)

with open(header, 'w') as f:
    f.write(src)
print(f"OK: {header}")
PYEOF

echo ""
echo "Patch applied. Backup: ${HEADER}.bak_dnnl"
echo ""
echo "Now build:"
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
