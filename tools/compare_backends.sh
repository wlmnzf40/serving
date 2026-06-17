#!/usr/bin/env bash
# compare_backends.sh — 同一个 binary，--backend=kblas 和 --backend=eigen 各跑一次，打印 GFLOPS 对比
#
# 用法：
#   bash tools/compare_backends.sh [MODE] [EXTRA_CLIENT_FLAGS...]
#
#   MODE:
#     shape_sweep  (默认) 生产 shapes（57 个小矩阵，adx/cvr/hmv/presort）——
#                  验证结论必须以这个模式为准，sweep 只是直观对比工具
#     sweep        方矩阵 128~2048
#     compute      单次 512×512 往返
#
# 示例：
#   bash tools/compare_backends.sh
#   bash tools/compare_backends.sh sweep --sizes=256,512,1024,2048 --iters=50
#   bash tools/compare_backends.sh shape_sweep --iters=30 --warmup=20
#
# 小矩阵诊断提示：
#   如果 KBLAS 在小矩阵上的 avg_ms 远高于 Eigen（GFLOPS 远低于 Eigen），先看新增的
#   min_ms 列：min_ms 也接近 avg_ms（而不是远低于）说明这是每次调用都有的稳定开销，
#   不是个别离群点。最可能的原因是 KML 的 OMP 变体在调用间隔之间把线程组 park 掉了，
#   下一次 cblas_sgemm 触发的 OpenMP parallel region 要先把线程唤醒（fork/join +
#   futex wake），这个延迟在大矩阵上被计算时间摊薄，在小矩阵上就是主要开销。
#   下面给 KBLAS 实例设置 OMP_WAIT_POLICY=active + GOMP_SPINCOUNT 让线程组保持自旋、
#   不进入睡眠，用来验证/缓解这个假设——这是 benchmark 进程的环境变量，不是改生产
#   代码；在真实鲲鹏机器上重跑这个脚本确认是否改善。

set -euo pipefail

MODE="${1:-shape_sweep}"
shift 2>/dev/null || true   # 剩余参数传给 client

KML_LIB="${KML_LIB:-}"

# 自动探测 KML lib 路径
if [[ -z "$KML_LIB" ]]; then
    for d in \
        "$(pwd)/third_party/kml/lib/kblas/omp" \
        "$(pwd)/third_party/kml/lib/kblas" \
        "$(pwd)/third_party/kml/lib" \
        "/usr/local/kml/lib"
    do
        if [[ -f "$d/libkblas.so" ]]; then
            KML_LIB="$d"; break
        fi
    done
fi

if [[ -z "$KML_LIB" ]]; then
    echo "WARNING: libkblas.so not found. KBLAS backend may crash."
    echo "  Set KML_LIB=/path/to/dir/containing/libkblas.so before running."
fi

SERVER="./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server"
CLIENT="./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client"

PORT_KBLAS=50052
PORT_EIGEN=50053

if [[ ! -f "$SERVER" ]]; then
    echo "ERROR: $SERVER not found. Run: bash tools/build_backends.sh"
    exit 1
fi
if [[ ! -f "$CLIENT" ]]; then
    echo "ERROR: $CLIENT not found"
    exit 1
fi

cleanup() {
    echo ""
    echo "Stopping servers..."
    [[ -n "${PID_KBLAS:-}" ]] && kill "$PID_KBLAS" 2>/dev/null || true
    [[ -n "${PID_EIGEN:-}" ]] && kill "$PID_EIGEN"  2>/dev/null || true
    rm -f /tmp/kblas_server.log /tmp/eigen_server.log
}
trap cleanup EXIT

# ── 启动两个实例：同一个 binary，不同 --backend ──────────────────────────────
#
# OMP_WAIT_POLICY=active + GOMP_SPINCOUNT：只给 KBLAS 实例设置。libkblas.so(omp
# 变体) 内部用 GNU OpenMP(libgomp) 并行；libgomp 默认在线程空闲一段时间后把它们
# park 掉（pthread_cond_wait），下次 cblas_sgemm 进入 parallel region 要先唤醒，
# 这个 fork/join + futex wake 延迟在小矩阵 benchmark 里会被放大成可见的每次调用
# 固定开销。ACTIVE 让线程组保持自旋等待，避免这次唤醒。Eigen 路径(tf_serving_
# eigen_sgemm)不经过 libgomp，这两个变量对它是无操作，所以即使全局导出也无害，
# 但只在这里设置能让"为什么设置它"的意图保持清晰。
echo "Starting KBLAS backend  (port $PORT_KBLAS)..."
LD_LIBRARY_PATH="${KML_LIB:+$KML_LIB:}${LD_LIBRARY_PATH:-}" \
    OMP_WAIT_POLICY=active \
    GOMP_SPINCOUNT="${GOMP_SPINCOUNT:-30000000000}" \
    "$SERVER" --addr="0.0.0.0:$PORT_KBLAS" --backend=kblas \
    > /tmp/kblas_server.log 2>&1 &
PID_KBLAS=$!

echo "Starting Eigen backend  (port $PORT_EIGEN)..."
LD_LIBRARY_PATH="${KML_LIB:+$KML_LIB:}${LD_LIBRARY_PATH:-}" \
    "$SERVER" --addr="0.0.0.0:$PORT_EIGEN" --backend=eigen \
    > /tmp/eigen_server.log 2>&1 &
PID_EIGEN=$!

echo "Waiting for servers to initialize..."
sleep 2

# 检查两个实例是否还在跑
if ! kill -0 "$PID_KBLAS" 2>/dev/null; then
    echo "ERROR: KBLAS server crashed. Log:"
    cat /tmp/kblas_server.log
    exit 1
fi
if ! kill -0 "$PID_EIGEN" 2>/dev/null; then
    echo "ERROR: Eigen server crashed. Log:"
    cat /tmp/eigen_server.log
    exit 1
fi

echo "Both servers running."
echo ""

# ── 设置 client 参数 ──────────────────────────────────────────────────────────
# warmup 不低于 server 端的默认值(Sweep=10, ShapeSweep=20)——之前这里把两个模式都
# 压到了 --warmup=5，比 server 自己的默认还低，等于主动削弱了 warmup。如果延迟是
# 稳定的每次调用开销（用新加的 min_ms 列确认：min_ms 接近 avg_ms），加多少 warmup
# 都不会让它消失，因为测量窗口本身也会摊上这个开销；warmup 在这里只是用来排除
# "纯冷启动一次性开销"这个备选解释，不是修复手段本身（修复手段见上面的 OMP 变量）。
case "$MODE" in
    sweep)
        MODE_FLAGS=(--mode=sweep --sizes=128,256,512,1024,2048 --iters=30 --warmup=10)
        ;;
    shape_sweep)
        MODE_FLAGS=(--mode=shape_sweep --iters=30 --warmup=20)
        ;;
    compute)
        MODE_FLAGS=(--mode=compute --M=512 --K=512 --N=512 --iters=50 --warmup=10)
        ;;
    *)
        echo "Unknown mode: $MODE. Use sweep|shape_sweep|compute"
        exit 1
        ;;
esac

# 追加用户额外参数（覆盖默认值）
MODE_FLAGS+=("$@")

# ── 运行 benchmark ────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "  KBLAS backend  (port $PORT_KBLAS)"
echo "════════════════════════════════════════"
"$CLIENT" --host="localhost:$PORT_KBLAS" "${MODE_FLAGS[@]}"

echo ""
echo "════════════════════════════════════════"
echo "  Eigen backend  (port $PORT_EIGEN)"
echo "════════════════════════════════════════"
"$CLIENT" --host="localhost:$PORT_EIGEN" "${MODE_FLAGS[@]}"

echo ""
echo "════════════════════════════════════════"
echo "  Done. Compare GFLOPS above."
echo "════════════════════════════════════════"
