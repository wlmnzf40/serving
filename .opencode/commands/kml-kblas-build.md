---
description: 鲲鹏 aarch64 上用华为 KML 的 cblas_sgemm 对比 Eigen 矩阵乘性能，运行时 --backend 切换
---

# KML KBLAS Build

`gemm_server` 是单个 binary，运行时用 `--backend=kblas|eigen` 切换 GEMM 实现，两条路径都直接
调用 `cblas_sgemm` / `Eigen::Map`，绕开 TF Session，**不需要 patch TF 内部头文件**。

在鲲鹏（aarch64/Kunpeng）机器上完成以下流程：
1. 获取 KML（vendor 到仓库内，无需 root）
2. 运行 `tools/setup_kblas.sh`（自动探测路径，修复 repo.bzl，写 .bazelrc 路径）
3. 编译一个 binary，验证 KBLAS 符号
4. 用 `--backend=kblas` / `--backend=eigen` 起两个实例，对比 GFLOPS

**重要约束**：TF 已编译过，不要 `bazel clean --expunge`。

---

## Step 1 — 获取 KML

### 推荐：vendor 到仓库内（无需 root，路径稳定）

```bash
cd /home/wanglimin/tf_serving

wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

mkdir -p third_party/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D third_party/kml
```

**找到 libkblas.so 的实际目录（记住，后面 setup 要用）：**
```bash
find third_party/kml -name "libkblas.so" 2>/dev/null
# 典型输出：third_party/kml/usr/local/kml/lib/kblas/omp/libkblas.so  ← OMP 版（推荐）
KML_LIB=$(dirname $(find third_party/kml -name "libkblas.so" | grep omp | head -1))
echo $KML_LIB   # 记住这个路径
```

---

## Step 2 — 一键 Setup

```bash
BAZEL=/home/wanglimin/bazel-7.4.1

# 自动探测 KML 路径（搜索 third_party/kml 和 /usr/local/kml）
bash tools/setup_kblas.sh $BAZEL

# 或手动指定 libkblas.so 所在目录
bash tools/setup_kblas.sh $BAZEL $KML_LIB
```

setup 脚本自动完成（每步幂等）：
- **repo.bzl 修复**：追加 `tf_serving_vendored`（缺失会报 `file does not contain symbol` 错误）
- **.bazelrc 路径**：用实际 KML lib 绝对路径写入 `-L` 和 `-rpath`

这两步是编译 `gemm_server --config=kml_kblas` 所需的全部准备工作。脚本还会尝试 patch
`eigen_contraction_kernel.h`，但那一步对这个 benchmark **是可选的**（仅用于让 TF 内部 MatMul
全局走 KBLAS，见文末「进阶」），跳过不影响 `gemm_server --backend=kblas` 正常工作。

---

## Step 3 — 编译

一个 binary，同时支持两种 backend：

```bash
bash tools/build_backends.sh $BAZEL
```

等价于：
```bash
BAZEL=/home/wanglimin/bazel-7.4.1
DISTDIR=/home/wanglimin/tf_new/dist
GCC_RPATH=/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64

$BAZEL build -c opt \
  --distdir=$DISTDIR \
  --define=no_cuda_support=true --define=no_nccl_support=true \
  --define=no_kafka_support=true --define=no_google_cloud_support=true \
  --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++ \
  --host_linkopt=-Wl,--disable-new-dtags \
  --host_linkopt=-Wl,-rpath,$GCC_RPATH \
  --linkopt=-Wl,--disable-new-dtags \
  --linkopt=-Wl,-rpath,$GCC_RPATH \
  --config=kml_kblas \
  //tf_serving_gemm/tf_gemm_server:gemm_server \
  //tf_serving_gemm/tf_gemm_server:gemm_client
```

**编译后必做验证：**
```bash
# cblas_sgemm 必须是 U（undefined 由 libkblas 提供）
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm

# ldd 必须找到 libkblas.so
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
```

不带 `--config=kml_kblas` 编译的话，二进制只有 Eigen 路径，`--backend=kblas` 会在启动时
直接报错退出（不会静默 fallback 到 Eigen）。

---

## Step 4 — 运行与对比

### 必须设置 LD_LIBRARY_PATH

KML 在非标准路径（如仓库 third_party 里），运行时动态链接器找不到：
```bash
export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH
```

### 一键对比（推荐）

同一个 binary 起两个实例，`--backend` 不同：

```bash
bash tools/compare_backends.sh              # sweep：方阵 128~2048
bash tools/compare_backends.sh shape_sweep  # 57 个生产 shapes
```

### 手动起两个实例

```bash
nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=kblas --addr=0.0.0.0:50052 > kblas.log 2>&1 &
sleep 2
cat kblas.log   # 必须有 "backend=kblas"

nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=eigen --addr=0.0.0.0:50053 > eigen.log 2>&1 &
sleep 2
cat eigen.log   # backend=eigen

./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50052 --mode=sweep
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50053 --mode=sweep
```

正常输出有实际数值行：
```
=== adx ===
MxKxN         cnt  avg_ms  GFLOPS
512x512x512    30   2.34   115.2
...
```

**如果只有表头没有数据行**（`=== adx ===` 下空白）：
```bash
cat server.log           # 看有无 "requires building with --config=kml_kblas" / SIGFPE / 段错误
nm -D bazel-bin/... | grep cblas_sgemm  # 确认 KBLAS 被链入
ldd bazel-bin/... | grep kblas          # 确认 so 能找到
```

---

## 常见问题速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `does not contain symbol 'tf_serving_vendored'` | repo.bzl 版本旧 | `setup_kblas.sh` 自动追加，或手动 `cat >>` |
| `--backend=kblas requires building with --config=kml_kblas` | 这个 binary 没带 KML 编译 | 重新 `bash tools/build_backends.sh`，或改用 `--backend=eigen` |
| `libkblas.so: cannot open` | LD_LIBRARY_PATH 未设 | `export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH` |
| client 输出无数据行 | server crash 或超时 | 看 server.log；检查 nm/ldd |
| `CONTENT_DOES_NOT_MATCH_TARGET` in fetch | 改了 WORKSPACE 触发 re-fetch | 不要改 WORKSPACE，只跑 setup_kblas.sh |

---

## 调用链（当前实现，无需 patch TF）

```
gemm_server --backend=kblas
  → direct_sgemm()  [server.cc]
    → cblas_sgemm(CblasRowMajor, ...)
      → libkblas.so  [鲲鹏 SVE/NEON 汇编]

gemm_server --backend=eigen
  → direct_sgemm()  [server.cc]
    → Eigen::Map<MatRM>.noalias() = eA * eB
      → Eigen 原生 GEBP 内核
```

两条路径都不经过 TF Session/Graph 执行，测的是纯 GEMM 时间，可直接比较 GFLOPS。

---

## 进阶（可选）：让 TF 内部 MatMul 全局走 KBLAS

上面的对比不需要这一步。如果想让 TF **任意**模型（不只是这个 benchmark）的
`MatMul`/`BatchMatMul` 都走 KBLAS，需要 patch Bazel cache 里的 Eigen 矩阵乘内核：

```bash
bash tools/apply_kblas_patch.sh $BAZEL
```

不改 WORKSPACE、不触发 TF re-fetch，正则匹配 + 内联前向声明（避免 `-I` 触发
hermetic 工具链的 "path outside of the execution root"），幂等可重复跑。

**如果提示 "patch deferred"**：
```bash
$BAZEL build -c opt --distdir=/home/wanglimin/tf_new/dist \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | tail -5
bash tools/apply_kblas_patch.sh $BAZEL
```

Patch 后的调用链：
```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]  ← patch 后
      → cblas_sgemm(CblasColMajor, ...)
        → libkblas.so
```
