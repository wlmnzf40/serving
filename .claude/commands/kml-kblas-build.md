# KML KBLAS Build Skill
# 用途：在鲲鹏（aarch64）机器上，用华为 KML 的 cblas_sgemm 对比 Eigen 矩阵乘性能

## 概述

`gemm_server` 是单个 binary，内置两条 GEMM 实现，**运行时**用 `--backend=kblas|eigen` 切换：
- `--backend=kblas` → 直接调用 `cblas_sgemm`（libkblas.so）
- `--backend=eigen` → 直接调用 `Eigen::Map` 矩阵乘

两条路径都绕开 TF Session::Run，在 server.cc 里直接分发，所以**不需要 patch TF 内部的
`eigen_contraction_kernel.h`**，也不需要改 WORKSPACE、不会触发 TF re-fetch。

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

**找到 libkblas.so 的实际目录（记住，后面要用）：**
```bash
find third_party/kml -name "libkblas.so" 2>/dev/null
# 典型输出：third_party/kml/usr/local/kml/lib/kblas/omp/libkblas.so  ← OMP 版（推荐）
KML_LIB=$(dirname $(find third_party/kml -name "libkblas.so" | grep omp | head -1))
echo $KML_LIB   # 记住这个路径
nm -D $KML_LIB/libkblas.so | grep cblas_sgemm   # 必须有输出
```

### 方式 B：系统安装（需要 sudo）

```bash
sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm
# 安装到 /usr/local/kml/lib/
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

setup 脚本完成：
- **repo.bzl 修复**：追加 `tf_serving_vendored`（缺失会报 `file does not contain symbol` 错误）
- **.bazelrc 路径**：用实际 KML lib 绝对路径写入 `-L` 和 `-rpath`

这两步是编译 `gemm_server --config=kml_kblas` 所必需的全部准备工作。脚本还会尝试 patch
`eigen_contraction_kernel.h`，但**这一步对本 benchmark 是可选的**（见文末「进阶」），跳过/失败都不影响
`gemm_server` 的 `--backend=kblas` 正常工作。

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

**编译后必做的验证（若 cblas_sgemm 不在，说明宏或链接没生效）：**
```bash
# ① cblas_sgemm 符号必须是 U（undefined，由 libkblas 提供）
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 期望：  U cblas_sgemm

# ② 动态库链接必须找到 libkblas.so
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
# 期望：  libkblas.so => /path/to/kml/lib/.../libkblas.so

# ③ 如果 ldd 报 "not found"，说明 rpath 没烧进去，需要用 LD_LIBRARY_PATH（见 Step 4）
```

如果**不**带 `--config=kml_kblas` 编译（普通 `bazel build`），二进制只有 Eigen 路径，
`--backend=kblas` 会在启动时直接报错退出（不会 fallback 静默跑 Eigen）。

---

## Step 4 — 运行与对比

### 关键：设置 LD_LIBRARY_PATH

KML 在非标准路径时，**即使编译时烧了 rpath，运行时仍可能找不到 `.so`**：
```bash
export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH
```

### 方式 A：一键对比脚本（推荐）

同一个 binary 起两个实例，`--backend` 不同，自动跑 benchmark 并打印两边结果：

```bash
bash tools/compare_backends.sh              # sweep：方阵 128~2048
bash tools/compare_backends.sh shape_sweep  # 57 个生产 shapes
bash tools/compare_backends.sh compute      # 单次 512x512 往返
```

### 方式 B：手动起两个实例

```bash
nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=kblas --addr=0.0.0.0:50052 > kblas.log 2>&1 &
sleep 2
cat kblas.log   # 必须有 "TF GEMM server  addr=... backend=kblas"

nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=eigen --addr=0.0.0.0:50053 > eigen.log 2>&1 &
sleep 2
cat eigen.log   # backend=eigen

./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50052 --mode=sweep
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50053 --mode=sweep
```

### 检查结果是否有效

正常输出有实际数值行：
```
=== adx ===
MxKxN         cnt  avg_ms  p50_ms  p99_ms  GFLOPS
----------------------------------------------
512x512x512    30   2.34    2.31    2.89    115.2
...
```

**如果 GFLOPS 栏全空或只有表头没有行**：
- 检查 server 日志有没有 `--backend=kblas requires building with --config=kml_kblas`
  （说明这个 binary 是不带 KML 编译的，但传了 `--backend=kblas`）
- 检查 `cat server.log` 有无 SIGFPE/段错误（KBLAS 内部问题，通常是矩阵很小时的边界条件）
- 检查 `nm -D gemm_server | grep cblas_sgemm` 和 `ldd gemm_server | grep kblas`

---

## 常见问题速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `does not contain symbol 'tf_serving_vendored'` | repo.bzl 版本旧 | `setup_kblas.sh` 自动追加，或手动 `cat >>` |
| `--backend=kblas requires building with --config=kml_kblas` | 用的是不带 KML 的 binary | 重新 `bash tools/build_backends.sh`，或改用 `--backend=eigen` |
| `libkblas.so: cannot open shared object file` | LD_LIBRARY_PATH 未设 | `export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH` |
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

## 进阶（可选）：让 TF 内部的 MatMul/Eigen 全局走 KBLAS

上面的 benchmark 不需要这一步。但如果你想让 TF **任意**模型里的 `MatMul`/`BatchMatMul`
（不只是这个 benchmark server）都走 KBLAS，需要 patch TF 内部的 Eigen 矩阵乘内核：

```bash
bash tools/apply_kblas_patch.sh $BAZEL
```

这个脚本就地修改 Bazel cache 里的 `eigen_contraction_kernel.h`：
- 把 `dnnl_sgemm` 调用替换成 `cblas_sgemm`（正则匹配，容忍 TF 版本间的小差异）
- 内联 `cblas_sgemm` 前向声明，不用 `#include "kblas.h"`，避免 hermetic 工具链的
  `-I` 路径限制（`path outside of the execution root`）
- 不改 WORKSPACE，不触发 TF re-fetch，幂等可重复跑

**如果提示 "patch deferred"**（头文件还没在 Bazel cache 里）：
```bash
$BAZEL build -c opt --distdir=/home/wanglimin/tf_new/dist \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | tail -5
bash tools/apply_kblas_patch.sh $BAZEL
```

Patch 后任何用 `--config=kml_kblas` 编译、且代码路径会触发 Eigen `TensorContraction`
（包括 TF 自己的 MatMul kernel）的程序都会走 KBLAS：

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]  ← patch 后
      → cblas_sgemm(CblasColMajor, ...)
        → libkblas.so
```

Eigen 以列主序打包矩阵块（col-major panels），`cblas_sgemm(CblasColMajor)` 原生支持列主序，
无需像 `dnnl_sgemm`（行主序约定）那样对调 A/B 和 M/N 参数。

**dnnl_sgemm pattern mismatch**（patch 脚本警告）：脚本会打印实际找到的代码片段（repr 格式），
TF 版本略有不同时正则可能失配，把实际内容反馈给维护者即可更新 `apply_kblas_patch.sh` 里的正则。
