# KML KBLAS Build Skill
# 用途：在鲲鹏（aarch64）机器上，用华为 KML 的 cblas_sgemm 对比 Eigen 矩阵乘性能

## 概述

`gemm_server` 是单个 binary，**运行时**用 `--backend=kblas|eigen` 切换：
- `--backend=kblas` → TF 内核最底层调用 `cblas_sgemm`（libkblas.so）
- `--backend=eigen` → TF 内核最底层调用 Eigen 原生实现

两者走的是**完全相同**的 TF 执行路径：

```
ClientSession::Run(MatMul/BatchMatMul) → OpKernel::Compute()
  → Eigen::Tensor::contract() → eigen_contraction_kernel.h
```

只在这条路径最底层的 GEMM micro-kernel 调用上二选一。`--backend` 标志做的事情只是
设置一个 `extern "C" int tf_serving_kblas_enabled` 全局开关（定义在 `server.cc`，
在 `tools/apply_kblas_patch.sh` patch 过的 `eigen_contraction_kernel.h` 里被读取），
**不会**绕开 TF Session 另起一条 `Eigen::Map` 调用——那样测的就不是 TF 真实执行路径了。

**关键含义**：`eigen_contraction_kernel.h` 的 patch 步骤（Step 2）是**必需的**，
不是可选项。这个开关本身就是 patch 加进 TF 内核里的东西；没 patch 过，
`--backend=kblas|eigen` 这个运行时选择在 TF 内部根本不存在。

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

## Step 2 — 一键 Setup（含必需的头文件 patch）

```bash
BAZEL=/home/wanglimin/bazel-7.4.1

# 自动探测 KML 路径（搜索 third_party/kml 和 /usr/local/kml）
bash tools/setup_kblas.sh $BAZEL

# 或手动指定 libkblas.so 所在目录
bash tools/setup_kblas.sh $BAZEL $KML_LIB
```

setup 脚本完成（每步幂等）：
1. **repo.bzl 修复**：追加 `tf_serving_vendored`（缺失会报 `file does not contain symbol` 错误）
2. **.bazelrc 路径**：用实际 KML lib 绝对路径写入 `-L` 和 `-rpath`
3. **patch `eigen_contraction_kernel.h`**（**必需**——`--backend` 运行时开关就是这一步加进去的）

如果脚本提示 "patch deferred"（Bazel cache 里还没有这个头文件），按提示先跑一次不带
`--config=kml_kblas` 的普通 build 让 Bazel 解压 TF，再重新执行本脚本或直接跑：
```bash
bash tools/apply_kblas_patch.sh $BAZEL
```

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

如果**不**带 `--config=kml_kblas` 编译（普通 `bazel build`），二进制里 MatMul 走的是
TF 自己原本的 Eigen 原生内核（没有 patch 过的代码路径），`--backend=kblas` 会在启动时
直接报错退出（不会 fallback 静默跑 Eigen）。

如果带了 `--config=kml_kblas` 编译，但 Step 2 的头文件 patch 没有真正生效（比如 Bazel
cache 被重新解压过、或 patch 脚本报过 WARNING），`cblas_sgemm` 符号仍然可能不存在——
**先看 nm/ldd，再开始对比**，不要假设编译成功就等于 patch 生效。

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

**如果两个 backend 的 GFLOPS 几乎一样（怀疑开关没生效）**：
- 大概率是头文件 patch 没真正应用——重新跑 `bash tools/apply_kblas_patch.sh $BAZEL`，
  确认输出里没有 `WARNING: dnnl_sgemm found but regex didn't match`，然后重新编译。
- TF 版本升级后 `dnnl_sgemm(...)` 调用点的格式可能变了，导致正则没匹配上，patch
  脚本会把实际代码片段打印到 stderr，反馈给维护者更新正则即可。

---

## 常见问题速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `does not contain symbol 'tf_serving_vendored'` | repo.bzl 版本旧 | `setup_kblas.sh` 自动追加，或手动 `cat >>` |
| `--backend=kblas requires building with --config=kml_kblas` | 用的是不带 KML 编译的 binary | 重新 `bash tools/build_backends.sh`，或改用 `--backend=eigen` |
| `libkblas.so: cannot open shared object file` | LD_LIBRARY_PATH 未设 | `export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH` |
| client 输出无数据行 | server crash 或超时 | 看 server.log；检查 nm/ldd |
| 两个 backend GFLOPS 几乎相同 | 头文件 patch 没生效（开关没编译进内核） | 重跑 `apply_kblas_patch.sh`，检查 WARNING，重新编译 |
| `CONTENT_DOES_NOT_MATCH_TARGET` in fetch | 改了 WORKSPACE 触发 re-fetch | 不要改 WORKSPACE，只跑 setup_kblas.sh |

---

## 调用链

```
gemm_server --backend=kblas|eigen
  → main() 设置 tf_serving_kblas_enabled = 1|0   [server.cc]
  → ClientSession::Run(MatMul / BatchMatMul)      [server.cc：GEMMRunner / BatchGEMMRunner]
    → MatMulOp / BatchMatMulOp::Compute()          [TF 内核，all_kernels]
      → Eigen::Tensor::contract()
        → ParallelMatMulKernel                     [eigen_contraction_kernel.h，patch 后]
          if (tf_serving_kblas_enabled)
            → cblas_sgemm(CblasColMajor, ...) → libkblas.so   [鲲鹏 SVE/NEON 汇编]
          else
            → tf_serving_eigen_sgemm(...)    → Eigen::Map（同一头文件内联，TF 原生 fallback）
```

从 `ClientSession::Run()` 到 `Eigen::Tensor::contract()` 这一整条路径，两个 backend
**完全一致**；只有最后一步的 GEMM micro-kernel 调用不同。这是 `--backend=eigen` 能
代表"原始 TF 执行路径"的关键——它不是另起一个 `Eigen::Map` 计算，而是patch 后
contraction kernel 在禁用 KBLAS 时退回到的、与 TF 自身 Eigen 路径同构的实现。

---

## 实现细节：patch 做了什么

`tools/apply_kblas_patch.sh` 就地修改 Bazel cache 里的 `eigen_contraction_kernel.h`：

1. **宏重命名**：`TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` → `..._KML`，避免和 TF
   原生 oneDNN 路径的宏冲突。
2. **`#include "dnnl.h"` → 内联声明**：替换成 `cblas_sgemm` 的 `extern "C"` 前向声明
   （避免 `-I` 触发 hermetic 工具链的 `path outside of the execution root`），加上：
   - `extern int tf_serving_kblas_enabled;` —— 运行时开关，定义在 `server.cc`
   - `tf_serving_eigen_sgemm(...)` —— 与 `cblas_sgemm` 同签名的 Eigen `Map` 实现，
     `tf_serving_kblas_enabled == 0` 时调用，保证 `--backend=eigen` 和
     `--backend=kblas` 在 contraction kernel 这一层之前完全同路径。
3. **`dnnl_sgemm(...)` 调用点 → 运行时 if/else**：正则匹配（容忍 TF 版本间的换行/缩进
   差异），生成 `if (tf_serving_kblas_enabled) cblas_sgemm(...); else tf_serving_eigen_sgemm(...);`，
   按 `CblasColMajor` 自然顺序重排参数（dnnl 是行主序换 A/B、M/N 的约定，cblas 列主序
   不需要那一步）。
4. **`dnnl_gemm_u8s8s32(...)` → memset**：KML 没有 int8 GEMM，这条路径直接清零输出
   （这个 benchmark 只测 fp32，int8 路径只是让代码能编译过）。
5. 清理悬空的 `EIGEN_UNUSED_VARIABLE(st)`。

不改 WORKSPACE，不触发 TF re-fetch，幂等可重复跑。

**dnnl_sgemm pattern mismatch**（patch 脚本警告）：脚本会打印实际找到的代码片段（repr 格式），
TF 版本略有不同时正则可能失配，把实际内容反馈给维护者即可更新 `apply_kblas_patch.sh` 里的正则。
