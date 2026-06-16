# KML KBLAS Build Skill
# 用途：在鲲鹏（aarch64）机器上，用华为 KML 替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server

## 概述

本 Skill 覆盖从 KML 安装到运行验证的完整流程，包含真实遇到的所有坑。

核心思路（不改 WORKSPACE，不触发 TF re-fetch）：
1. KML 可 vendor 到 `third_party/kml/`（无需 root），也可系统安装
2. `tools/setup_kblas.sh` 自动探测 KML 路径，修复 `repo.bzl`，更新 `.bazelrc`，patch 头文件
3. `--config=kml_kblas` 激活宏，`-lkblas` 链接库，内联前向声明代替 `#include "kblas.h"`

**重要约束**：TF 已编译过，不要 `bazel clean --expunge`。

---

## Step 1 — 获取 KML

### 方式 A：vendor 到仓库内（推荐，无需 root，路径稳定）

```bash
cd /home/wanglimin/tf_serving

# 下载 RPM
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

# 解压到仓库的 third_party/kml/
mkdir -p third_party/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D third_party/kml
```

**验证目录结构**（重要，lib 子路径会因版本不同而异）：
```bash
find third_party/kml -name "libkblas.so" 2>/dev/null
# 典型输出之一：
#   third_party/kml/usr/local/kml/lib/kblas/omp/libkblas.so  ← OMP 版（推荐）
#   third_party/kml/usr/local/kml/lib/kblas/libkblas.so       ← 非 OMP 版
```

### 方式 B：系统安装（需要 sudo）

```bash
sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm
# 安装到 /usr/local/kml/lib/
```

**验证（任意方式）：**
```bash
# 找到 libkblas.so 的实际目录，记下来（后面要用）
KML_LIB=$(dirname $(find third_party/kml /usr/local/kml -name "libkblas.so" 2>/dev/null | grep omp | head -1))
echo "KML_LIB=$KML_LIB"
nm -D $KML_LIB/libkblas.so | grep cblas_sgemm   # 必须有输出
```

---

## Step 2 — 一键 Setup

```bash
BAZEL=/home/wanglimin/bazel-7.4.1
cd /home/wanglimin/tf_serving

# setup_kblas.sh 会自动探测 KML lib 路径（按优先级搜索）
bash tools/setup_kblas.sh $BAZEL

# 或者手动指定 lib 目录（包含 libkblas.so 的那一级）
bash tools/setup_kblas.sh $BAZEL /home/wanglimin/tf_serving/third_party/kml/usr/local/kml/lib/kblas/omp
```

脚本自动完成以下操作（每步幂等）：

| 步骤 | 做什么 | 失败时提示 |
|------|--------|-----------|
| **1. repo.bzl 修复** | 检测并追加 `tf_serving_vendored` | 自动修复 |
| **2. .bazelrc 路径** | 用实际 KML lib 绝对路径替换 `-L` 和 `-rpath` | 打印手动操作命令 |
| **3. 头文件 patch** | 在 Bazel 缓存就地替换 `dnnl_sgemm → cblas_sgemm`（正则匹配） | 提示先跑一次普通 build |

**如果提示 "patch deferred"**（头文件不在缓存）：
```bash
# 先不加 kml_kblas 触发一次 TF 解压
$BAZEL build -c opt --distdir=/home/wanglimin/tf_new/dist \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | tail -5
# 然后重跑 setup
bash tools/setup_kblas.sh $BAZEL
```

---

## Step 3 — 编译

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

**编译后必做的验证（若 cblas_sgemm 不在，说明 patch 或宏没生效）：**
```bash
# ① cblas_sgemm 符号必须是 U（undefined，由 libkblas 提供）
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 期望：  U cblas_sgemm

# ② 动态库链接必须找到 libkblas.so
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
# 期望：  libkblas.so => /path/to/kml/lib/.../libkblas.so

# ③ 如果 ldd 报 "not found"，说明 rpath 没烧进去，需要用 LD_LIBRARY_PATH（见 Step 4）
```

---

## Step 4 — 运行与测试

### 关键：设置 LD_LIBRARY_PATH

KML 在非标准路径时，**即使编译时烧了 rpath，运行时仍可能找不到 `.so`**（动态链接器优先级问题）。
建议始终显式设置：

```bash
KML_LIB=/home/wanglimin/tf_serving/third_party/kml/usr/local/kml/lib/kblas/omp
# 或者：KML_LIB=/usr/local/kml/lib

export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH
```

### 启动 server（后台运行，日志重定向）

```bash
nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --addr=0.0.0.0:50052 > server.log 2>&1 &
SERVER_PID=$!

# 等待启动（必须等，TF 初始化约 2-3 秒）
sleep 3
cat server.log   # 确认有 "TF GEMM server on 0.0.0.0:50052" 输出
```

### 运行 client 测试

```bash
# 快速验证
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=30 --warmup=5

# 稳定测试
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=200 --warmup=50

kill $SERVER_PID
```

### 检查结果是否有效

正常的输出应该有实际的数值行：
```
=== adx ===
MxKxN         cnt  avg_ms  p50_ms  p99_ms  GFLOPS
----------------------------------------------
512x512x512    30   2.34    2.31    2.89    115.2
...
```

**如果 GFLOPS 栏全空或只有表头没有行**：
- 原因 A：KBLAS 没生效，fallback 到 Eigen，计算极慢导致 gRPC 超时
  - 检查：`nm -D gemm_server | grep cblas_sgemm`（应有 U 符号）
  - 检查：`ldd gemm_server | grep kblas`（应找到 so）
- 原因 B：server.log 有 SIGFPE 或段错误（KBLAS 内部问题）
  - 检查：`cat server.log`

---

## 常见问题

### `tf_serving_vendored` 找不到（编译报错）
`setup_kblas.sh` 自动修复。手动修复：
```bash
cat >> tensorflow_serving/repo.bzl << 'EOF'

def _tf_serving_vendored_impl(ctx):
    ctx.symlink(ctx.path(ctx.attr.root).dirname.get_child(ctx.attr.path), ".")

tf_serving_vendored = repository_rule(
    implementation = _tf_serving_vendored_impl,
    attrs = {
        "root": attr.label(mandatory = True),
        "path": attr.string(mandatory = True),
    },
)
EOF
```

### `The include path '...' references a path outside of the execution root`
**已修复**：新版 patch 脚本不使用 `#include "kblas.h"`，直接内联前向声明，无需 `-I` 标志。
确保使用最新版 `tools/apply_kblas_patch.sh`（无 `-I` flag in `.bazelrc`）。

### `tensor_testutil.cc: CONTENT_DOES_NOT_MATCH_TARGET`（fetch 时）
不要改 WORKSPACE（会触发 re-fetch）。只跑 `tools/setup_kblas.sh`。

### `libkblas.so: cannot open shared object file`（运行时）
```bash
export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH
# 永久：echo "$KML_LIB" | sudo tee /etc/ld.so.conf.d/kml.conf && sudo ldconfig
```

### server 启动后 client 输出全空（没有 GFLOPS 数值）
```bash
# 1. 确认 KBLAS 符号存在
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm

# 2. 确认 server 日志无报错
cat server.log

# 3. 确认 client 超时时间足够（首次有 JIT 开销）
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=5 --warmup=2
```

### dnnl_sgemm pattern mismatch（patch 脚本警告）
脚本会打印实际找到的代码片段（repr 格式），TF 版本略有不同时正则可能失配。
把实际内容反馈给维护者，更新 `apply_kblas_patch.sh` 里的正则即可。

---

## 调用链

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]
      → cblas_sgemm(CblasColMajor, ...)   ← patch 激活后
        → libkblas.so  [鲲鹏 SVE/NEON 汇编]
```

Eigen 以列主序打包矩阵块（col-major panels），`cblas_sgemm(CblasColMajor)` 原生支持列主序，
无需像 `dnnl_sgemm`（行主序约定）那样对调 A/B 和 M/N 参数。
