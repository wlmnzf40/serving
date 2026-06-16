---
description: 鲲鹏 aarch64 上用华为 KML 替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server
---

# KML KBLAS Build

在鲲鹏（aarch64/Kunpeng）机器上完成以下流程，覆盖真实遇到的所有坑：
1. 获取 KML（vendor 到仓库内，无需 root）
2. 运行 `tools/setup_kblas.sh`（自动探测路径，修复 repo.bzl，patch 头文件）
3. 编译，验证 KBLAS 符号，运行测试

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
- **头文件 patch**：`dnnl_sgemm → cblas_sgemm` 内联前向声明（无需 `-I`，避免 hermetic 工具链报 "path outside of execution root"）

**如果提示 "patch deferred"**：
```bash
$BAZEL build -c opt --distdir=/home/wanglimin/tf_new/dist \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | tail -5
bash tools/setup_kblas.sh $BAZEL $KML_LIB
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

**编译后必做验证：**
```bash
# cblas_sgemm 必须是 U（undefined 由 libkblas 提供）
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm

# ldd 必须找到 libkblas.so
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
```

---

## Step 4 — 运行

### 必须设置 LD_LIBRARY_PATH

KML 在非标准路径（如仓库 third_party 里），运行时动态链接器找不到：
```bash
export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH
```

### 启动 server

```bash
nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --addr=0.0.0.0:50052 > server.log 2>&1 &

sleep 3
cat server.log   # 必须有 "TF GEMM server on 0.0.0.0:50052" 才算启动成功
```

### 运行测试

```bash
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=30 --warmup=5
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=200 --warmup=50
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
cat server.log           # 看有无 SIGFPE/段错误
nm -D bazel-bin/... | grep cblas_sgemm  # 确认 KBLAS 被链入
ldd bazel-bin/... | grep kblas          # 确认 so 能找到
```

---

## 常见问题速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `does not contain symbol 'tf_serving_vendored'` | repo.bzl 版本旧 | `setup_kblas.sh` 自动追加，或手动 `cat >>` |
| `path outside of the execution root` | 旧版本用了 `-I` flag | 换新版 patch 脚本（用内联前向声明） |
| `CONTENT_DOES_NOT_MATCH_TARGET` in fetch | tensorflow.patch 版本问题 | 不要改 WORKSPACE，只跑 setup_kblas.sh |
| `libkblas.so: cannot open` | LD_LIBRARY_PATH 未设 | `export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH` |
| client 输出无数据行 | KBLAS 未生效或超时 | 检查 nm/ldd，看 server.log |

---

## 调用链

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]  ← patch 后
      → cblas_sgemm(CblasColMajor, ...)
        → libkblas.so  [鲲鹏 SVE/NEON 汇编]
```
