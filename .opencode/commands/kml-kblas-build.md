---
description: 鲲鹏 aarch64 上用华为 KML 替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server
---

# KML KBLAS Build

在鲲鹏（aarch64/Kunpeng）机器上完成以下三件事：
1. 下载安装华为 KML（鲲鹏数学库）
2. 确认仓库已预置 WORKSPACE patch 和 .bazelrc config
3. 编译 gemm_server / gemm_client 并验证 KBLAS 替换生效

**重要约束**：TF 已编译过，不要执行 `bazel clean --expunge`，增量编译即可。

---

## Step 1 — 下载并安装 KML

```bash
# 下载 RPM（~40 MB）
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

# 方式 A：有 sudo，直接安装到 /usr/local/kml/
sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm

# 方式 B：无 root，解压到用户目录
mkdir -p $HOME/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D $HOME/kml
# 解压后找到实际子目录，例如 $HOME/kml/usr/local/kml/
```

**验证：**
```bash
KML_ROOT=/usr/local/kml          # 方式 B 改为 $HOME/kml/usr/local/kml
ls $KML_ROOT/include/kblas.h     # 必须存在
ls $KML_ROOT/lib/libkblas.so     # 必须存在
nm -D $KML_ROOT/lib/libkblas.so | grep cblas_sgemm   # 必须有输出
```

---

## Step 2 — 确认仓库 patch 已就位（只读，不需修改）

### 2-A 检查 WORKSPACE patch

```bash
grep -c "KBLAS_PATCH_EOF" WORKSPACE   # 应输出 2（开头+结尾各一次）
```

WORKSPACE 里的 `patch_cmds` Python 脚本在 Bazel 首次解压 TF 源码时自动执行，将
`third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h` 里的
`dnnl_sgemm(transposeB, transposeA, n, m, k, ...)` 替换为
`cblas_sgemm(CblasColMajor, transposeA, transposeB, m, n, k, ...)`。

### 2-B 检查 .bazelrc config

```bash
grep -A 12 "^build:kml_kblas" .bazelrc
```

预期关键行：
```
build:kml_kblas --copt=-DTENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL
build:kml_kblas --copt=-DTENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
build:kml_kblas --copt=-I/usr/local/kml/include
build:kml_kblas --linkopt=-L/usr/local/kml/lib
build:kml_kblas --linkopt=-lkblas
build:kml_kblas --linkopt=-Wl,-rpath,/usr/local/kml/lib
```

**如果 KML 装在非默认路径**，执行（把 `/home/wanglimin/kml` 改成实际路径）：
```bash
KML_ACTUAL=$HOME/kml/usr/local/kml
sed -i "s|/usr/local/kml/include|$KML_ACTUAL/include|g" .bazelrc
sed -i "s|/usr/local/kml/lib|$KML_ACTUAL/lib|g"         .bazelrc
```

---

## Step 3 — 编译命令

> 把下面的路径变量替换成你机器上的实际值：

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

### 编译后验证 KBLAS 确实链入

```bash
# 应看到：U cblas_sgemm
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm

# 应看到：libkblas.so => /usr/local/kml/lib/libkblas.so
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
```

---

## Step 4 — 运行测试

```bash
# 启动 server
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --addr=0.0.0.0:50052 &
SERVER_PID=$!
sleep 2

# 快速验证
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=30 --warmup=5

# 稳定性测试（和原来用法完全一样）
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=200 --warmup=50

kill $SERVER_PID
```

---

## 常见问题

### kblas.h: No such file or directory
```bash
find /usr/local/kml $HOME/kml -name "kblas.h" 2>/dev/null
# 把找到的路径填入 .bazelrc 的 --copt=-I<path>
```

### libkblas.so: cannot open shared object file（运行时）
```bash
# 临时
export LD_LIBRARY_PATH=/usr/local/kml/lib:$LD_LIBRARY_PATH
# 永久
echo "/usr/local/kml/lib" | sudo tee /etc/ld.so.conf.d/kml.conf && sudo ldconfig
```

### 性能没有提升（KBLAS 未生效）
```bash
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 无输出说明宏未激活，检查 --config=kml_kblas 是否传入了两个 -D 宏
```

### patch 报 `dnnl_sgemm pattern not found`（首次 build 时）
```bash
# 找到 Bazel 沙箱里实际的文件，对比字符串
find $(${BAZEL} info output_base 2>/dev/null) \
  -name "eigen_contraction_kernel.h" 2>/dev/null | head -3
```
把 WORKSPACE 里 `old_sgemm` 字符串改成与实际文件一致的形式。

---

## 调用链说明

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]
      → cblas_sgemm(CblasColMajor, ...)   ← KBLAS patch 激活后
        → libkblas.so  [鲲鹏 SVE/NEON 汇编]
```

**为什么用 CblasColMajor 而不需要对调 A/B**：
Eigen 以列主序打包矩阵块后传入内核。原 dnnl_sgemm 是行主序约定，
需对调 A↔B 和 M↔N 来绕过。KBLAS `cblas_sgemm(CblasColMajor)` 原生支持列主序，
直接用自然参数顺序，无需对调。
