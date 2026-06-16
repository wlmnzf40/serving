# KML KBLAS Build Skill
# 用途：在鲲鹏（aarch64）机器上，用华为 KML 库替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server

## 概述

这个 Skill 完成三件事：
1. 下载并安装华为 KML（鲲鹏数学库）
2. 确认 WORKSPACE patch 和 .bazelrc config 已到位（代码已预置在仓库里）
3. 用 `--config=kml_kblas` 编译 gemm_server / gemm_client，运行对比测试

---

## 前置确认（运行前请告知以下信息）

- Bazel 可执行路径（默认 `/home/wanglimin/bazel-7.4.1`）
- distdir 路径（默认 `/home/wanglimin/tf_new/dist`）
- GCC rpath（默认 `/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64`）
- KML 安装目标路径（默认 `/usr/local/kml`，需要 root；也可选用户目录）
- **TF 已编译过，不要 `bazel clean --expunge`**

---

## Step 1 — 下载并安装 KML

```bash
# 下载 RPM（~40 MB）
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

# 方式 A：有 root，直接 rpm 安装（装到 /usr/local/kml/）
sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm

# 方式 B：无 root，手动解压到用户目录
mkdir -p /home/wanglimin/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D /home/wanglimin/kml
# 解压后库在 /home/wanglimin/kml/usr/local/kml/ 或类似子目录，ln -s 到 /home/wanglimin/kml/
```

**验证安装：**
```bash
KML_ROOT=/usr/local/kml          # 方式 B 改成 /home/wanglimin/kml
ls $KML_ROOT/include/kblas.h     # 必须存在
ls $KML_ROOT/lib/libkblas.so     # 必须存在
nm -D $KML_ROOT/lib/libkblas.so | grep cblas_sgemm   # 必须有输出
```

---

## Step 2 — 确认仓库已包含 patch（只需检查，不需修改）

仓库里已预置两处改动：

### 2-A：WORKSPACE `patch_cmds`（自动 patch eigen_contraction_kernel.h）

`WORKSPACE` 的 `tensorflow_http_archive` 里最后一段 `patch_cmds` 会在 Bazel 解压 TF 源码后自动执行 Python 脚本，将 `eigen_contraction_kernel.h` 里的：
- `dnnl_sgemm(transposeB, transposeA, n, m, k, ...)` → `cblas_sgemm(CblasColMajor, transposeA, transposeB, m, n, k, ...)`
- 头文件 `dnnl.h` → `kblas.h`
- 宏名 `TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL` → `..._KML`

**检查是否已到位：**
```bash
grep -n "KBLAS_PATCH_EOF" WORKSPACE | head -2   # 应有输出
```

### 2-B：.bazelrc `kml_kblas` config

**检查：**
```bash
grep -A 12 "^build:kml_kblas" .bazelrc
```

预期输出：
```
build:kml_kblas --copt=-DTENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL
build:kml_kblas --copt=-DTENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML
build:kml_kblas --copt=-I/usr/local/kml/include
build:kml_kblas --linkopt=-L/usr/local/kml/lib
build:kml_kblas --linkopt=-lkblas
build:kml_kblas --linkopt=-Wl,-rpath,/usr/local/kml/lib
...
```

**如果 KML 装在非默认路径**（如 `/home/wanglimin/kml`），修改这三行：
```bash
sed -i 's|/usr/local/kml/include|/home/wanglimin/kml/include|g' .bazelrc
sed -i 's|/usr/local/kml/lib|/home/wanglimin/kml/lib|g'         .bazelrc
```

---

## Step 3 — 编译命令

```bash
/home/wanglimin/bazel-7.4.1 build -c opt \
  --distdir=/home/wanglimin/tf_new/dist \
  --define=no_cuda_support=true --define=no_nccl_support=true \
  --define=no_kafka_support=true --define=no_google_cloud_support=true \
  --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++ \
  --host_linkopt=-Wl,--disable-new-dtags \
  --host_linkopt=-Wl,-rpath,/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64 \
  --linkopt=-Wl,--disable-new-dtags \
  --linkopt=-Wl,-rpath,/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64 \
  --config=kml_kblas \
  //tf_serving_gemm/tf_gemm_server:gemm_server \
  //tf_serving_gemm/tf_gemm_server:gemm_client
```

> **注意**：`--config=kml_kblas` 的 `-DTENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL` 和
> `-DTENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML` 这两个宏激活了 WORKSPACE patch
> 写入的 `cblas_sgemm` 代码路径。缺少任意一个宏，会 fallback 到 Eigen 原生内核。

### 3-A 编译过程中的关键确认点

编译成功后验证 KBLAS 符号确实被链入：
```bash
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  | grep cblas_sgemm
# 预期：U cblas_sgemm   （U = undefined，由 libkblas.so 提供）

ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  | grep kblas
# 预期：libkblas.so => /usr/local/kml/lib/libkblas.so
```

---

## Step 4 — 运行对比测试

### 启动 server

```bash
# 运行前确保 libkblas.so 可找到（rpath 已烧入，通常不需要 LD_LIBRARY_PATH）
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --addr=0.0.0.0:50052 &
SERVER_PID=$!
sleep 2   # 等待 server 初始化
```

### 基准测试（和原来用法完全一样）

```bash
# 快速验证
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=30 --warmup=5

# 稳定性测试
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=200 --warmup=50

kill $SERVER_PID
```

### 预期结果（相对 Eigen 原生）

| 矩阵规模 | Eigen 原生 | KBLAS | 提升 |
|---------|-----------|-------|------|
| 512×512  | ~X ms     | ~Y ms | ~1.5-3× |
| 1024×1024| ~X ms     | ~Y ms | ~2-4×   |
| 2048×2048| ~X ms     | ~Y ms | ~2-5×   |

（实际数字取决于鲲鹏型号和线程数，KML 对 Kunpeng 920 优化效果最显著）

---

## 常见问题

### Q1：编译报 `kblas.h: No such file or directory`
KML 头文件路径不对。检查：
```bash
find /usr/local/kml /home/wanglimin/kml -name "kblas.h" 2>/dev/null
```
然后修改 `.bazelrc` 里 `--copt=-I<path>` 对应路径。

### Q2：运行时报 `libkblas.so: cannot open shared object file`
rpath 有问题，临时解决：
```bash
export LD_LIBRARY_PATH=/usr/local/kml/lib:$LD_LIBRARY_PATH
```
永久解决：
```bash
echo "/usr/local/kml/lib" | sudo tee /etc/ld.so.conf.d/kml.conf
sudo ldconfig
```

### Q3：性能没有提升，看起来还是 Eigen
确认宏被激活：
```bash
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
```
如果没有输出，说明宏未生效，检查 `--config=kml_kblas` 是否传入了两个 `-D` 宏。

### Q4：`dnnl_sgemm pattern not found` 报错（build 时）
说明 TF 源码的 `eigen_contraction_kernel.h` 版本与 WORKSPACE patch 预期的字符串不匹配。
检查实际文件内容：
```bash
# 找到 Bazel sandbox 里的文件
find $(bazel info output_base) -name "eigen_contraction_kernel.h" 2>/dev/null | head -3
```
然后把 WORKSPACE 里 `old_sgemm` 字符串改成实际代码里的形式。

### Q5：`--config=kml_kblas` 与原有 `--config=mkl_aarch64` 冲突
两者不要同时用。`kml_kblas` 完全替代 `mkl_aarch64`：
- `mkl_aarch64` 用 oneDNN（需要 dnnl 库，鲲鹏上不一定最优）
- `kml_kblas` 用 KML BLAS（针对鲲鹏硬件微架构手工调优）

---

## 代码原理说明（供排查用）

### 调用链

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel (eigen_contraction_kernel.h)
      → [KBLAS patch 激活后] cblas_sgemm(CblasColMajor, ...)
        → libkblas.so  [鲲鹏 SVE/NEON 汇编实现]
```

### 为什么用 CblasColMajor 而不是 RowMajor

Eigen 的 `tensor.contract()` 把矩阵块（panels）以**列主序**格式打包后传入 `dnnl_sgemm`。
原 oneDNN 代码用的是行主序约定，所以要把 A/B 对调、M/N 对调来绕过。
KBLAS 支持 `CblasColMajor`，因此可以直接用自然顺序，无需对调。

---

## 参考

- KML 下载：`https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/`
- 鲲鹏 KBLAS 接入博客（CSDN，Kunpeng开发者）：搜索"鲲鹏 KML cblas_sgemm TensorFlow contraction kernel"
- 仓库改动文件：`WORKSPACE`（patch_cmds 段）、`.bazelrc`（kml_kblas config）
