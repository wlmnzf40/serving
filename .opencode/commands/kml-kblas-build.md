---
description: 鲲鹏 aarch64 上用华为 KML 替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server
---

# KML KBLAS Build

在鲲鹏（aarch64/Kunpeng）机器上完成以下三件事：
1. 下载安装华为 KML（鲲鹏数学库）
2. 运行 `tools/apply_kblas_patch.sh` 直接 patch Bazel 缓存里的 eigen_contraction_kernel.h
3. 编译 gemm_server / gemm_client 并验证 KBLAS 替换生效

**重要约束**：TF 已编译过，不要 `bazel clean --expunge`。
本方案不修改 WORKSPACE（避免 Bazel 重新 fetch TF），改为独立脚本就地 patch 缓存文件。

---

## Step 1 — 下载并安装 KML

```bash
# 下载 RPM（~40 MB）
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

# 方式 A：有 sudo，直接安装到 /usr/local/kml/
sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm

# 方式 B：无 root，解压到用户目录
mkdir -p /home/wanglimin/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D /home/wanglimin/kml
# 找到实际子目录，例如 /home/wanglimin/kml/usr/local/kml/
```

**验证：**
```bash
KML_ROOT=/usr/local/kml          # 方式 B 改为实际路径
ls $KML_ROOT/include/kblas.h     # 必须存在
ls $KML_ROOT/lib/libkblas.so     # 必须存在
nm -D $KML_ROOT/lib/libkblas.so | grep cblas_sgemm   # 必须有输出
```

**如果 KML 装在非默认路径**，修改 `.bazelrc`：
```bash
KML_ACTUAL=/home/wanglimin/kml/usr/local/kml   # 改成实际路径
sed -i "s|/usr/local/kml/include|$KML_ACTUAL/include|g" .bazelrc
sed -i "s|/usr/local/kml/lib|$KML_ACTUAL/lib|g"         .bazelrc
```

---

## Step 2 — 运行 KBLAS patch 脚本

脚本直接修改 Bazel output base 里已解压的 `eigen_contraction_kernel.h`，
**不碰 WORKSPACE，Bazel 指纹不变，之前的编译缓存全部保留**。

```bash
BAZEL=/home/wanglimin/bazel-7.4.1   # 改成实际路径
REPO=/home/wanglimin/tf_serving      # 改成仓库根目录

cd $REPO
bash tools/apply_kblas_patch.sh $BAZEL
```

**成功输出示例：**
```
Patching: /home/wanglimin/.cache/bazel/_bazel_wanglimin/xxxx/external/org_tensorflow/.../eigen_contraction_kernel.h
OK: ...eigen_contraction_kernel.h
Patch applied. Backup saved at: ...bak_dnnl
```

脚本是**幂等**的：重复运行输出 `Already patched. Nothing to do.`

**如果报 `Header not found`**：
说明 org_tensorflow 还没被 fetch，先不加 `--config=kml_kblas` 跑一次普通 build，
让 Bazel 解压 TF，再执行本脚本。

**如果报 `dnnl_sgemm pattern mismatch`**：
脚本会打印实际找到的字符串，把它更新到脚本的 `old_sgemm` 变量即可。

---

## Step 3 — 编译命令

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

> `--config=kml_kblas` 里的 `-DTENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML` 宏
> 激活了 Step 2 patch 写入的 `cblas_sgemm` 代码路径。

**编译后验证：**
```bash
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 预期：U cblas_sgemm

ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
# 预期：libkblas.so => /usr/local/kml/lib/libkblas.so
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
find /usr/local/kml /home/wanglimin/kml -name "kblas.h" 2>/dev/null
```

### libkblas.so: cannot open shared object file（运行时）
```bash
export LD_LIBRARY_PATH=/usr/local/kml/lib:$LD_LIBRARY_PATH
# 永久：echo "/usr/local/kml/lib" | sudo tee /etc/ld.so.conf.d/kml.conf && sudo ldconfig
```

### 性能没有提升
```bash
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 无输出 → 检查 --config=kml_kblas 是否正确传入
```

### 报 `tensor_testutil.cc: CONTENT_DOES_NOT_MATCH_TARGET`
这是 `tensorflow.patch` 本身的问题，**不要修改 WORKSPACE**（会触发 re-fetch）。
只需运行 `tools/apply_kblas_patch.sh` 就地 patch 缓存即可。

---

## 调用链

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]
      → cblas_sgemm(CblasColMajor, ...)   ← patch 激活后
        → libkblas.so  [鲲鹏 SVE/NEON 汇编]
```

Eigen 以列主序打包矩阵块，cblas_sgemm(CblasColMajor) 原生支持列主序，
无需像 dnnl_sgemm（行主序约定）那样对调 A/B 和 M/N。
