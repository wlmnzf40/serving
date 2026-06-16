# KML KBLAS Build Skill
# 用途：在鲲鹏（aarch64）机器上，用华为 KML 库替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server

## 概述

这个 Skill 完成三件事：
1. 下载并安装华为 KML（鲲鹏数学库）
2. 运行 `tools/apply_kblas_patch.sh` 直接 patch Bazel 缓存里的 eigen_contraction_kernel.h
3. 用 `--config=kml_kblas` 编译 gemm_server / gemm_client，运行对比测试

**重要约束**：TF 已编译过，不要 `bazel clean --expunge`。
本方案不修改 WORKSPACE（避免 Bazel 重新 fetch TF），改为独立脚本就地 patch 缓存文件。

---

## 前置确认（运行前请告知以下信息）

- Bazel 可执行路径（默认 `/home/wanglimin/bazel-7.4.1`）
- distdir 路径（默认 `/home/wanglimin/tf_new/dist`）
- GCC rpath（默认 `/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64`）
- KML 安装目标路径（默认 `/usr/local/kml`，需要 root；也可选用户目录）

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

**验证安装：**
```bash
KML_ROOT=/usr/local/kml          # 方式 B 改成实际路径
ls $KML_ROOT/include/kblas.h     # 必须存在
ls $KML_ROOT/lib/libkblas.so     # 必须存在
nm -D $KML_ROOT/lib/libkblas.so | grep cblas_sgemm   # 必须有输出
```

**如果 KML 装在非默认路径**，修改 `.bazelrc` 里的三行路径：
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
# 切换到仓库根目录
cd /home/wanglimin/tf_serving

# 运行 patch 脚本（第一个参数是 bazel 可执行路径）
bash tools/apply_kblas_patch.sh /home/wanglimin/bazel-7.4.1
```

**成功输出示例：**
```
Patching: /home/wanglimin/.cache/bazel/_bazel_wanglimin/xxxx/external/org_tensorflow/third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h
OK: /home/wanglimin/.cache/bazel/.../.../eigen_contraction_kernel.h
Patch applied. Backup saved at: ...bak_dnnl
```

脚本是**幂等**的：重复运行会直接输出 `Already patched. Nothing to do.`

**如果报 `dnnl_sgemm pattern mismatch`**：
脚本会打印实际找到的字符串（`repr()` 格式），把它更新到脚本的 `old_sgemm` 变量即可。

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

`--config=kml_kblas` 展开后等价于（来自 `.bazelrc`）：
```
--copt=-DTENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL
--copt=-DTENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML   ← 激活 patch 后的代码路径
--copt=-I/usr/local/kml/include
--linkopt=-L/usr/local/kml/lib  --linkopt=-lkblas
--linkopt=-Wl,-rpath,/usr/local/kml/lib
--copt=-fopenmp  --linkopt=-fopenmp
--copt=-O3  --copt=-march=armv8.2-a
```

**编译后验证 KBLAS 确实链入：**
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
# 无输出 → 宏未激活，检查 --config=kml_kblas 是否正确传入
```

### 报 `tensor_testutil.cc: CONTENT_DOES_NOT_MATCH_TARGET`（fetch 时）
这是 `tensorflow.patch` 本身的问题，与 KBLAS 无关。
**不要修改 WORKSPACE** 的 patch_cmds（那会触发 re-fetch）。
只需运行 `tools/apply_kblas_patch.sh` 就地 patch 缓存即可。

### patch 脚本报 `Header not found`
说明 org_tensorflow 还没被 fetch。先不加 `--config=kml_kblas` 跑一次编译，
让 Bazel 把 TF 解压到 output base，再跑 patch 脚本。

---

## 调用链说明

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]
      → cblas_sgemm(CblasColMajor, ...)   ← patch 激活后
        → libkblas.so  [鲲鹏 SVE/NEON 汇编]
```

**为什么用 CblasColMajor 而不需要对调 A/B**：
Eigen 以列主序打包矩阵块后传入内核。原 dnnl_sgemm 是行主序约定，
需对调 A↔B 和 M↔N 来绕过。KBLAS `cblas_sgemm(CblasColMajor)` 原生支持列主序，
直接用自然参数顺序，无需对调。
