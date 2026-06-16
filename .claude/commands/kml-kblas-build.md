# KML KBLAS Build Skill
# 用途：在鲲鹏（aarch64）机器上，用华为 KML 替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server

## 概述

这个 Skill 完成三件事：
1. 下载并安装华为 KML（鲲鹏数学库）
2. 运行 `tools/setup_kblas.sh`（一键处理所有编译前依赖）
3. 用 `--config=kml_kblas` 编译并验证

**重要约束**：TF 已编译过，不要 `bazel clean --expunge`。

---

## 前置确认

- Bazel 可执行路径（默认 `/home/wanglimin/bazel-7.4.1`）
- distdir 路径（默认 `/home/wanglimin/tf_new/dist`）
- GCC rpath（默认 `/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64`）
- KML 安装路径（默认 `/usr/local/kml`）

---

## Step 1 — 安装 KML

```bash
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

# 有 sudo（推荐，装到 /usr/local/kml/）
sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm

# 无 root（装到用户目录）
mkdir -p /home/wanglimin/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D /home/wanglimin/kml
```

**验证：**
```bash
KML_ROOT=/usr/local/kml    # 无 root 改成实际子目录
ls $KML_ROOT/include/kblas.h && ls $KML_ROOT/lib/libkblas.so
nm -D $KML_ROOT/lib/libkblas.so | grep cblas_sgemm   # 必须有输出
```

---

## Step 2 — 一键 Setup（`tools/setup_kblas.sh`）

这个脚本处理以下三件事，每件都是幂等的：

| 子步骤 | 做什么 | 出错时怎么处理 |
|--------|--------|----------------|
| **repo.bzl 修复** | 检查 `tensorflow_serving/repo.bzl` 是否有 `tf_serving_vendored`，没有则自动追加 | 自动修复，无需手动操作 |
| **`.bazelrc` 路径** | 如果 KML 不在默认 `/usr/local/kml`，自动 `sed` 修改路径 | 传第二个参数指定路径 |
| **头文件 patch** | 在 Bazel 缓存里就地替换 `dnnl_sgemm → cblas_sgemm`（正则匹配，容忍不同TF版本） | 若缓存还不存在，提示先跑一次普通 build |

```bash
cd /home/wanglimin/tf_serving

# KML 在默认路径 /usr/local/kml
bash tools/setup_kblas.sh /home/wanglimin/bazel-7.4.1

# KML 在自定义路径
bash tools/setup_kblas.sh /home/wanglimin/bazel-7.4.1 /home/wanglimin/kml/usr/local/kml
```

**如果脚本提示"patch deferred"（头文件不在缓存）**，先跑一次不带 kml_kblas 的普通 build，然后重新运行 setup：

```bash
# 先触发 TF 解压（只 fetch，不编译）
/home/wanglimin/bazel-7.4.1 fetch @org_tensorflow//:unused 2>/dev/null || \
/home/wanglimin/bazel-7.4.1 build -c opt \
  --distdir=/home/wanglimin/tf_new/dist \
  --define=no_cuda_support=true --define=no_nccl_support=true \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | head -50

# 然后重新 setup
bash tools/setup_kblas.sh /home/wanglimin/bazel-7.4.1
```

---

## Step 3 — 编译

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

**验证 KBLAS 确实链入：**
```bash
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 预期：U cblas_sgemm

ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
# 预期：libkblas.so => /usr/local/kml/lib/libkblas.so
```

---

## Step 4 — 测试

```bash
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --addr=0.0.0.0:50052 &
SERVER_PID=$!
sleep 2

./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=200 --warmup=50

kill $SERVER_PID
```

---

## 常见问题

### `tf_serving_vendored` 找不到（编译报错）
`setup_kblas.sh` 会自动修复。如果手动修复：
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

### `kblas.h: No such file or directory`（编译报错）
```bash
find /usr/local/kml /home/wanglimin/kml -name "kblas.h" 2>/dev/null
# 找到路径后重新跑 setup：
bash tools/setup_kblas.sh /home/wanglimin/bazel-7.4.1 <找到的KML根目录>
```

### `libkblas.so: cannot open shared object file`（运行时）
```bash
export LD_LIBRARY_PATH=/usr/local/kml/lib:$LD_LIBRARY_PATH
# 永久：
echo "/usr/local/kml/lib" | sudo tee /etc/ld.so.conf.d/kml.conf && sudo ldconfig
```

### `dnnl_sgemm pattern mismatch`（patch 脚本警告）
脚本会打印实际找到的代码片段（`repr()` 格式）。
把这个内容告诉维护者，更新 `apply_kblas_patch.sh` 里的正则即可。
大多数情况下正则已经足够宽泛，这个警告不会出现。

### `tensor_testutil.cc: CONTENT_DOES_NOT_MATCH_TARGET`（fetch 时报错）
**不要改 WORKSPACE**，那会触发 Bazel re-fetch TF。
只需运行 `tools/setup_kblas.sh` 就地 patch 缓存即可。

---

## 调用链

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]
      → cblas_sgemm(CblasColMajor, ...)   ← patch 激活后
        → libkblas.so  [鲲鹏 SVE/NEON 汇编]
```

Eigen 以列主序打包矩阵块，`cblas_sgemm(CblasColMajor)` 原生支持列主序，
无需像 `dnnl_sgemm`（行主序约定）那样对调 A/B 和 M/N。
