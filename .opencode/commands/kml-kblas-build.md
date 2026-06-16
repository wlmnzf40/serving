---
description: 鲲鹏 aarch64 上用华为 KML 替换 Eigen 矩阵乘内核，编译并测试 TF-Serving GEMM server
---

# KML KBLAS Build

在鲲鹏（aarch64/Kunpeng）机器上完成以下三件事：
1. 下载安装华为 KML（鲲鹏数学库）
2. 运行 `tools/setup_kblas.sh`（一键处理所有编译前依赖）
3. 编译 gemm_server / gemm_client 并验证 KBLAS 替换生效

**重要约束**：TF 已编译过，不要 `bazel clean --expunge`。

---

## Step 1 — 安装 KML

```bash
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

# 有 sudo（推荐）
sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm

# 无 root
mkdir -p /home/wanglimin/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D /home/wanglimin/kml
```

**验证：**
```bash
KML_ROOT=/usr/local/kml    # 无 root 改成实际子目录
ls $KML_ROOT/include/kblas.h && ls $KML_ROOT/lib/libkblas.so
nm -D $KML_ROOT/lib/libkblas.so | grep cblas_sgemm
```

---

## Step 2 — 一键 Setup（`tools/setup_kblas.sh`）

这个脚本处理以下三件事，每件都是幂等的：

| 子步骤 | 做什么 | 出错时怎么处理 |
|--------|--------|----------------|
| **repo.bzl 修复** | 检查 `tensorflow_serving/repo.bzl` 是否有 `tf_serving_vendored`，没有则自动追加 | 自动修复 |
| **`.bazelrc` 路径** | KML 不在默认 `/usr/local/kml` 时，自动 `sed` 修改路径 | 传第二个参数指定路径 |
| **头文件 patch** | 在 Bazel 缓存就地替换 `dnnl_sgemm → cblas_sgemm`（正则匹配，容忍不同 TF 版本） | 缓存不存在时提示先跑一次普通 build |

```bash
BAZEL=/home/wanglimin/bazel-7.4.1
cd /home/wanglimin/tf_serving

# KML 在默认路径 /usr/local/kml
bash tools/setup_kblas.sh $BAZEL

# KML 在自定义路径
bash tools/setup_kblas.sh $BAZEL /home/wanglimin/kml/usr/local/kml
```

**如果提示 "patch deferred"（头文件不在缓存）**，先触发一次普通编译让 Bazel 解压 TF，再重跑 setup：

```bash
$BAZEL build -c opt --distdir=/home/wanglimin/tf_new/dist \
  --define=no_cuda_support=true --define=no_nccl_support=true \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | tail -5

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

**验证：**
```bash
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
```

---

## Step 4 — 测试

```bash
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --addr=0.0.0.0:50052 &
sleep 2
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=200 --warmup=50
kill %1
```

---

## 常见问题

### `tf_serving_vendored` 找不到（编译报错）
`setup_kblas.sh` 会自动修复。手动修复：
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

### `kblas.h: No such file or directory`
```bash
find /usr/local/kml /home/wanglimin/kml -name "kblas.h" 2>/dev/null
bash tools/setup_kblas.sh /home/wanglimin/bazel-7.4.1 <KML根目录>
```

### `libkblas.so: cannot open shared object file`（运行时）
```bash
export LD_LIBRARY_PATH=/usr/local/kml/lib:$LD_LIBRARY_PATH
```

### `tensor_testutil.cc: CONTENT_DOES_NOT_MATCH_TARGET`
**不要改 WORKSPACE**，只跑 `tools/setup_kblas.sh` 就地 patch 缓存。

---

## 调用链

```
TF Session::Run(MatMul)
  → eigen::TensorContraction<float>
    → ParallelMatMulKernel  [eigen_contraction_kernel.h]
      → cblas_sgemm(CblasColMajor, ...)   ← patch 激活后
        → libkblas.so  [鲲鹏 SVE/NEON 汇编]
```
