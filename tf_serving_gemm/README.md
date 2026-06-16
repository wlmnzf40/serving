# TF Serving GEMM Benchmark

纯编译运行形态，不涉及 Docker。

## 整体流程

```
1. 编译 tensorflow_model_server (TF Serving 源码 + bazel)
2. python3 generate_model.py   → gemm_model/1/  (SavedModel)
3. ./start_serving.sh          → 直接跑 model server 二进制
4. benchmark_rest.py / benchmark.cc  → benchmark
```

---

## Step 1: 编译 tensorflow_model_server

### 1.1 克隆 TF Serving 源码（版本须与你的 TF 对齐）

```bash
git clone https://github.com/tensorflow/serving /home/wanglimin/tf_serving
cd /home/wanglimin/tf_serving

# 版本对齐：查看你的 TF 版本
git -C /home/wanglimin/tf_new/tensorflow log --oneline -1
# TF Serving 和 TF 的 release 版本号是一一对应的，例如 2.15.x
git checkout 2.15.0
```

### 1.2 bazel 编译命令

基于你现有的 TF 编译参数，TF Serving 的命令是：

```bash
cd /home/wanglimin/tf_serving

/home/wanglimin/bazel-7.4.1 build -c opt --distdir=/home/wanglimin/tf_new/dist --define=no_cuda_support=true --define=no_nccl_support=true --define=no_kafka_support=true --define=no_google_cloud_support=true --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++ --host_linkopt=-Wl,--disable-new-dtags --host_linkopt=-Wl,-rpath,/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64 --linkopt=-Wl,--disable-new-dtags --linkopt=-Wl,-rpath,/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64 --override_repository=org_boost=/home/wanglimin/tf_dist/boost //tensorflow_serving/model_servers:tensorflow_model_server
```

与你的 TF 命令相比的差异：

| 变化点 | 说明 |
|--------|------|
| 目标改为 `//tensorflow_serving/model_servers:tensorflow_model_server` | serving 的 main 二进制 |
| 去掉 `--per_file_copt=tensorflow/dtensor/.*@-std=c++20` | dtensor 是 TF core 特有的，serving repo 里没有 |
| 新增 `--define=no_kafka_support=true` 等 | 去掉不需要的可选依赖，减少编译量 |
| 新增 `--linkopt`（不只是 `--host_linkopt`） | 最终可执行文件也需要 rpath，host_linkopt 只影响构建工具链 |

编译产物在：
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

### 1.3 版本不匹配时：用本地 TF override

如果 serving 的 WORKSPACE 里 TF 版本与本地不一致：

```bash
# 在上面的 bazel 命令里追加：
  --override_repository=org_tensorflow=/home/wanglimin/tf_new/tensorflow
```

---

## Step 2: 生成 GEMM SavedModel

```bash
cd /path/to/tf_serving_gemm

# 如果本地 TF Python 符号有问题，用干净的 conda 环境：
conda create -n tf_gen python=3.10 -y
conda activate tf_gen
pip install tensorflow-cpu      # 轻量，不需要 GPU
python3 generate_model.py       # 生成 gemm_model/1/
conda deactivate
```

---

## Step 3: 启动服务

```bash
export TF_MODEL_SERVER=/home/wanglimin/tf_serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
./start_serving.sh

# 验证
curl http://localhost:8501/v1/models/gemm
```

---

## Step 4: 运行 benchmark

### Python REST（只需 requests + numpy）

```bash
pip install requests numpy
python3 python_client/benchmark_rest.py --M 1024 --K 1024 --N 1024 --iters 50
```

### Python gRPC

```bash
pip install grpcio tensorflow-serving-api
python3 python_client/benchmark_grpc.py --M 1024 --K 1024 --N 1024
```

### C++ REST（libcurl）

```bash
sudo apt install libcurl4-openssl-dev cmake
cd cpp_client && mkdir build && cd build
cmake .. && make -j$(nproc)
./gemm_benchmark --M 1024 --K 1024 --N 1024 --iters 50 --warmup 10
```

---

## 参数说明

| 参数 | 含义 | 默认 |
|------|------|------|
| `--M` | A 行数 | 512 |
| `--K` | A 列 / B 行 | 512 |
| `--N` | B 列数 | 512 |
| `--iters` | 正式测量轮数 | 50 |
| `--warmup` | 预热轮数 | 10 |
| `--host` | 服务地址 | localhost |
| `--port` | 端口 (REST=8501, gRPC=8500) | 8501/8500 |

---

## 停止服务

```bash
./stop_serving.sh
```
