# Build Instructions

## Step 1 — 安装 / 编译 gRPC + protobuf

### 方式 A：系统包（如果 apt 可用）

```bash
sudo apt install -y \
    libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    cmake libssl-dev
```

### 方式 B：从源码编译（aarch64 + 自定义 GCC 推荐）

```bash
git clone -b v1.62.0 --depth 1 --recurse-submodules \
    https://github.com/grpc/grpc /home/wanglimin/grpc_src
cd /home/wanglimin/grpc_src

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/home/wanglimin/grpc_install \
      -DCMAKE_C_COMPILER=/usr/bin/gcc \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/home/wanglimin/gcc-12.3.1-2025.12-aarch64-linux/lib64" \
      -DgRPC_INSTALL=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DgRPC_BUILD_BENCHMARKS=OFF \
      ..

make -j$(nproc)
make install
```

---

## Step 2 — 编译 gemm_server / gemm_client

```bash
cd tf_serving_gemm/grpc_gemm_server
mkdir build && cd build

# 如果 gRPC 安装到自定义前缀：
cmake -DCMAKE_PREFIX_PATH=/home/wanglimin/grpc_install \
      -DCMAKE_C_COMPILER=/usr/bin/gcc \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      ..

# 如果 gRPC 在系统路径（apt 安装）：
# cmake ..

make -j$(nproc)
```

Cmake 会自动：
- 用 `protoc` + `grpc_cpp_plugin` 从 `proto/gemm.proto` 生成 `gemm.pb.cc/h` 和 `gemm.grpc.pb.cc/h`
- 用 FetchContent 拉取 Eigen 3.4.0（header-only）
- 编译 `gemm_server` 和 `gemm_client`

---

## Step 3 — 运行

```bash
# 终端 1：启动 server
./gemm_server --addr=0.0.0.0:50051

# 终端 2：sweep benchmark（server 内部测量，无网络开销）
./gemm_client --mode=sweep \
              --sizes=128,256,512,1024,2048 \
              --iters=50 --warmup=10

# 或者单次 compute（测端到端 RTT + server 纯计算时间）
./gemm_client --mode=compute --M=1024 --K=1024 --N=1024 --iters=50
```

---

## 典型 sweep 输出

```
Size        Avg(ms)     P50(ms)     P99(ms)     GFLOPS
------------------------------------------------------------
128x128     0.082       0.081       0.095       51.2
256x256     0.431       0.428       0.461       61.8
512x512     3.214       3.198       3.311       83.6
1024x1024   24.70       24.65       25.10       87.2
2048x2048   195.3       194.8       197.1       88.1
```

---

## 说明

| 点 | 说明 |
|----|------|
| GEMM kernel | Eigen（TF CPU matmul 的底层）；`-march=native` 开启 NEON/SVE |
| Sweep 测量 | 完全在 server 侧，结果不受序列化/网络影响 |
| Compute 测量 | 同时报告 client RTT 和 server 纯计算时间，方便对比开销 |
| 线程数 | `--threads=N` 控制 Eigen 并行线程（默认 = hardware_concurrency） |
