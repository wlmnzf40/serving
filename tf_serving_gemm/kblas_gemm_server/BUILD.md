# kblas_gemm_server — 构建说明

纯 KBLAS 后端 gRPC 服务，用于与 grpc_gemm_server（Eigen）和 tf_gemm_server（TF 运行时）做横向对比。

## 前置依赖

| 依赖 | 说明 |
|------|------|
| KML ≥ 23.x | 华为 Kunpeng Math Library，含 KBLAS 模块 |
| CMake ≥ 3.16 | |
| Protobuf ≥ 3.21 | |
| gRPC ≥ 1.54 | |
| GCC ≥ 10 (aarch64) | 需要 OpenMP 支持（`-fopenmp`）|

## KML 安装

```bash
# RPM（openEuler / CentOS）
rpm -ivh kml-<version>.aarch64.rpm
# 默认安装到 /usr/local/kml

# 或者解压到自定义路径
tar -xf kml-<version>.tar.gz -C /opt/kml
```

## 编译

```bash
cd tf_serving_gemm/kblas_gemm_server
mkdir build && cd build

# KML 在默认路径
cmake ..

# KML 在自定义路径
cmake -DKML_ROOT=/opt/kml ..

make -j$(nproc)
```

## 运行

```bash
./kblas_gemm_server --addr=0.0.0.0:50053 --threads=64
```

`--threads` 控制 `OMP_NUM_THREADS`（如未在环境中设置）。

## 测试

用 tf_gemm_server 的 client 即可（proto 接口相同）：

```bash
# 方形矩阵 sweep
../../tf_serving_gemm_client --addr=localhost:50053 \
    --mode=sweep --sizes=128,256,512,1024,2048

# 生产 shape sweep（ShapeSweep RPC）
../../tf_serving_gemm_client --addr=localhost:50053 \
    --mode=shape_sweep --shapes_file=../../shapes.json
```
