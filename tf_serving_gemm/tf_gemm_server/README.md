# tf_gemm_server — TF-runtime-backed GEMM gRPC service

The bazel-integrated, **TensorFlow-backed** counterpart to `../grpc_gemm_server`
(raw Eigen + CMake). Both expose the **same `gemm.proto` / gRPC interface**, so
the existing `python_client` / `cpp_client` work against either — only the
backend and port differ.

| | `../grpc_gemm_server` | `tf_gemm_server` (this) |
|---|---|---|
| Backend | raw Eigen | **TF C++ `MatMul` op kernel** (same kernel as `tf.linalg.matmul`) |
| Build | CMake | **bazel**, inside the TF Serving repo |
| gRPC / protobuf | system / vendored | `@com_github_grpc_grpc` (same as Serving → no symbol clash) |
| Default port | 50051 | **50052** |

The graph (`Placeholder(A), Placeholder(B) -> MatMul -> C`, dynamic shapes) is
built once and reused via one mutex-protected `ClientSession`.

## Build

Same flags as your `tensorflow_model_server` build — only the target changes.
Run from the repo root (`/home/wanglimin/tf_serving`):

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
  //tf_serving_gemm/tf_gemm_server:gemm_server \
  //tf_serving_gemm/tf_gemm_server:gemm_client
```

Binaries land at:

```
bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server
bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client
```

## Run

```bash
# terminal 1 — TF server on :50052
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server

# terminal 2 — server-side sweep (measurement has no network in the loop)
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client \
  --mode=sweep --sizes=128,256,512,1024,2048 --iters=50

# single round-trip compute (client RTT vs pure server compute)
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client \
  --mode=compute --M=1024 --K=1024 --N=1024 --iters=50
```

Point the same client at the Eigen server to compare backends:

```bash
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50051 --mode=sweep
```
