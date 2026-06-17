# TF-runtime-backed GEMM gRPC server, built inside the TF Serving repo.
#
# TensorFlow is pulled into the Serving WORKSPACE as @org_tensorflow, so every
# TF core target is referenced with that repo prefix. gRPC is @com_github_grpc_grpc
# (the same repo TF/Serving already use, so protobuf/grpc versions match and there
# are no symbol clashes).
#
# Proto plumbing uses the plain bazel rules (proto_library / cc_proto_library /
# cc_grpc_library) rather than @org_tensorflow's tf_proto_library: this proto
# imports nothing and needs no TF proto deps, so the plain rules are simpler and
# don't couple us to any tf_proto_library signature changes across TF versions.

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(default_visibility = ["//visibility:public"])

# ── Proto + gRPC stubs ────────────────────────────────────────────────────────
# Header paths are repo-root-relative to the package, i.e.
#   tf_serving_gemm/tf_gemm_server/proto/gemm.pb.h
#   tf_serving_gemm/tf_gemm_server/proto/gemm.grpc.pb.h
# which is exactly what server.cc / client.cc #include.

proto_library(
    name = "gemm_proto",
    srcs = ["proto/gemm.proto"],
)

cc_proto_library(
    name = "gemm_cc_proto",
    deps = [":gemm_proto"],
)

cc_grpc_library(
    name = "gemm_cc_grpc",
    srcs = [":gemm_proto"],
    grpc_only = True,  # messages come from :gemm_cc_proto; generate only the stubs
    deps = [":gemm_cc_proto"],
)

# ── Server (TF C++ Session + MatMul op kernel) ────────────────────────────────
# Three distinct link-time requirements:
#   direct_session : registers the local-process SessionFactory (target: "")
#                    without this you get "No session factory registered" at startup
#   all_kernels    : links the MatMul compute kernel
#                    without this you get "No OpKernel registered for MatMul" at Run()
#   core_cpu       : TF executor / threadpool runtime
#
# Built without --config=kml_kblas: plain TF, MatMul always runs Eigen's native
#   contraction kernel (--backend=eigen, default; --backend=kblas errors at startup)
# Built with    --config=kml_kblas: requires tools/apply_kblas_patch.sh to have
#   already patched eigen_contraction_kernel.h in the Bazel cache — that patch is
#   what makes --backend=kblas|eigen exist as a runtime switch inside MatMul's
#   Eigen::Tensor::contract() call (see server.cc's tf_serving_kblas_enabled).

cc_binary(
    name = "gemm_server",
    srcs = ["server.cc"],
    deps = [
        ":gemm_cc_grpc",
        ":gemm_cc_proto",
        "@org_tensorflow//tensorflow/cc:cc_ops",           # MatMul / Placeholder graph builders
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:all_kernels",    # MatMul compute kernel
        "@org_tensorflow//tensorflow/core:core_cpu",       # executor / threadpool runtime
        "@org_tensorflow//tensorflow/core:direct_session", # local SessionFactory (REQUIRED)
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

# ── Client (gRPC only, no TF runtime dep) ────────────────────────────────────

cc_binary(
    name = "gemm_client",
    srcs = ["client.cc"],
    deps = [
        ":gemm_cc_grpc",
        ":gemm_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
    ],
)
