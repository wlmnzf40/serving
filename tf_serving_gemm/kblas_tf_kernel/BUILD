# Custom TF MatMul / BatchMatMul kernels backed by KML KBLAS.
#
# Build:
#   bazel build -c opt                                     \
#     --define=kml_root=/usr/local/kml                    \
#     //tf_serving_gemm/kblas_tf_kernel:kblas_matmul_kernel.so
#
# The .so is loaded at serving startup via --custom_op_paths.

package(default_visibility = ["//visibility:public"])

# ── External KML library ──────────────────────────────────────────────────────
# Bazel doesn't have a built-in "find_library", so we point directly at the
# installed KML files.  Override kml_root on the command line:
#   bazel build --define=kml_root=/opt/kml ...

KML_ROOT = select({
    "//conditions:default": "/usr/local/kml",
}, no_match_error = "Set --define=kml_root=<path> if KML is not in /usr/local/kml")

# We declare KML as a cc_import so Bazel tracks the dependency correctly.
cc_import(
    name = "kblas",
    shared_library = "/usr/local/kml/lib/libkblas.so",
    hdrs = [],  # cblas.h added via includes below
)

# ── Kernel shared library ─────────────────────────────────────────────────────
cc_binary(
    name = "kblas_matmul_kernel.so",
    srcs = ["kblas_matmul_kernel.cc"],
    linkshared = True,
    copts = [
        "-O3",
        "-march=native",
        "-fopenmp",
        "-I/usr/local/kml/include",   # cblas.h
    ],
    linkopts = [
        "-L/usr/local/kml/lib",
        "-lkblas",
        "-fopenmp",
        "-Wl,-rpath,/usr/local/kml/lib",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
)
