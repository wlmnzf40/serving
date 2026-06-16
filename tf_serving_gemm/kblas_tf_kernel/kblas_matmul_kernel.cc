/*
 * Custom TensorFlow MatMul / BatchMatMul kernels backed by KML KBLAS.
 *
 * Why Priority(2):
 *   TF's built-in MatMul kernel is registered with Priority(1).
 *   A kernel with Priority(2) wins in the kernel registry lookup,
 *   transparently replacing the Eigen/tensorContract path without
 *   any change to the model or the serving binary.
 *
 * Coverage:
 *   MatMul      → replaces both ParallelMatMulKernel and SequentialMatMulKernel
 *   BatchMatMul / BatchMatMulV2 → replaces the batched path
 *
 * Build:
 *   bazel build //tf_serving_gemm/kblas_tf_kernel:kblas_matmul_kernel.so
 *
 * Load in TF Serving:
 *   tensorflow_model_server ... \
 *       --tensorflow_intra_op_parallelism_threads=1 \
 *       --custom_op_paths=/path/to/kblas_matmul_kernel.so
 *   (or set TF_SERVING_KERNEL_REGISTRY_PATH env var)
 */

#include <cblas.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// ── MatMul ────────────────────────────────────────────────────────────────────
// Computes C[M,N] = op(A) * op(B)
// op(X) = X (transpose=false) or X^T (transpose=true)
//
// TF tensors are row-major.  For row-major CBLAS:
//   transpose=false: stored as [M,K] → lda=K, TransA=NoTrans
//   transpose=true:  stored as [K,M] → lda=M, TransA=Trans

class KblasMatMulOp : public OpKernel {
public:
    explicit KblasMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& a = ctx->input(0);
        const Tensor& b = ctx->input(1);

        OP_REQUIRES(ctx, a.dims() == 2,
            errors::InvalidArgument("MatMul: A must be 2-D, got shape ",
                                    a.shape().DebugString()));
        OP_REQUIRES(ctx, b.dims() == 2,
            errors::InvalidArgument("MatMul: B must be 2-D, got shape ",
                                    b.shape().DebugString()));

        // Mathematical dimensions after transposition
        const int64 M = a.dim_size(transpose_a_ ? 1 : 0);
        const int64 K = a.dim_size(transpose_a_ ? 0 : 1);
        const int64 N = b.dim_size(transpose_b_ ? 0 : 1);
        const int64 Kb = b.dim_size(transpose_b_ ? 1 : 0);

        OP_REQUIRES(ctx, K == Kb,
            errors::InvalidArgument("MatMul: K dimension mismatch: ", K, " vs ", Kb));

        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({M, N}), &out));

        // lda/ldb = number of columns of the stored (pre-transposition) matrix
        const int lda = static_cast<int>(a.dim_size(1));
        const int ldb = static_cast<int>(b.dim_size(1));

        cblas_sgemm(
            CblasRowMajor,
            transpose_a_ ? CblasTrans : CblasNoTrans,
            transpose_b_ ? CblasTrans : CblasNoTrans,
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            1.0f,
            a.flat<float>().data(), lda,
            b.flat<float>().data(), ldb,
            0.0f,
            out->flat<float>().data(), static_cast<int>(N));
    }

private:
    bool transpose_a_, transpose_b_;
};

// Priority(2) overrides the built-in Priority(1) kernel.
REGISTER_KERNEL_BUILDER(
    Name("MatMul").Device(DEVICE_CPU).TypeConstraint<float>("T").Priority(2),
    KblasMatMulOp);

// ── BatchMatMul / BatchMatMulV2 ───────────────────────────────────────────────
// Computes C[b,M,N] = op(A[b,M,K]) * op(B[b,K,N])
//
// BatchMatMul uses adj_x/adj_y (conjugate transpose); for float it is identical
// to regular transpose, so we handle it the same way.
// BatchMatMulV2 uses the same attributes but broadcasts; we only handle the
// common case where batch dims are identical (no broadcasting).

class KblasBatchMatMulOp : public OpKernel {
public:
    explicit KblasBatchMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        // Both BatchMatMul and BatchMatMulV2 expose adj_x / adj_y.
        OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &adj_x_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &adj_y_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& a = ctx->input(0);
        const Tensor& b = ctx->input(1);

        OP_REQUIRES(ctx, a.dims() >= 2,
            errors::InvalidArgument("BatchMatMul: A must be at least 2-D"));
        OP_REQUIRES(ctx, b.dims() >= 2,
            errors::InvalidArgument("BatchMatMul: B must be at least 2-D"));
        OP_REQUIRES(ctx, a.dims() == b.dims(),
            errors::InvalidArgument("BatchMatMul: A and B must have the same rank"));

        // Treat all leading dimensions as a flat batch count.
        const int ndim  = a.dims();
        int64     batch = 1;
        for (int i = 0; i < ndim - 2; ++i) {
            OP_REQUIRES(ctx, a.dim_size(i) == b.dim_size(i),
                errors::InvalidArgument("BatchMatMul: batch dim mismatch at axis ", i));
            batch *= a.dim_size(i);
        }

        const int64 M  = a.dim_size(ndim - (adj_x_ ? 1 : 2));
        const int64 K  = a.dim_size(ndim - (adj_x_ ? 2 : 1));
        const int64 N  = b.dim_size(ndim - (adj_y_ ? 2 : 1));
        const int64 Kb = b.dim_size(ndim - (adj_y_ ? 1 : 2));

        OP_REQUIRES(ctx, K == Kb,
            errors::InvalidArgument("BatchMatMul: K mismatch: ", K, " vs ", Kb));

        // Build output shape: batch dims + [M, N]
        TensorShape out_shape;
        for (int i = 0; i < ndim - 2; ++i) out_shape.AddDim(a.dim_size(i));
        out_shape.AddDim(M);
        out_shape.AddDim(N);

        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

        const CBLAS_TRANSPOSE transA = adj_x_ ? CblasTrans : CblasNoTrans;
        const CBLAS_TRANSPOSE transB = adj_y_ ? CblasTrans : CblasNoTrans;
        const int lda = static_cast<int>(a.dim_size(ndim - 1));
        const int ldb = static_cast<int>(b.dim_size(ndim - 1));
        const int ldC = static_cast<int>(N);

        const float* pa = a.flat<float>().data();
        const float* pb = b.flat<float>().data();
        float*       pc = out->flat<float>().data();

        const long long strideA = (long long)a.dim_size(ndim-2) * a.dim_size(ndim-1);
        const long long strideB = (long long)b.dim_size(ndim-2) * b.dim_size(ndim-1);
        const long long strideC = (long long)M * N;

        for (int64 i = 0; i < batch; ++i) {
            cblas_sgemm(
                CblasRowMajor, transA, transB,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0f,
                pa + i * strideA, lda,
                pb + i * strideB, ldb,
                0.0f,
                pc + i * strideC, ldC);
        }
    }

private:
    bool adj_x_, adj_y_;
};

REGISTER_KERNEL_BUILDER(
    Name("BatchMatMul").Device(DEVICE_CPU).TypeConstraint<float>("T").Priority(2),
    KblasBatchMatMulOp);

REGISTER_KERNEL_BUILDER(
    Name("BatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<float>("T").Priority(2),
    KblasBatchMatMulOp);

}  // namespace tensorflow
