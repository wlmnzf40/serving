/*
 * GEMM gRPC server — backed by the TensorFlow C++ runtime.
 *
 * RPCs:
 *   Compute    — single MatMul with client-supplied data
 *   Sweep      — server-side benchmark for square sizes
 *   ShapeSweep — server-side benchmark for arbitrary production shapes
 *                (batch=1 → plain MatMul, batch>1 → BatchMatMul)
 *
 * Build (same flags as tensorflow_model_server, different target):
 *   //tf_serving_gemm/tf_gemm_server:gemm_server
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "grpcpp/grpcpp.h"
#include "tf_serving_gemm/tf_gemm_server/proto/gemm.grpc.pb.h"
#include "tf_serving_gemm/tf_gemm_server/proto/gemm.pb.h"

namespace tf    = tensorflow;
namespace tfops = tensorflow::ops;
using Clock = std::chrono::steady_clock;

// ── helpers ───────────────────────────────────────────────────────────────────

struct Stats { double avg, p50, p99; };

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) s += x;
    return {s / v.size(), v[v.size() / 2], v[v.size() * 99 / 100]};
}

// 2-D random tensor [rows, cols]
static tf::Tensor make_random(int rows, int cols) {
    tf::Tensor t(tf::DT_FLOAT, tf::TensorShape({rows, cols}));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;
    auto flat = t.flat<float>();
    for (int i = 0; i < flat.size(); ++i) flat.data()[i] = dist(rng);
    return t;
}

// 3-D random tensor [batch, rows, cols] for BatchMatMul
static tf::Tensor make_random_3d(int batch, int rows, int cols) {
    tf::Tensor t(tf::DT_FLOAT, tf::TensorShape({batch, rows, cols}));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;
    auto flat = t.flat<float>();
    for (int i = 0; i < flat.size(); ++i) flat.data()[i] = dist(rng);
    return t;
}

// ── GEMMRunner (batch=1, plain MatMul) ────────────────────────────────────────

class GEMMRunner {
public:
    GEMMRunner() : scope_(tf::Scope::NewRootScope()) {
        auto ph_a = tfops::Placeholder(scope_.WithOpName("A"), tf::DT_FLOAT);
        auto ph_b = tfops::Placeholder(scope_.WithOpName("B"), tf::DT_FLOAT);
        auto mm   = tfops::MatMul(scope_.WithOpName("C"), ph_a, ph_b);
        TF_CHECK_OK(scope_.status());
        A_ph_ = ph_a;
        B_ph_ = ph_b;
        C_op_ = mm;
        session_ = std::make_unique<tf::ClientSession>(scope_);
    }

    double Run(const tf::Tensor& A, const tf::Tensor& B, tf::Tensor* C_out) {
        std::vector<tf::Tensor> outputs;
        std::lock_guard<std::mutex> lk(mu_);
        auto t0 = Clock::now();
        TF_CHECK_OK(session_->Run({{A_ph_, A}, {B_ph_, B}}, {C_op_}, &outputs));
        double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();
        *C_out = std::move(outputs[0]);
        return ms;
    }

private:
    tf::Scope  scope_;
    tf::Output A_ph_, B_ph_, C_op_;
    std::unique_ptr<tf::ClientSession> session_;
    std::mutex mu_;
};

// ── BatchGEMMRunner (batch>1, BatchMatMul [b,M,K] x [b,K,N]) ─────────────────

class BatchGEMMRunner {
public:
    BatchGEMMRunner() : scope_(tf::Scope::NewRootScope()) {
        auto ph_a = tfops::Placeholder(scope_.WithOpName("A"), tf::DT_FLOAT);
        auto ph_b = tfops::Placeholder(scope_.WithOpName("B"), tf::DT_FLOAT);
        auto mm   = tfops::BatchMatMul(scope_.WithOpName("C"), ph_a, ph_b);
        TF_CHECK_OK(scope_.status());
        A_ph_ = ph_a;
        B_ph_ = ph_b;
        C_op_ = mm;
        session_ = std::make_unique<tf::ClientSession>(scope_);
    }

    double Run(const tf::Tensor& A, const tf::Tensor& B, tf::Tensor* C_out) {
        std::vector<tf::Tensor> outputs;
        std::lock_guard<std::mutex> lk(mu_);
        auto t0 = Clock::now();
        TF_CHECK_OK(session_->Run({{A_ph_, A}, {B_ph_, B}}, {C_op_}, &outputs));
        double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();
        *C_out = std::move(outputs[0]);
        return ms;
    }

private:
    tf::Scope  scope_;
    tf::Output A_ph_, B_ph_, C_op_;
    std::unique_ptr<tf::ClientSession> session_;
    std::mutex mu_;
};

// ── gRPC service ──────────────────────────────────────────────────────────────

class GEMMServiceImpl final : public gemm::GEMMService::Service {
public:
    // ── Compute ───────────────────────────────────────────────────────────────
    grpc::Status Compute(grpc::ServerContext*,
                          const gemm::ComputeRequest* req,
                          gemm::ComputeResponse* resp) override {
        int M = req->m(), K = req->k(), N = req->n();
        if (req->a_data_size() != M * K || req->b_data_size() != K * N)
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "a_data / b_data size mismatch with M, K, N"};

        tf::Tensor A(tf::DT_FLOAT, tf::TensorShape({M, K}));
        tf::Tensor B(tf::DT_FLOAT, tf::TensorShape({K, N}));
        std::copy(req->a_data().begin(), req->a_data().end(),
                  A.flat<float>().data());
        std::copy(req->b_data().begin(), req->b_data().end(),
                  B.flat<float>().data());

        tf::Tensor C;
        double ms = runner_.Run(A, B, &C);
        auto flat = C.flat<float>();
        resp->mutable_c_data()->Assign(flat.data(), flat.data() + M * N);
        resp->set_server_compute_ms(ms);
        return grpc::Status::OK;
    }

    // ── Sweep (square sizes) ──────────────────────────────────────────────────
    grpc::Status Sweep(grpc::ServerContext*,
                        const gemm::SweepRequest* req,
                        gemm::SweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 50;
        int warmup = req->warmup() > 0 ? req->warmup() : 10;

        for (int sz : req->sizes()) {
            tf::Tensor A = make_random(sz, sz);
            tf::Tensor B = make_random(sz, sz);

            for (int i = 0; i < warmup; ++i) {
                tf::Tensor C;  runner_.Run(A, B, &C);
            }
            std::vector<double> lats(iters);
            for (int i = 0; i < iters; ++i) {
                tf::Tensor C;  lats[i] = runner_.Run(A, B, &C);
            }

            auto st  = compute_stats(lats);
            double gfl = 2.0 * sz * sz * sz / (st.avg / 1e3) / 1e9;
            std::cout << "[sweep] " << sz << "x" << sz
                      << "  avg=" << st.avg << " ms  GFLOPS=" << gfl << "\n"
                      << std::flush;

            auto* r = resp->add_results();
            r->set_m(sz);  r->set_k(sz);  r->set_n(sz);
            r->set_avg_ms(st.avg); r->set_p50_ms(st.p50); r->set_p99_ms(st.p99);
            r->set_gflops(gfl);
        }
        return grpc::Status::OK;
    }

    // ── ShapeSweep (production shapes) ───────────────────────────────────────
    // batch=1  → plain  MatMul  A[M,K] x B[K,N]
    // batch>1  → BatchMatMul   A[b,M,K] x B[b,K,N]
    grpc::Status ShapeSweep(grpc::ServerContext*,
                              const gemm::ShapeSweepRequest* req,
                              gemm::ShapeSweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 100;
        int warmup = req->warmup() > 0 ? req->warmup() : 20;

        for (const auto& shape : req->shapes()) {
            int  b = shape.batch(), M = shape.m(), K = shape.k(), N = shape.n();
            bool batched = (b > 1);

            tf::Tensor A = batched ? make_random_3d(b, M, K) : make_random(M, K);
            tf::Tensor B = batched ? make_random_3d(b, K, N) : make_random(K, N);

            for (int i = 0; i < warmup; ++i) {
                tf::Tensor C;
                batched ? batch_runner_.Run(A, B, &C) : runner_.Run(A, B, &C);
            }
            std::vector<double> lats(iters);
            for (int i = 0; i < iters; ++i) {
                tf::Tensor C;
                lats[i] = batched ? batch_runner_.Run(A, B, &C)
                                  : runner_.Run(A, B, &C);
            }

            auto st   = compute_stats(lats);
            double fl = 2.0 * b * M * K * N;
            double gfl = fl / (st.avg / 1e3) / 1e9;

            std::cout << "[shape] " << shape.model()
                      << " b=" << b << " [" << M << "x" << K << "x" << N << "]"
                      << "  avg=" << st.avg << " ms  GFLOPS=" << gfl << "\n"
                      << std::flush;

            auto* r = resp->add_results();
            *r->mutable_shape() = shape;
            r->set_avg_ms(st.avg);
            r->set_p50_ms(st.p50);
            r->set_p99_ms(st.p99);
            r->set_gflops(gfl);
        }
        return grpc::Status::OK;
    }

private:
    GEMMRunner      runner_;        // for batch=1
    BatchGEMMRunner batch_runner_;  // for batch>1
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string addr = "0.0.0.0:50052";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--addr=", 0) == 0) addr = a.substr(7);
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    GEMMServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(256 << 20);
    builder.SetMaxSendMessageSize(256 << 20);

    auto server = builder.BuildAndStart();
    std::cout << "TF GEMM server on " << addr
              << "  (MatMul / BatchMatMul kernels)\n";
    server->Wait();
    return 0;
}
