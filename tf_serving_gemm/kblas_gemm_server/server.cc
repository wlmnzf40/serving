/*
 * GEMM gRPC server — KML KBLAS backend
 *
 * Directly calls cblas_sgemm (KML's CBLAS interface) instead of Eigen.
 * Row-major layout matches TF tensor memory order.
 *
 * Build: see BUILD.md
 * Run:   ./kblas_gemm_server [--addr=0.0.0.0:50053] [--threads=N]
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <cblas.h>
#include <grpcpp/grpcpp.h>

#include "gemm.grpc.pb.h"
#include "gemm.pb.h"

using Clock = std::chrono::steady_clock;

// ── helpers ───────────────────────────────────────────────────────────────────

static volatile float g_sink = 0.f;

struct Stats { double avg, p50, p99; };

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) s += x;
    return {s / v.size(), v[v.size() / 2], v[v.size() * 99 / 100]};
}

static void fill_random(float* data, int n) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;
    for (int i = 0; i < n; ++i) data[i] = dist(rng);
}

// C[M,N] = A[M,K] * B[K,N]  (row-major, no transpose)
static inline void sgemm(int M, int K, int N,
                          const float* A, const float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K, B, N,
                0.0f, C, N);
}

// Batched: A[b,M,K] x B[b,K,N] → C[b,M,N]
static inline void sgemm_batched(int batch, int M, int K, int N,
                                  const float* A, const float* B, float* C) {
    for (int i = 0; i < batch; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    A + (long long)i * M * K, K,
                    B + (long long)i * K * N, N,
                    0.0f,
                    C + (long long)i * M * N, N);
    }
}

// bench a single MxKxN shape, return latencies (ms) for `iters` measured runs.
static std::vector<double> bench_shape(int batch, int M, int K, int N,
                                        int iters, int warmup) {
    long long szA = (long long)batch * M * K;
    long long szB = (long long)batch * K * N;
    long long szC = (long long)batch * M * N;

    std::vector<float> A(szA), B(szB), C(szC);
    fill_random(A.data(), szA);
    fill_random(B.data(), szB);

    auto run = [&] {
        if (batch == 1) sgemm(M, K, N, A.data(), B.data(), C.data());
        else            sgemm_batched(batch, M, K, N, A.data(), B.data(), C.data());
        g_sink += C[0];
    };

    for (int i = 0; i < warmup; ++i) run();

    std::vector<double> lats(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        run();
        lats[i] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
    return lats;
}

// ── service impl ──────────────────────────────────────────────────────────────

class GEMMServiceImpl final : public gemm::GEMMService::Service {

    grpc::Status Compute(grpc::ServerContext*,
                          const gemm::ComputeRequest* req,
                          gemm::ComputeResponse* resp) override {
        int M = req->m(), K = req->k(), N = req->n();
        if (req->a_data_size() != M * K || req->b_data_size() != K * N)
            return {grpc::StatusCode::INVALID_ARGUMENT, "a/b size mismatch with M,K,N"};

        std::vector<float> C(M * N);
        auto t0 = Clock::now();
        sgemm(M, K, N, req->a_data().data(), req->b_data().data(), C.data());
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        resp->mutable_c_data()->Assign(C.begin(), C.end());
        resp->set_server_compute_ms(ms);
        return grpc::Status::OK;
    }

    grpc::Status Sweep(grpc::ServerContext*,
                        const gemm::SweepRequest* req,
                        gemm::SweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 50;
        int warmup = req->warmup() > 0 ? req->warmup() : 10;

        for (int sz : req->sizes()) {
            auto lats = bench_shape(1, sz, sz, sz, iters, warmup);
            auto st   = compute_stats(lats);
            double gfl = 2.0 * sz * sz * sz / (st.avg / 1e3) / 1e9;

            std::cout << "[sweep] " << sz << "x" << sz
                      << "  avg=" << st.avg << " ms  GFLOPS=" << gfl << "\n" << std::flush;

            auto* r = resp->add_results();
            r->set_m(sz); r->set_k(sz); r->set_n(sz);
            r->set_avg_ms(st.avg); r->set_p50_ms(st.p50); r->set_p99_ms(st.p99);
            r->set_gflops(gfl);
        }
        return grpc::Status::OK;
    }

    grpc::Status ShapeSweep(grpc::ServerContext*,
                              const gemm::ShapeSweepRequest* req,
                              gemm::ShapeSweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 100;
        int warmup = req->warmup() > 0 ? req->warmup() : 20;

        for (const auto& shape : req->shapes()) {
            int b = shape.batch(), M = shape.m(), K = shape.k(), N = shape.n();

            auto lats = bench_shape(b, M, K, N, iters, warmup);
            auto st   = compute_stats(lats);
            double fl  = 2.0 * b * M * K * N;
            double gfl = fl / (st.avg / 1e3) / 1e9;

            std::cout << "[shape] " << shape.model()
                      << " b=" << b << " [" << M << "x" << K << "x" << N << "]"
                      << "  avg=" << st.avg << " ms  GFLOPS=" << gfl << "\n" << std::flush;

            auto* r = resp->add_results();
            *r->mutable_shape() = shape;
            r->set_avg_ms(st.avg); r->set_p50_ms(st.p50); r->set_p99_ms(st.p99);
            r->set_gflops(gfl);
        }
        return grpc::Status::OK;
    }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string addr    = "0.0.0.0:50053";
    int         threads = static_cast<int>(std::thread::hardware_concurrency());

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--addr=",    0) == 0) addr    = a.substr(7);
        else if (a.rfind("--threads=", 0) == 0) threads = std::stoi(a.substr(10));
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    // Let KML's OpenMP/thread pool use the right number of threads.
    // KML respects OMP_NUM_THREADS; set it before the first BLAS call.
    std::string omp_val = std::to_string(threads);
    setenv("OMP_NUM_THREADS", omp_val.c_str(), /*overwrite=*/0);

    std::cout << "KBLAS threads (OMP_NUM_THREADS): "
              << (getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "default") << "\n";

    GEMMServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(256 << 20);
    builder.SetMaxSendMessageSize(256 << 20);

    auto server = builder.BuildAndStart();
    std::cout << "KBLAS GEMM server listening on " << addr << "\n";
    server->Wait();
    return 0;
}
