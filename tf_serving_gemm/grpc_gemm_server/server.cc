/*
 * GEMM gRPC server — Eigen backend (same kernel TF uses for CPU matmul).
 *
 * Build: see BUILD.md
 * Run:   ./gemm_server [--addr=0.0.0.0:50051] [--threads=N]
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <grpcpp/grpcpp.h>

#include "gemm.grpc.pb.h"
#include "gemm.pb.h"

using Clock = std::chrono::steady_clock;
using RowMatF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ── helpers ───────────────────────────────────────────────────────────────────

// Prevent the compiler from eliminating the result of a GEMM.
static volatile float g_sink = 0.f;

struct Stats {
    double avg, p50, p99;
};

Stats compute_stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    return {
        sum / v.size(),
        v[v.size() / 2],
        v[static_cast<size_t>(v.size() * 0.99)],
    };
}

// Run MxKxN GEMM for (warmup + iters) iterations using Eigen.
// Returns sorted latencies for the 'iters' measured iterations.
std::vector<double> bench_gemm(int M, int K, int N, int iters, int warmup) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;

    // Column-major (Eigen default) — optimal layout for Eigen's GEMM packer.
    Eigen::MatrixXf A(M, K), B(K, N);
    for (int i = 0; i < M * K; ++i) A.data()[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) B.data()[i] = dist(rng);

    for (int i = 0; i < warmup; ++i) {
        Eigen::MatrixXf C = A * B;
        g_sink += C(0, 0);
    }

    std::vector<double> lats(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        Eigen::MatrixXf C = A * B;
        lats[i] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        g_sink += C(0, 0);
    }
    return lats;
}

// ── service impl ──────────────────────────────────────────────────────────────

class GEMMServiceImpl final : public gemm::GEMMService::Service {

    // Single GEMM: client sends row-major A, B; we return row-major C.
    grpc::Status Compute(grpc::ServerContext*,
                          const gemm::ComputeRequest* req,
                          gemm::ComputeResponse* resp) override {
        int M = req->m(), K = req->k(), N = req->n();

        if (req->a_data_size() != M * K || req->b_data_size() != K * N) {
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "a_data or b_data size does not match M/K/N"};
        }

        Eigen::Map<const RowMatF> A(req->a_data().data(), M, K);
        Eigen::Map<const RowMatF> B(req->b_data().data(), K, N);

        auto t0 = Clock::now();
        RowMatF C = A * B;
        double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();

        resp->mutable_c_data()->Assign(C.data(), C.data() + M * N);
        resp->set_server_compute_ms(ms);
        return grpc::Status::OK;
    }

    // Sweep: benchmark multiple square sizes; report stats, no data transfer.
    grpc::Status Sweep(grpc::ServerContext*,
                        const gemm::SweepRequest* req,
                        gemm::SweepResponse* resp) override {
        int iters  = req->iters()  > 0 ? req->iters()  : 50;
        int warmup = req->warmup() > 0 ? req->warmup() : 10;

        for (int sz : req->sizes()) {
            auto lats = bench_gemm(sz, sz, sz, iters, warmup);
            auto st = compute_stats(lats);
            double flops  = 2.0 * sz * sz * sz;
            double gflops = flops / (st.avg / 1e3) / 1e9;

            std::cout << "[sweep] " << sz << "x" << sz
                      << "  avg=" << st.avg << " ms"
                      << "  " << gflops << " GFLOPS\n" << std::flush;

            auto* r = resp->add_results();
            r->set_m(sz);  r->set_k(sz);  r->set_n(sz);
            r->set_avg_ms(st.avg);
            r->set_p50_ms(st.p50);
            r->set_p99_ms(st.p99);
            r->set_gflops(gflops);
        }
        return grpc::Status::OK;
    }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string addr    = "0.0.0.0:50051";
    int         threads = static_cast<int>(std::thread::hardware_concurrency());

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a.rfind("--addr=",    0) == 0) addr    = a.substr(7);
        else if (a.rfind("--threads=", 0) == 0) threads = std::stoi(a.substr(10));
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    // Tell Eigen how many threads to use for parallel GEMM.
    Eigen::setNbThreads(threads);
    std::cout << "Eigen threads: " << Eigen::nbThreads() << "\n";

    GEMMServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    // Allow large matrices (up to 256 MiB per message).
    builder.SetMaxReceiveMessageSize(256 << 20);
    builder.SetMaxSendMessageSize(256 << 20);

    auto server = builder.BuildAndStart();
    std::cout << "GEMM gRPC server listening on " << addr << "\n";
    server->Wait();
    return 0;
}
