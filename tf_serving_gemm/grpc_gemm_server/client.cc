/*
 * GEMM gRPC client / benchmark driver.
 *
 * Two modes:
 *   compute  Send actual matrix data, measure end-to-end + server compute.
 *   sweep    Ask server to benchmark a list of square sizes (pure server-side).
 *
 * Build: see BUILD.md
 * Run:
 *   # sweep (default):
 *   ./gemm_client --mode=sweep --sizes=128,256,512,1024,2048 --iters=50
 *
 *   # single compute call:
 *   ./gemm_client --mode=compute --M=1024 --K=1024 --N=1024 --iters=50
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "gemm.grpc.pb.h"
#include "gemm.pb.h"

using Clock = std::chrono::steady_clock;
using Stub  = gemm::GEMMService::Stub;

// ── helpers ───────────────────────────────────────────────────────────────────

static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> v;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stoi(tok));
    return v;
}

struct Stats { double avg, p50, p99; };

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    return {sum / v.size(), v[v.size() / 2], v[v.size() * 99 / 100]};
}

// ── compute mode ──────────────────────────────────────────────────────────────

static void run_compute(Stub& stub, int M, int K, int N, int iters, int warmup) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist;

    gemm::ComputeRequest req;
    req.set_m(M); req.set_k(K); req.set_n(N);
    for (int i = 0; i < M * K; ++i) req.add_a_data(dist(rng));
    for (int i = 0; i < K * N; ++i) req.add_b_data(dist(rng));

    auto call = [&]() -> std::pair<double, double> {
        gemm::ComputeResponse resp;
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));
        auto t0 = Clock::now();
        auto st = stub.Compute(&ctx, req, &resp);
        double rtt = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (!st.ok()) throw std::runtime_error(st.error_message());
        return {rtt, resp.server_compute_ms()};
    };

    // Warmup.
    std::cout << "Warmup (" << warmup << " iters)..." << std::flush;
    for (int i = 0; i < warmup; ++i) call();
    std::cout << " done\n";

    // Benchmark.
    std::cout << "Benchmarking (" << iters << " iters)...\n";
    std::vector<double> rtt_lats(iters), srv_lats(iters);
    for (int i = 0; i < iters; ++i) {
        auto [rtt, srv] = call();
        rtt_lats[i] = rtt;
        srv_lats[i] = srv;
    }

    auto rtt_s = compute_stats(rtt_lats);
    auto srv_s = compute_stats(srv_lats);
    double flops  = 2.0 * M * K * N;
    double gflops = flops / (srv_s.avg / 1e3) / 1e9;

    std::cout << "\n"
              << "Matrix: (" << M << "x" << K << ") x (" << K << "x" << N << ")\n"
              << "  Client RTT  : avg=" << rtt_s.avg << " ms"
              << "  p50=" << rtt_s.p50 << " ms"
              << "  p99=" << rtt_s.p99 << " ms\n"
              << "  Server pure : avg=" << srv_s.avg << " ms"
              << "  p50=" << srv_s.p50 << " ms"
              << "  p99=" << srv_s.p99 << " ms\n"
              << "  Throughput  : " << gflops << " GFLOPS (server-side)\n";
}

// ── sweep mode ────────────────────────────────────────────────────────────────

static void run_sweep(Stub& stub, const std::vector<int>& sizes,
                       int iters, int warmup) {
    gemm::SweepRequest req;
    for (int s : sizes) req.add_sizes(s);
    req.set_iters(iters);
    req.set_warmup(warmup);

    gemm::SweepResponse resp;
    grpc::ClientContext ctx;
    // Large sweeps can take minutes; use a generous deadline.
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(30));

    std::cout << "Requesting sweep from server: sizes=[";
    for (int i = 0; i < (int)sizes.size(); ++i)
        std::cout << sizes[i] << (i + 1 < (int)sizes.size() ? "," : "");
    std::cout << "] iters=" << iters << " warmup=" << warmup << "\n";

    auto st = stub.Sweep(&ctx, req, &resp);
    if (!st.ok()) {
        std::cerr << "[ERROR] Sweep failed: " << st.error_message() << "\n";
        return;
    }

    // Print results table.
    const int W = 12;
    std::cout << "\n"
              << std::left
              << std::setw(W)   << "Size"
              << std::setw(W)   << "Avg(ms)"
              << std::setw(W)   << "P50(ms)"
              << std::setw(W)   << "P99(ms)"
              << std::setw(W)   << "GFLOPS"
              << "\n"
              << std::string(W * 5, '-') << "\n";

    for (const auto& r : resp.results()) {
        std::string size_str =
            std::to_string(r.m()) + "x" + std::to_string(r.n());
        std::cout << std::left  << std::setw(W) << size_str
                  << std::fixed << std::setprecision(3)
                  << std::setw(W) << r.avg_ms()
                  << std::setw(W) << r.p50_ms()
                  << std::setw(W) << r.p99_ms()
                  << std::setw(W) << r.gflops()
                  << "\n";
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string host      = "localhost:50051";
    std::string mode      = "sweep";
    std::string sizes_str = "128,256,512,1024,2048";
    int M = 512, K = 512, N = 512;
    int iters = 50, warmup = 10;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](size_t n){ return a.substr(n); };
        if      (a.rfind("--host=",   0)==0) host      = val(7);
        else if (a.rfind("--mode=",   0)==0) mode      = val(7);
        else if (a.rfind("--sizes=",  0)==0) sizes_str = val(8);
        else if (a.rfind("--M=",      0)==0) M         = std::stoi(val(4));
        else if (a.rfind("--K=",      0)==0) K         = std::stoi(val(4));
        else if (a.rfind("--N=",      0)==0) N         = std::stoi(val(4));
        else if (a.rfind("--iters=",  0)==0) iters     = std::stoi(val(8));
        else if (a.rfind("--warmup=", 0)==0) warmup    = std::stoi(val(9));
        else { std::cerr << "Unknown flag: " << a << "\n"; return 1; }
    }

    auto channel = grpc::CreateChannel(host, grpc::InsecureChannelCredentials());
    auto stub    = gemm::GEMMService::NewStub(channel);

    try {
        if (mode == "compute")
            run_compute(*stub, M, K, N, iters, warmup);
        else if (mode == "sweep")
            run_sweep(*stub, parse_ints(sizes_str), iters, warmup);
        else {
            std::cerr << "Unknown mode: " << mode << "  (use: compute | sweep)\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
