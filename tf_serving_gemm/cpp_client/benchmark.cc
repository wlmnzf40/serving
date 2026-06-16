/*
 * GEMM benchmark against TF Serving REST API.
 *
 * Dependencies: libcurl, nlohmann/json (fetched by CMake)
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make -j$(nproc)
 *
 * Run:
 *   ./gemm_benchmark --M 512 --K 512 --N 512 --iters 50 --warmup 10
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// ── libcurl helpers ──────────────────────────────────────────────────────────

struct CurlResponse {
    std::string body;
    long http_code{0};
};

static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* resp = static_cast<CurlResponse*>(userdata);
    resp->body.append(ptr, size * nmemb);
    return size * nmemb;
}

// Performs a single HTTP POST and returns the response.
CurlResponse http_post(CURL* curl, const std::string& url,
                        const std::string& body) {
    CurlResponse resp;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        throw std::runtime_error(std::string("curl error: ") +
                                  curl_easy_strerror(rc));
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.http_code);
    curl_slist_free_all(headers);
    return resp;
}

// ── Matrix helpers ───────────────────────────────────────────────────────────

std::vector<float> random_matrix(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> m(rows * cols);
    for (auto& v : m) v = dist(rng);
    return m;
}

// Encode a matrix (row-major flat float vector) as a nested JSON 2-D array.
json matrix_to_json(const std::vector<float>& data, int rows, int cols) {
    json arr = json::array();
    for (int r = 0; r < rows; ++r) {
        json row = json::array();
        for (int c = 0; c < cols; ++c) {
            row.push_back(data[r * cols + c]);
        }
        arr.push_back(row);
    }
    return arr;
}

// ── Benchmark ────────────────────────────────────────────────────────────────

struct Config {
    std::string host{"localhost"};
    int port{8501};
    int M{512}, K{512}, N{512};
    int iters{50};
    int warmup{10};
    std::string model_name{"gemm"};
};

void run(const Config& cfg) {
    std::string url = "http://" + cfg.host + ":" + std::to_string(cfg.port) +
                      "/v1/models/" + cfg.model_name + ":predict";

    // Build request body once (same matrices for all iterations).
    std::mt19937 rng(42);
    auto A = random_matrix(cfg.M, cfg.K, rng);
    auto B = random_matrix(cfg.K, cfg.N, rng);

    json req_json;
    req_json["inputs"]["A"] = matrix_to_json(A, cfg.M, cfg.K);
    req_json["inputs"]["B"] = matrix_to_json(B, cfg.K, cfg.N);
    std::string req_body = req_json.dump();

    curl_global_init(CURL_GLOBAL_ALL);
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init() failed");

    // Check server is up with a GET.
    {
        std::string status_url = "http://" + cfg.host + ":" +
                                  std::to_string(cfg.port) + "/v1/models/" +
                                  cfg.model_name;
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        CurlResponse r;
        curl_easy_setopt(curl, CURLOPT_URL, status_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r);
        curl_easy_perform(curl);
        if (r.http_code != 200) {
            std::cerr << "Server returned HTTP " << r.http_code
                      << " — is TF Serving running?\n";
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            std::exit(1);
        }
        std::cout << "Server OK: " << status_url << "\n";
    }

    // Reset to POST mode.
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 0L);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    // Warmup.
    std::cout << "Warmup (" << cfg.warmup << " iters) ...\n" << std::flush;
    for (int i = 0; i < cfg.warmup; ++i) {
        auto resp = http_post(curl, url, req_body);
        if (resp.http_code != 200) {
            throw std::runtime_error("Predict failed HTTP " +
                                      std::to_string(resp.http_code) + ": " +
                                      resp.body);
        }
    }

    // Benchmark.
    std::cout << "Benchmarking (" << cfg.iters << " iters) ...\n" << std::flush;
    std::vector<double> latencies_ms(cfg.iters);
    for (int i = 0; i < cfg.iters; ++i) {
        auto t0 = Clock::now();
        auto resp = http_post(curl, url, req_body);
        double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();
        if (resp.http_code != 200) {
            throw std::runtime_error("Predict failed HTTP " +
                                      std::to_string(resp.http_code));
        }
        latencies_ms[i] = ms;
    }

    curl_easy_cleanup(curl);
    curl_global_cleanup();

    // Stats.
    std::sort(latencies_ms.begin(), latencies_ms.end());
    double avg = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0)
                 / latencies_ms.size();
    double p50 = latencies_ms[latencies_ms.size() / 2];
    double p99 = latencies_ms[static_cast<size_t>(latencies_ms.size() * 0.99)];
    double flops = 2.0 * cfg.M * cfg.N * cfg.K;
    double gflops = flops / (avg / 1e3) / 1e9;

    std::cout << "\n";
    std::cout << "==================================================\n";
    std::cout << "  Matrix sizes : (" << cfg.M << "x" << cfg.K << ") x ("
              << cfg.K << "x" << cfg.N << ")\n";
    std::cout << "  FLOPs/call   : " << flops / 1e9 << " GFLOP\n";
    std::cout << "  Latency avg  : " << avg  << " ms\n";
    std::cout << "  Latency p50  : " << p50  << " ms\n";
    std::cout << "  Latency p99  : " << p99  << " ms\n";
    std::cout << "  Throughput   : " << gflops << " GFLOPS\n";
    std::cout << "==================================================\n";
}

// ── CLI ──────────────────────────────────────────────────────────────────────

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " [--host H] [--port P] [--M m] [--K k] [--N n]"
                 " [--iters I] [--warmup W] [--model NAME]\n";
}

int main(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* flag) {
            return std::strcmp(argv[i], flag) == 0;
        };
        auto next_int = [&]() {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return std::stoi(argv[++i]);
        };
        auto next_str = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };

        if      (eq("--host"))   cfg.host       = next_str();
        else if (eq("--port"))   cfg.port        = next_int();
        else if (eq("--M"))      cfg.M           = next_int();
        else if (eq("--K"))      cfg.K           = next_int();
        else if (eq("--N"))      cfg.N           = next_int();
        else if (eq("--iters"))  cfg.iters       = next_int();
        else if (eq("--warmup")) cfg.warmup      = next_int();
        else if (eq("--model"))  cfg.model_name  = next_str();
        else { usage(argv[0]); return 1; }
    }

    try {
        run(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
