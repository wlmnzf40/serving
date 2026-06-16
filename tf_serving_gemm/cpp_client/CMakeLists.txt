cmake_minimum_required(VERSION 3.16)
project(gemm_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── libcurl (system) ──────────────────────────────────────────────────────────
find_package(CURL REQUIRED)

# ── nlohmann/json (fetched at configure time) ─────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Benchmark target ──────────────────────────────────────────────────────────
add_executable(gemm_benchmark benchmark.cc)
target_link_libraries(gemm_benchmark PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)
target_compile_options(gemm_benchmark PRIVATE -O2 -Wall -Wextra)
