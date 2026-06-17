cmake_minimum_required(VERSION 3.16)
project(kblas_gemm CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── KML / KBLAS ───────────────────────────────────────────────────────────────
# Default install location used by KML RPM/DEB packages on Kunpeng.
# Override with:  cmake -DKML_ROOT=/path/to/kml ..
set(KML_ROOT "/usr/local/kml" CACHE PATH "KML installation root (contains include/ and lib/)")

find_path(KML_INCLUDE_DIR
    NAMES cblas.h
    PATHS ${KML_ROOT}/include
    NO_DEFAULT_PATH
    DOC "Directory containing KML cblas.h")

find_library(KML_BLAS_LIB
    NAMES kblas kml_blas
    PATHS ${KML_ROOT}/lib ${KML_ROOT}/lib64
    NO_DEFAULT_PATH
    DOC "KML BLAS shared/static library")

if(NOT KML_INCLUDE_DIR OR NOT KML_BLAS_LIB)
    message(FATAL_ERROR
        "KML not found. Set KML_ROOT to your KML installation directory.\n"
        "  cmake -DKML_ROOT=/path/to/kml ..\n"
        "Expected: ${KML_ROOT}/include/cblas.h and ${KML_ROOT}/lib/libkblas.so")
endif()

message(STATUS "KML include : ${KML_INCLUDE_DIR}")
message(STATUS "KML library : ${KML_BLAS_LIB}")

# ── Protobuf ──────────────────────────────────────────────────────────────────
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf ${Protobuf_VERSION}  protoc=${Protobuf_PROTOC_EXECUTABLE}")

# ── gRPC ──────────────────────────────────────────────────────────────────────
find_package(gRPC CONFIG QUIET)
if(gRPC_FOUND)
    message(STATUS "gRPC ${gRPC_VERSION} (cmake config)")
    set(GRPC_LIBS gRPC::grpc++ gRPC::grpc++_reflection)
    get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPCPP REQUIRED grpc++)
    message(STATUS "gRPC ${GRPCPP_VERSION} (pkg-config)")
    link_directories(${GRPCPP_LIBRARY_DIRS})
    set(GRPC_LIBS ${GRPCPP_LIBRARIES})
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
endif()

# ── Proto code generation ─────────────────────────────────────────────────────
set(PROTO_FILE       "${CMAKE_CURRENT_SOURCE_DIR}/proto/gemm.proto")
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(PROTO_CC "${PROTO_BINARY_DIR}/gemm.pb.cc")
set(PROTO_H  "${PROTO_BINARY_DIR}/gemm.pb.h")
set(GRPC_CC  "${PROTO_BINARY_DIR}/gemm.grpc.pb.cc")
set(GRPC_H   "${PROTO_BINARY_DIR}/gemm.grpc.pb.h")

add_custom_command(
    OUTPUT  ${PROTO_CC} ${PROTO_H} ${GRPC_CC} ${GRPC_H}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS    --cpp_out  "${PROTO_BINARY_DIR}"
            --grpc_out "${PROTO_BINARY_DIR}"
            --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/proto"
            "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating protobuf/gRPC stubs from gemm.proto"
)

add_library(gemm_proto STATIC ${PROTO_CC} ${GRPC_CC})
target_link_libraries(gemm_proto PUBLIC ${GRPC_LIBS} ${Protobuf_LIBRARIES})
target_include_directories(gemm_proto PUBLIC "${PROTO_BINARY_DIR}")

# ── Server ────────────────────────────────────────────────────────────────────
add_executable(kblas_gemm_server server.cc)

target_include_directories(kblas_gemm_server PRIVATE ${KML_INCLUDE_DIR})

target_link_libraries(kblas_gemm_server PRIVATE
    gemm_proto
    ${KML_BLAS_LIB}
    pthread)

# -O3 -march=native: let KML's NEON/SVE kernels fire on aarch64.
# -fopenmp: KML uses OpenMP for intra-GEMM parallelism.
target_compile_options(kblas_gemm_server PRIVATE -O3 -march=native -fopenmp)
target_link_options(kblas_gemm_server PRIVATE -fopenmp)

# Embed the KML library rpath so the binary can run without LD_LIBRARY_PATH.
set_target_properties(kblas_gemm_server PROPERTIES
    INSTALL_RPATH "${KML_ROOT}/lib;${KML_ROOT}/lib64"
    BUILD_RPATH   "${KML_ROOT}/lib;${KML_ROOT}/lib64")
