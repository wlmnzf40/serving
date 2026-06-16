cmake_minimum_required(VERSION 3.16)
project(grpc_gemm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── Protobuf ──────────────────────────────────────────────────────────────────
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf ${Protobuf_VERSION}  protoc=${Protobuf_PROTOC_EXECUTABLE}")

# ── gRPC ──────────────────────────────────────────────────────────────────────
# Try cmake config first (installed via make install), then pkg-config.
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
message(STATUS "grpc_cpp_plugin: ${GRPC_CPP_PLUGIN}")

# ── Eigen (header-only, fetched) ──────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

# ── Proto code generation ─────────────────────────────────────────────────────
set(PROTO_FILE   "${CMAKE_CURRENT_SOURCE_DIR}/proto/gemm.proto")
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(PROTO_CC  "${PROTO_BINARY_DIR}/gemm.pb.cc")
set(PROTO_H   "${PROTO_BINARY_DIR}/gemm.pb.h")
set(GRPC_CC   "${PROTO_BINARY_DIR}/gemm.grpc.pb.cc")
set(GRPC_H    "${PROTO_BINARY_DIR}/gemm.grpc.pb.h")

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
add_executable(gemm_server server.cc)
target_link_libraries(gemm_server PRIVATE gemm_proto Eigen3::Eigen)
# -O3 -march=native: let Eigen/NEON do its best on aarch64
target_compile_options(gemm_server PRIVATE -O3 -march=native)

# ── Client ────────────────────────────────────────────────────────────────────
add_executable(gemm_client client.cc)
target_link_libraries(gemm_client PRIVATE gemm_proto)
target_compile_options(gemm_client PRIVATE -O2)
