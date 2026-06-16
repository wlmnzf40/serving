#!/bin/bash
# Start tensorflow_model_server directly (no Docker).
# gRPC port: 8500    REST port: 8501
#
# Build the binary first:
#   cd /path/to/serving && bazel build -c opt ... //tensorflow_serving/model_servers:tensorflow_model_server
#
# Then either:
#   export TF_MODEL_SERVER=/path/to/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
# or put the binary in PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/gemm_model"
PID_FILE="/tmp/tf-serving-gemm.pid"
LOG_FILE="/tmp/tf-serving-gemm.log"

# Locate binary: env var > PATH
if [ -n "${TF_MODEL_SERVER:-}" ]; then
    BINARY="${TF_MODEL_SERVER}"
elif command -v tensorflow_model_server &>/dev/null; then
    BINARY="tensorflow_model_server"
else
    echo "[ERROR] tensorflow_model_server not found."
    echo "        Set TF_MODEL_SERVER=/path/to/tensorflow_model_server"
    echo "        or add it to PATH."
    exit 1
fi

if [ ! -d "${MODEL_DIR}/1" ]; then
    echo "[ERROR] Model not found at ${MODEL_DIR}/1"
    echo "        Run: python3 generate_model.py"
    exit 1
fi

# Kill stale instance if any.
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "Stopping previous instance (PID=${OLD_PID}) ..."
        kill "${OLD_PID}"
    fi
    rm -f "${PID_FILE}"
fi

echo "Starting TF Serving ..."
"${BINARY}" \
    --port=8500 \
    --rest_api_port=8501 \
    --model_name=gemm \
    --model_base_path="${MODEL_DIR}" \
    > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "PID=$(cat ${PID_FILE})  log=${LOG_FILE}"

# Wait until REST endpoint is ready (up to 30 s).
echo -n "Waiting for server"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8501/v1/models/gemm >/dev/null 2>&1; then
        echo " ready!"
        echo ""
        echo "  gRPC -> localhost:8500"
        echo "  REST -> http://localhost:8501"
        echo ""
        echo "Stop with: ./stop_serving.sh"
        exit 0
    fi
    echo -n "."
    sleep 1
done
echo ""
echo "[WARN] Server did not respond within 30 s. Check log: ${LOG_FILE}"
