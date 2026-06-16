#!/bin/bash
PID_FILE="/tmp/tf-serving-gemm.pid"
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}" && echo "Stopped TF Serving (PID=${PID})"
    else
        echo "Process ${PID} not running"
    fi
    rm -f "${PID_FILE}"
else
    echo "No PID file found (${PID_FILE})"
fi
