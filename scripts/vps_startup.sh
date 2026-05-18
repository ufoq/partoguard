#!/bin/bash
# vps_startup.sh — PartoGuard llama-server boot script for the VPS.
#
# Purpose: relaunch llama-server with the V7 fine-tuned Gemma 4 E2B-it model
# (bf16 GGUF by default; Q8_0 fallback) plus its F16 mmproj after every
# container/VM restart.
#
# Hardware (verified 2026-05-17): 31 GiB RAM, 32× i9-13900K, 159 GiB swap,
# 915 GB /home, CPU-only inference. bf16 fits comfortably (~12 GB resident,
# 9.6 GiB swap baseline). Q8_0 kept as a smaller-RAM fallback.
#
# Install on the VPS as `/home/admin/.startup` (admin's docker host runs
# `.startup` on boot per project AGENTS.md). Only `/home` survives container
# rebuilds, so both the model files AND this script must live under /home/.
#
# Required files (must already exist on /home/admin/):
#   - v7_bf16.gguf                       (~9.27 GB, bf16; default)
#   - v7_q8_0.gguf                       (~4.93 GB, q8_0; fallback via $MODEL)
#   - v7_mmproj_f16.gguf                 (~986 MB, vision projector)
#   - llama.cpp/build/bin/llama-server   (built with libgomp.so.1 available)
#
# Override the model with:    MODEL=/home/admin/v7_q8_0.gguf bash .startup
#
# Verify after restart:
#   curl http://vps-box:8080/health                          -> {"status":"ok"}
#   curl http://vps-box:8080/props | jq .modalities.vision   -> true
#   curl http://vps-box:8080/v1/models | jq -r .data[0].id   -> v7_bf16.gguf
#
# Critical knobs:
#   -b 2048 -ub 2048   Vision encoder for Gemma 4 multimodal can emit ~1024+
#                       tokens per image; -ub MUST be >= image-token count or
#                       llama_context aborts with
#                       "non-causal attention requires n_ubatch >= n_tokens".
#                       (Reproduced 2026-05-17 with -ub 256.)
#   --jinja            Required for Gemma 4 chat template.
#   --no-mmproj-offload  Keep vision encoder on CPU (no GPU on this host).

set -euo pipefail

# Container rebuilds wipe /lib (only /home persists). If the system libgomp
# is gone but we have a stashed copy under /home/admin/lib, fall back to it
# via LD_LIBRARY_PATH so llama-server can still load. Reproduced 2026-05-17:
# fresh container had /lib/x86_64-linux-gnu/libgomp.so.1 missing entirely,
# llama-server failed at exec with "error while loading shared libraries".
if [[ ! -f /lib/x86_64-linux-gnu/libgomp.so.1 ]] && [[ -f /home/admin/lib/libgomp.so.1 ]]; then
    export LD_LIBRARY_PATH="/home/admin/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "Using stashed libgomp from /home/admin/lib (system copy missing)."
fi

LLAMA_BIN="${LLAMA_BIN:-/home/admin/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/admin/v7_bf16.gguf}"
MMPROJ="${MMPROJ:-/home/admin/v7_mmproj_f16.gguf}"
LOG_DIR="${LOG_DIR:-/home/admin/logs}"
LOG_FILE_LIVE="${LOG_FILE_LIVE:-/home/admin/llama-server.log}"
PID_FILE="${PID_FILE:-/home/admin/llama-server.pid}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-4096}"
# Half the cores keeps the box responsive and matches what tested OK at 8.7s/img.
THREADS="${THREADS:-16}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-2048}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/llama-server-$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ ! -x "${LLAMA_BIN}" ]]; then
    echo "ERROR: llama-server binary not found or not executable: ${LLAMA_BIN}" >&2
    exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
    echo "ERROR: model GGUF not found: ${MODEL}" >&2
    exit 1
fi
if [[ ! -f "${MMPROJ}" ]]; then
    echo "ERROR: mmproj GGUF not found: ${MMPROJ}" >&2
    exit 1
fi

# Clean up stale PID file if process is not running.
if [[ -f "${PID_FILE}" ]] && [[ -s "${PID_FILE}" ]]; then
    if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Removing stale PID file for dead process $(cat "${PID_FILE}")"
        rm -f "${PID_FILE}"
    fi
fi

if [[ -f "${PID_FILE}" ]] && [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "llama-server already running (pid $(cat "${PID_FILE}")). Exiting."
    exit 0
fi

# Belt + braces: kill any orphan llama-server that lost its pidfile.
pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
sleep 1

# Truncate the live log so it always reflects the latest boot.
: > "${LOG_FILE_LIVE}"

# setsid + nohup detaches from this shell so SSH session ending doesn't kill it.
setsid nohup "${LLAMA_BIN}" \
    -m "${MODEL}" \
    --mmproj "${MMPROJ}" \
    --no-mmproj-offload \
    -c "${CTX_SIZE}" \
    -t "${THREADS}" \
    -tb "${THREADS}" \
    -b "${BATCH}" \
    -ub "${UBATCH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --jinja \
    > "${LOG_FILE_LIVE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"
# Mirror the live log to a timestamped one for postmortems.
ln -sf "${LOG_FILE_LIVE}" "${LOG_FILE}" 2>/dev/null || cp "${LOG_FILE_LIVE}" "${LOG_FILE}" 2>/dev/null || true
echo "llama-server started: pid=${PID} model=${MODEL} log=${LOG_FILE_LIVE}"
