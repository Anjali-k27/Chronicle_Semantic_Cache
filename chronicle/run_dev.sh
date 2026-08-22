#!/usr/bin/env bash
# run_dev.sh — Chronicle Session 14.1 one-shot dev launcher
#
# Brings up the entire local stack in dependency order:
#   1. venv (created + deps installed on first run, reused after)
#   2. .env (copied from .env.example on first run — you still have to
#      paste in a real GEMINI_API_KEY before anything will actually call
#      Gemini; the script won't do that part for you)
#   3. 5 MCP data-source servers (ports 3001-3005)
#   4. Local Phoenix (port 6006) — without this, every request blocks for
#      ~13s per span while OTel retries a dead OTLP export target
#   5. The Chronicle API + UI (port 8000), in the foreground
#
# Ctrl+C stops the API server AND every background process this script
# started (MCP servers, Phoenix) — see the trap below. Re-running this
# script is safe: it skips anything already listening on its port.
#
# Usage: ./run_dev.sh   (from inside chronicle/)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── 1. venv ──────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "→ No .venv found — creating one and installing dependencies (first run only)..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
else
  source .venv/bin/activate
fi

# ── 2. .env ──────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "✗ Created .env from .env.example — it still has the placeholder key."
  echo "  Edit .env, set GEMINI_API_KEY to a real key from https://aistudio.google.com,"
  echo "  then re-run ./run_dev.sh."
  exit 1
fi

if grep -q "your_actual_key_here" .env; then
  echo ""
  echo "✗ .env still has the placeholder GEMINI_API_KEY (your_actual_key_here)."
  echo "  Edit .env with a real key, then re-run ./run_dev.sh."
  exit 1
fi

export $(grep -v '^#' .env | xargs)

# ── 3. MCP servers (idempotent — skips ports already bound) ─────────
# NOTE: plain indexed arrays, not `declare -A` — macOS ships bash 3.2
# (pre-GPLv3) by default, which has no associative arrays. Don't
# "modernize" this without checking `bash --version` on a real Mac first.
echo "→ Starting MCP data-source servers (3001-3005)..."
MCP_PORTS=(3001 3002 3003 3004 3005)
MCP_MODULES=(mcp_servers.spotify_server mcp_servers.finance_server mcp_servers.fitness_server mcp_servers.github_server mcp_servers.journal_server)
for i in 0 1 2 3 4; do
  port="${MCP_PORTS[$i]}"
  module="${MCP_MODULES[$i]}"
  if lsof -ti:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  · port $port already bound — leaving it alone"
  else
    uvicorn "${module}:app" --port "$port" --log-level warning &
  fi
done

# ── 4. Phoenix (idempotent) ──────────────────────────────────────────
if lsof -ti:6006 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "→ Phoenix already running on :6006 — leaving it alone"
else
  echo "→ Starting Phoenix on :6006..."
  python -m phoenix.server.main serve &
  PHOENIX_PID=$!
fi

# ── Cleanup on exit ───────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "→ Shutting down background services started by this script..."
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Wait for MCP + Phoenix health before starting the API ────────────
echo "→ Waiting for MCP servers + Phoenix to come up..."
for port in 3001 3002 3003 3004 3005 6006; do
  timeout=30
  until curl -sf "http://localhost:$port" >/dev/null 2>&1 || curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
    sleep 1
    timeout=$((timeout - 1))
    if [ "$timeout" -le 0 ]; then
      echo "✗ Port $port never came up — check the process didn't crash on startup."
      exit 1
    fi
  done
done
echo "✓ MCP servers + Phoenix are up."

# ── 5. The API + UI (foreground) ─────────────────────────────────────
echo "→ Starting Chronicle API + UI on :8000 (Ctrl+C to stop everything)..."
echo ""
uvicorn api:app --port 8000 --log-level info