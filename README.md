# Chronicle — Session 14.1: Semantic Cache + FinOps Cost Model

Chronicle is a 5-agent LangGraph swarm (FastAPI + OpenTelemetry + Arize
Phoenix) that produces a personal "life analysis" from five data sources
(Spotify, finance, fitness, GitHub, journal). Session 14.1 adds a
**semantic cache** in front of the swarm — a paraphrased repeat question
is served from cache in milliseconds instead of re-running the full
5-agent pipeline — plus a **FinOps cost model** that turns cache hit
rate into a monthly dollar figure, and a `served_hit_rate` SLO wired into
the Session 13.3 monitoring daemon.

This README covers three things:

1. **Cloning/setting this project up with zero extra steps.**
2. **Everything this project does, session by session**, from the first
   inference calculation (S11.1) through this session's semantic cache
   (S14.1).
3. **What's coming next** (S14.2 and beyond).

**This README lives at the repo root — the actual project is one level
down, in `chronicle/`.** Every command in this file, from §1 onward,
assumes your terminal's current directory is `chronicle/`, not wherever
this README is. §1 covers exactly how to get there; if you ever run a
command from this file and get a `No such file or directory` or
`ModuleNotFoundError`, the first thing to check is `pwd` — you're
almost certainly still at the repo root.

---

## Repo layout

```
<repo-root>/                    ← wherever `git clone` put things
├── README.md                    ← you are here
│
└── chronicle/                    ← the actual project — cd here first, always
    ├── run_dev.sh                  ← ONE command: brings up the entire stack (verified)
    ├── Dockerfile                    ← builds the api image (see §2d)
    ├── docker-compose.yml              ← phoenix + api + ui stack, fully verified (see §2d)
    ├── .env.example                     ← copy to .env, fill in your Gemini key
    ├── .gitignore
    ├── requirements.txt
    │
    ├── agent.py                          ← the 5-agent LangGraph swarm (S11.1–13.1)
    ├── api.py                             ← FastAPI gateway (S12.1–14.1)
    ├── otel_setup.py                       ← OpenTelemetry → Phoenix wiring (S13.1)
    ├── judge_pipeline.py                    ← LLM-as-Judge trajectory grading (S13.2)
    ├── monitoring_daemon.py                  ← SRE-style SLO/alerting daemon (S13.3–14.1)
    ├── semantic_cache.py                      ← THIS SESSION's deliverable (S14.1)
    ├── job_store.py                            ← async job state (S12.3)
    ├── stream_schemas.py                        ← SSE event schemas (S12.2)
    ├── index.html                                ← the UI, served by api.py at `/`
    │
    └── mcp_servers/                                ← 5 fake data-source MCP servers
        ├── spotify_server.py, finance_server.py, fitness_server.py,
        ├── github_server.py, journal_server.py
        └── start_all.sh                                ← starts all 5 (used by run_dev.sh)
```

---

## 0. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| Python 3.11+ | matches all pinned dependencies | `python3 --version` |
| A free Gemini API key | powers every agent, the embedder, and the LLM-as-Judge | https://aistudio.google.com → "Get API key" |
| Ports `3001`-`3005`, `6006`, `8000` free | 5 MCP servers, Phoenix, API+UI | `lsof -i :8000` etc. if something's wrong |
| macOS/Linux shell (`bash`) | `run_dev.sh` targets stock macOS `bash` (3.2) — no associative arrays, no bash-4-only syntax | — |

You do **not** need Docker, Redis, or a live Phoenix instance to run
`semantic_cache.py` or `monitoring_daemon.py` standalone — both have
fully offline/fixture-driven verification modes (see §2b).

---

## 1. Clone and set up (do this exactly once)

### Step 1 — Get the code

**If you already have a Git remote URL for this project** (GitHub,
GitLab, wherever your instructor or team put it) — copy that exact URL
and use it below. `<your-repo-url>` in the command is a **placeholder**;
if you run it exactly as written, with the angle brackets, it will fail
with `fatal: repository '<your-repo-url>' does not exist` — that error
means you forgot to substitute the real URL, not that anything is
broken.

```bash
git clone <your-repo-url> chronicle-project
cd chronicle-project
```

A filled-in example (yours will have a different URL — don't copy this
one, it's illustrative only):
```bash
git clone https://github.com/yourname/phase4-session10.git chronicle-project
cd chronicle-project
```

**If you don't have a Git remote yet** — for example you were just
handed this folder directly, or you're working from a local copy with
no `git clone` step at all — skip this step entirely. You don't need
Git or GitHub to run the project locally. Just open a terminal, `cd` to
wherever the folder containing `chronicle/` actually is, and continue
to Step 2.

**Verify you're in the right place before continuing, either way:**
```bash
ls chronicle/api.py chronicle/run_dev.sh
```
This should print both paths back with no error. If instead you see
`No such file or directory`, one of two things is true — either the
`git clone` above hasn't finished/failed silently (scroll up and check
for an error), or your terminal's current directory isn't the one
containing `chronicle/` yet. Run `pwd` and `ls` to see where you
actually are, then `cd` to the right place before moving on. Do not
proceed to Step 2 until this command succeeds.

### Step 2 — Move into the project directory

**Everything from here on — every command in the rest of this
README — assumes your current directory is `chronicle/` itself,** not
its parent:

```bash
cd chronicle
```

If you're not sure whether you've already done this, run `pwd` — it
should end in `.../chronicle`. Running `ls` should show `api.py`,
`run_dev.sh`, and `requirements.txt` directly (not inside another
subfolder).

### Step 3 — Virtual environment, dependencies, API key

```bash
# 1. Create an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and fill in your real key
cp .env.example .env
# edit .env — replace `your_actual_key_here` with your real GEMINI_API_KEY
```

**Do NOT run `pip install asyncio`.** It's part of the standard library
since Python 3.4; the PyPI package of the same name is an abandoned
pre-3.4 backport that conflicts with the stdlib module.

**Common first-run mistakes, in order of how often they actually
happen:**
- Forgetting `source .venv/bin/activate` before `pip install` — you'll
  install into your system Python instead, and every later `python
  api.py` / `./run_dev.sh` will mysteriously say `ModuleNotFoundError`
  even though `pip install` "worked." If a shell restart or a new
  terminal tab loses the venv, just re-run `source .venv/bin/activate`
  — it's required in every new terminal session, not just once ever.
- Leaving `.env`'s `GEMINI_API_KEY` as the literal placeholder
  `your_actual_key_here` — nothing will crash immediately, but every
  Gemini call will fail. See Troubleshooting §4D.
- Running any command in this README from the repo root instead of
  `chronicle/` (see Step 2 above).

That's the one-time setup. Everything below is how you *run* it.

---

## 2. Running things

### 2a. The whole stack, one command (recommended)

```bash
./run_dev.sh
```

This single script:
1. Creates `.venv` and installs dependencies if this is the very first
   run (skips this if `.venv` already exists).
2. Copies `.env.example` → `.env` if `.env` doesn't exist yet and tells
   you to add your real key (then exits — it won't guess your key for
   you).
3. Starts all 5 MCP data-source servers (skips any port already bound,
   so re-running it is safe).
4. Starts a local Phoenix instance on `:6006`. **This step matters more
   than it looks** — without it, OpenTelemetry's span exporter retries a
   dead connection with exponential backoff on *every single request*,
   which silently adds 10-15+ seconds to what should be instant
   responses. If Chronicle ever feels mysteriously slow, check Phoenix
   is actually running first.
5. Waits for all of the above to actually respond (not just "process
   started"), then starts the Chronicle API + UI on `:8000` in the
   foreground.

Open **http://localhost:8000** — that's the whole app. Phoenix's trace
explorer is at **http://localhost:6006**.

**Ctrl+C in that terminal stops everything** this script started (API,
MCP servers, Phoenix) — verified: it correctly tears down all 7
processes and frees all 7 ports. (If you ever run it detached/backgrounded
instead of in a foreground terminal, a plain `kill <script-pid>` will
*not* reach the child processes the same way Ctrl+C does — in that case
stop each port manually: `lsof -ti:8000,6006,3001,3002,3003,3004,3005 | xargs kill`.)

### 2b. Standalone: just the semantic cache (no server needed)

Runs the entire Session 14.1 deliverable — cost model, threshold
calibration, hit-rate simulation, and the cache-hit speed demo — with no
API server, no MCP servers, nothing else running.

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python semantic_cache.py
```

Expected: `6/6 checks passed`, a FinOps cost table, a threshold
calibration sweep, a hit-rate simulation landing in the 35-50% target
band, and:

```
[CACHE HIT] served in <100ms, bypassing the AI engine
SPEEDUP:     <N>×
```

### 2c. Standalone: just the monitoring daemon (no server needed)

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python monitoring_daemon.py
```

Expected: `6/6 checks passed`, then a degradation demo firing
`tool_hallucination`, `swarm_latency_exceeded`, `token_burn_rate_spike`,
and `trajectory_below_threshold_3x` alerts, ending with:

```
[CRITICAL ALERT] - Swarm Latency Exceeded. Paging On-Call Engineer.
```

### 2d. Docker

```bash
docker-compose up -d --build
```

Brings up `phoenix` (official image), `api` (built from the `Dockerfile`
here), and `ui` (nginx serving `index.html` statically on `:8080`, in
addition to `api.py`'s own `/` route on `:8000` — see the port note
below for why `ui` isn't on the usual `:3000`).

**Fully verified, all three services.** `docker-compose up -d --build`
was actually run against this `Dockerfile`/`docker-compose.yml` — image
built clean, `pip install -r requirements.txt` succeeded inside the
container, all three containers came up (`phoenix` reports `(healthy)`),
and:
```bash
curl http://localhost:6006/healthz        # → OK
curl http://localhost:8000/health/ready   # → {"status":"ready","graph":true,"mcp":true}
curl http://localhost:8000/health         # → "session":"14.1", cache_summary present
curl http://localhost:8080/               # → 200, serves index.html
```
all returned exactly that. Beyond health checks: a real question was
submitted through the **nginx-served UI in an actual headless browser**
(not curl) — `POST /analyze/async` queued in 612ms, the swarm ran for
real (against the calibration-dataset fallback — see the MCP caveat
below) and produced a genuine `final_result`. A close paraphrase
submitted right after rendered `status: completed` in **1 poll**
instead of walking the agent-progress UI again — a real cache hit,
confirmed both via the UI (`Semantic Cache` card: `served hits: 1`,
`hit rate: 33%`) and via `GET /health`'s `cache_summary` directly.
Zero browser console errors throughout.

**Port note**: `ui`'s host port is mapped to `8080`, not nginx's usual
`3000` — on the machine this was verified on, `3000` was already bound
by an unrelated `next-server` process from a different project, so
`docker-compose.yml` remaps it rather than fighting over the port.
Change it back to `"3000:80"` in `docker-compose.yml` if `3000` is free
on yours and you'd rather match the original session9 layout.

Ported from `session9/chronicle`'s Docker setup, which predates the
semantic cache — the only code change made here was adding
`semantic_cache.py` to the `Dockerfile`'s `COPY` line, since `api.py` now
imports it directly and the container would otherwise crash on startup.

**One thing that is NOT solved by Docker, in this session or any prior
one**: `docker-compose.yml` only ever defined `phoenix`, `api`, and
`ui` — nothing listens on `:3001`-`:3005` inside the compose network, so
the containerized `api`'s `MCPClientPool` silently falls back to the
calibration dataset instead of hitting real MCP data (confirmed: the
startup log line says `MCP pool ready: [...]` but every source falls
back until something answers on those ports). If you need to exercise
the actual live-MCP path — the whole point of Session 12.3 — use
`./run_dev.sh` instead, which really does start the 5 MCP servers.
Everything in §3's checklist was verified against `run_dev.sh`; Docker
gives you the same API behavior minus live MCP data.

---

## 3. Verification checklist

Run these after a build (`./run_dev.sh` or `docker-compose up`) to
confirm the whole chain — UI → API → semantic cache → 5-agent swarm →
OTel spans → Phoenix — actually works. Split into backend checks (curl)
and a dedicated UI checklist, since the UI is where a broken wire
between two working services usually shows up first — every checkbox
below was actually walked through in a real browser (twice: once
against `run_dev.sh`, once against the Docker `ui` container), not
written from reading the code.

### 3a. Backend (curl)

- [ ] `curl http://localhost:8000/health` → `"status":"ok"`, `"session":"14.1"`, `"version":"14.1.0"`, and a `cache_summary` object
- [ ] `curl http://localhost:6006/healthz` → `OK`
- [ ] `curl -X POST http://localhost:8000/analyze/async -H "Content-Type: application/json" -d '{"question": "test", "data_sources": ["fitness"]}'` → `202` with a `job_id` and `poll_url`
- [ ] Open http://localhost:6006 — find the `/analyze/async` span and
      confirm it carries `cache_hit`, `cache_bypass_reason`, and (on a
      hit) `cache_usd_saved` attributes

### 3b. UI — what to check in the browser, post build

Point your browser at **http://localhost:8000** (`run_dev.sh`, or
Docker's `api` container serving its own `/` route) or
**http://localhost:8080** (Docker's `ui` container, nginx — port `3000`
if you reverted the remap in §2d). Open devtools console before you
start; check it at the end too.

**On page load, before touching anything:**
- [ ] Header pill reads **`SESSION 14.1`** — not `12.3`, not blank.
      If you ever see a stale session number here again, check
      `index.html`'s `.session-pill` span and the FastAPI app's own
      `title`/`description`/`version` in `api.py` (both have been
      wrong independently before — fixing one doesn't fix the other).
- [ ] Header subtitle reads `Semantic Cache + FinOps Cost Model`
- [ ] Session-progress bar shows all of **S11.1 → S14.1** with green
      checkmarks, and **S14.1 is the active (blue/green) one** — not
      cut off at S12.3
- [ ] **Gateway** card (scroll the right panel down): `Session: 14.1`,
      `Version: 14.1.0`, `Uptime` counting up, `Graph: compiled ✓`
- [ ] **MCP Data Connectors** card lists all 5 sources (spotify,
      finance, fitness, github, journal). Via `run_dev.sh` these should
      flip to `LIVE` after the first real analysis; via Docker they'll
      stay on the calibration fallback — see §2d, not a bug
- [ ] **Semantic Cache** card is present (this is the new one, S14.1)
      showing `served hits`, `misses`, `hit rate`, `USD saved` — even
      at `0`/`0`/`0%`/`$0.0000` on a cold start, the card itself must
      exist and render without throwing
- [ ] Two placeholder cards below it — "Session 13.1 / OpenTelemetry
      Trace Viewer" and "Session 14.2 / Per-Agent Spend Ledger" — these
      are *supposed* to be inert placeholders, not broken widgets
- [ ] Devtools console: zero errors

**Submit a brand-new question via the orange "Queue (202)" button
(not the blue streaming one):**
- [ ] Chat shows `✓ Queued in <N>ms` where N is comfortably under
      100 — if it's several seconds, the cache check itself is
      blocking (see `api.py`'s `analyze_async`, the whole point of
      checking `cache.get()` directly instead of the
      `semantic_cache_dep()` convenience wrapper)
- [ ] Async Job Queue card appears with a real `job_id` and
      `status: queued`
- [ ] Right-hand **Agent Status** panel transitions each of
      `ingestion → pattern → timeline → brutality → synthesis` from
      `IDLE` to `DONE` in order as `active_node` updates (poll every
      few seconds — this takes ~60-120s for a real swarm run, that's
      expected, not a hang)
- [ ] A `SYNTHESIS` message bubble appears in the chat with real
      analysis text, a `confidence` value, and a non-zero
      `processing_ms`

**Then submit a close paraphrase of that same question:**
- [ ] Chat shows a fast `✓ Queued in <N>ms` again — this is still a
      real request, the *response* is what changes, not the ack
- [ ] Job card's `status` reads `completed` almost immediately —
      **`polls` should read `1`**, not climb into double digits.
      (If you see it polling for a while and eventually timing out
      instead, the cache hit isn't writing a job record the poll
      endpoint can find — this exact bug existed once and is fixed,
      see §5's S14.1 section.)
- [ ] The synthesis answer renders instantly with all 5 agents
      flipping straight to `DONE`, skipping the gradual
      `IDLE → running → DONE` walk you saw on the first question
- [ ] **Semantic Cache** card's `served hits` is now ≥ 1 and `hit rate`
      is > 0% — reload the page and confirm it's still there
      (proves it's reading live server state, not a client-side
      artifact that resets on refresh)
- [ ] Devtools console: still zero errors

If a paraphrase you try doesn't hit, that isn't automatically a bug —
cosine similarity between short/differently-phrased questions can
genuinely land below the `0.87` threshold (measured `0.83` for two
reasonable-sounding paraphrases during testing). Use a paraphrase that
keeps the same concrete entities and sentence shape as the original if
you want a reliable hit for this checklist.

---

## 4. Troubleshooting

**A. Everything feels slow — every request takes 10+ seconds**
Phoenix isn't running. `otel_setup.py` uses a `SimpleSpanProcessor`,
which exports every span *synchronously* and retries with exponential
backoff (~13s total) if `localhost:6006` refuses the connection. Start
Phoenix (`./run_dev.sh` does this for you) or check `lsof -i :6006`.

**B. `ModuleNotFoundError: No module named 'numpy'` (or similar) running `semantic_cache.py`**
You're not in the venv, or you installed dependencies before `numpy` was
added to `requirements.txt`. Re-run `pip install -r requirements.txt`
inside the venv.

**C. Embedding calls fail with `404 models/text-embedding-004 is not found`**
Google retired `text-embedding-004`. `semantic_cache.py` already targets
`models/gemini-embedding-001` instead — if you see this error, you're
looking at an older copy of the file, or a course guide that predates
this fix. The cosine-similarity threshold (`DEFAULT_COSINE_THRESHOLD =
0.87`) is calibrated specifically for this newer model's score
distribution, which runs lower than `text-embedding-004` did — don't
copy a `0.95` threshold from older material without re-running
`run_threshold_sweep()` first.

**D. Gemini calls fail / judge falls back to `7.0` every time**
`.env` still has the placeholder key:
```bash
grep GEMINI_API_KEY .env
```
should NOT show `your_actual_key_here`.

**E. Port already in use (`3001`-`3005`, `6006`, or `8000`)**
```bash
lsof -i :8000        # find what's holding it
kill $(lsof -ti:8000)
```
`run_dev.sh` skips starting a server on any port already bound, so a
half-stopped previous run won't error out — but it also won't restart
that server with your latest code changes. Kill stale processes first
if you've edited `api.py`, `agent.py`, or any MCP server file.

**F. `/health`'s `mcp_sources` all show `"connected": false` even though the servers are up**
Expected — `connected` only flips to `true` after a request actually
calls out to that source successfully (in `ingestion_node()`), not just
because the MCP server process is reachable. Run one real analysis and
check again.

**G. Editing `run_dev.sh` and adding `declare -A` (associative arrays)**
Don't. macOS ships `bash` 3.2 (the last GPLv2 release Apple can legally
distribute) as `/bin/bash` and as whatever `#!/usr/bin/env bash`
resolves to unless the user has installed a newer bash themselves.
Associative arrays are a bash-4+ feature and will hard-fail with
`declare: -A: invalid option` on stock macOS. This script already hit
that bug once — it now uses plain indexed arrays. Verify with
`bash --version` before adding any bash-4+ syntax.

---

## 5. Step-by-step: what this entire project is, session by session

Chronicle is built incrementally — each session is additive (nothing
gets removed, everything from a prior session keeps working) and is a
pure *consumer* of what came before it wherever possible.

### S11.1 — Concurrent 5-agent inference + VRAM budget
The first version: 5 specialized agents (`ingestion`, `pattern`,
`timeline`, `brutality`, `synthesis`) calling Gemini directly and
concurrently. `calculate_chronicle_vram_budget()` computes what it would
cost in VRAM to self-host all 5 agents at a single uniform precision.

### S11.2 — Tiered quantization + cost model
Not every agent needs the same precision. `calculate_tiered_vram_budget()`
assigns `int4` to the 3 cheap "utility" agents (ingestion/pattern/timeline)
and `fp16` to the 2 "frontier" agents (brutality/synthesis) that need real
reasoning quality. `calculate_monthly_gpu_cost()` and
`task_survivability_matrix()` turn that into a dollar figure and a
per-task risk matrix.

### S11.3 — GPU allocation, OOM prevention, co-location
`oom_prevention_check()` and `calculate_max_safe_concurrent()` compute
how many concurrent requests each agent's GPU can hold before running out
of VRAM, given its locked `max_model_len`. `colocation_partitioner()`
packs multiple utility agents onto one shared GPU. `vllm_config_per_agent()`
generates the actual `vllm serve` launch flags per agent.

### S12.1 — FastAPI gateway + live MCP data + LangGraph
The 5 agents become a **compiled LangGraph graph** (`build_chronicle_graph()`),
built once at FastAPI startup via `lifespan()`, not per-request.
`MCPClientPool` pulls live data from 5 MCP servers (falling back to a
calibration dataset if a source is unreachable). `POST /analyze` runs the
graph via `ainvoke()` and returns the final brief as JSON.

### S12.2 — Live SSE streaming
`POST /analyze/stream` replaces a 501 stub with a real Server-Sent
Events stream — `graph.astream()` yields an event after every node,
`chronicle_stream_events()` turns each into an SSE frame, disconnect
detection stops the graph early if the client leaves, and a keepalive
comment-frame prevents proxies from killing an idle connection during a
slow LLM call.

### S12.3 — Async 202 job queue + real MCP servers
Real MCP servers (actual FastAPI processes on ports 3001-3005) replace
the in-process fallback. Since a full analysis now takes 60-90 seconds —
past most API gateway timeouts — `POST /analyze/async` returns `202` +
a `job_id` in under 100ms and runs `run_chronicle_analysis()` via
`BackgroundTasks`, writing live progress to `job_store.py`.
`GET /analyze/jobs/{job_id}` polls it. An idempotency key (hash of
question + data sources) collapses duplicate submissions from retrying
clients into the same job.

### S13.1 — OpenTelemetry instrumentation
`otel_setup.py` wires a `TracerProvider` + OTLP exporter. Every LangGraph
node gets its own span; `token_count`, `temperature`, and
`langgraph_thread_id` are stamped as attributes. `otel_context` is
captured before `202` returns and re-attached inside the `BackgroundTasks`
coroutine, so the background analysis shows up as a *child* span of the
original HTTP request instead of an orphan trace.

### S13.2 — Phoenix + LLM-as-Judge
Spans export to a local **Arize Phoenix** instance (`localhost:6006`) via
OTLP. `judge_pipeline.py` implements an "extract-once, judge-many"
pipeline: pull a trajectory once from Phoenix, run **three independent
judge rubrics** against the same frozen artifact in parallel, write
verdicts back as span annotations. `grade_trajectory()` is the entry
point; `run_nightly_judge_pipeline()` is the fleet-wide batch version.

### S13.3 — SRE-style monitoring daemon
`monitoring_daemon.py`: every alert is declared as an `SLO` object up
front (`SWARM_SLOS`) — no alert ships without one. Five tripwires:
`TokenBurnRateCalculator` (rolling rate **AND** per-ticket mean — either
alone is noisy), `ToolHallucinationTripwire` (a called tool not in the
agent's registry pages immediately, no window), a two-window-hysteresis
latency p95 check (needs two consecutive breaching windows so one
cold-start doesn't page anyone at 3am), and `ThreeStrikesJudge` (three
consecutive judge scores below threshold, or one catastrophic score,
trips it). `CooldownRegistry` prevents alert spam per-signature without
masking a second, unrelated failure.

### S14.1 — Semantic cache + FinOps cost model *(this session)*
**New file `semantic_cache.py`:**
- `TokenCostModel` — explicit per-ticket/per-month token cost arithmetic;
  `print_finops_table()` shows the multiplier across DAU tiers and the
  dollar saving from a 40% cache hit rate.
- `GeminiEmbedder` — wraps `embed_content()`, L2-normalises every vector
  (so cosine similarity is a plain dot product), plus an LRU for
  literal-text repeats.
- `SemanticCache` / `CacheEntry` — a dict-backed cache (stand-in for a
  Redis Stack HNSW index in production). `CacheEntry.is_fresh()` enforces
  three independent gates: TTL, policy version, model version — any one
  failing is a miss. `bump_policy_version()` is an explicit, cheap
  invalidation lever.
- `run_threshold_sweep()` — calibrates the cosine similarity threshold
  against a labeled set of duplicate/non-duplicate ticket pairs, printing
  a TP/FP curve per threshold.

**Wired into `api.py`:** the cache is built once in `lifespan()`.
`POST /analyze/async` checks `cache.get()` **before** the idempotency
check and before scheduling the background swarm run — a hit returns
synchronously with zero graph/MCP/LLM cost. `cache_hit` and
`cache_bypass_reason` are stamped on the request's OTel span in both
paths. On a miss, the cache is populated **after** the real swarm
finishes (`_run_chronicle_analysis_and_cache()`), from the swarm's actual
`final_brief` — not from a stand-in response.

**Wired into `monitoring_daemon.py`:** a new `served_hit_rate` SLO
(5th in `SWARM_SLOS`) plus a matching tripwire in `tick()` that pages
`cache_hit_rate_collapsed` if the rolling hit rate drops below 20% once
at least 20 labelled requests have been seen (the 20-request floor
exists so a cold cache at startup doesn't page anyone).

**Two bugs caught by actually running this, not just reading it:**
- The spec's own example wired a cache-miss handler
  (`semantic_cache_dep()`) that called a stand-in `swarm_invoke_stub()`
  directly — which makes its own blocking Gemini call and would have (a)
  broken the `/analyze/async` endpoint's <100ms ack guarantee, and (b)
  cached a fake stand-in answer instead of the real swarm's output.
  Fixed by checking `cache.get()` directly on the hot path and populating
  the cache asynchronously from the real completed job.
- A cache hit returned a `job_id` that was never written to
  `job_store` — so the existing frontend's poll loop would 404 on its
  first poll and eventually show "Polling timeout" for the exact case
  (an instant cache hit) it should have shown off best. Fixed by writing
  a completed `JobRecord` synchronously on every cache hit.

Both were only found by actually driving the app end-to-end (curl, then
a real headless-browser walkthrough) — not by reading the code.

---

## 6. What's next

**Session 14.2 — LiteLLM + model routing** (per the handoff notes in
`semantic_cache.py`): a `model_router.py` selecting the cheapest viable
model per ticket tier (simple / complex / critical). The cache's miss
path calls `model_router.route()` instead of going straight to the
swarm. `cache_usd_saved` and a new `routing_model` attribute get added
to the OTel span. Nothing in `semantic_cache.py` itself is expected to
change — `SemanticCache`, `CacheEntry`, `GeminiEmbedder`, and
`cosine_similarity()` are all marked permanent/stable in this session's
handoff comment.

Beyond 14.2, this repo doesn't document further sessions yet — treat
anything past that as unconfirmed until it actually lands.
