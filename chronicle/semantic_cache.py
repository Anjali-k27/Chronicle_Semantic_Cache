"""
semantic_cache.py — Chronicle Semantic Cache
Session 14.1. Step 10 of the production infrastructure build.

Sits at the FastAPI Gateway, upstream of the LangGraph swarm.
On a hit: returns in ~40ms. Graph never touched. vLLM never called.
On a miss: falls through to the swarm; writes response back on the way out.

Consumes:
  - text-embedding-004 (Google) for cache key embeddings
  - numpy for cosine similarity
  - Pydantic for CacheEntry and TokenCostModel schemas

Produces:
  - cache_hit=true/false OTel span attribute (consumed by 13.3 daemon)
  - cache_bypass_reason attribute on every miss
  - cache_usd_saved on every hit (the CFO number)

Session 14.2 addition: the miss path routes to LiteLLM model selector.
Nothing in this file changes in 14.2.
"""

import asyncio
import json
import os
import random
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
import numpy as np
from pydantic import BaseModel, Field

# ── Config ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

MODEL       = "gemini-2.5-flash"
# NOTE: text-embedding-004 has been retired by Google (embedContent 404s on
# v1beta for current API keys). gemini-embedding-001 is the current stable
# embedding model — same embed_content() call surface, 3072-dim output.
EMBED_MODEL = "models/gemini-embedding-001"

# Calibrated for models/gemini-embedding-001 (text-embedding-004, the model
# this course originally shipped with, was retired by Google — see EMBED_MODEL
# above). gemini-embedding-001 produces a lower/wider cosine spread for
# genuine paraphrases than text-embedding-004 did, so the operating threshold
# is recalibrated accordingly — see run_threshold_sweep() for the measured
# TP/FP curve on LABELED_PAIRS (short single-sentence duplicates, the
# hardest case for this model). Full-length production tickets carry more
# shared entities/context and score meaningfully higher for true duplicates
# (~0.92 in run_end_to_end_demo) — always recalibrate against real traffic,
# not the synthetic short-phrase set alone.
DEFAULT_COSINE_THRESHOLD = 0.87
DEFAULT_TTL_SECONDS      = 86400  # 24h
POLICY_VERSION           = "v2025.01"
MODEL_VERSION            = MODEL + "::build-1"
USD_PER_MTOKEN           = 3.0    # blended hosted-model rate


# ── SECTION 1: Token Cost Model ────────────────────────────────────────────

class TokenCostModel(BaseModel):
    """
    Per-ticket and per-month token-spend calculator.
    All arithmetic is explicit so students can reproduce it by hand.
    Introduced: Session 14.1. Permanent.
    """
    tokens_per_call:       int   = 2000
    loops_per_ticket:      int   = 5
    dau:                   int   = 10_000
    tickets_per_dau_per_day: float = 1.0
    days_per_month:        int   = 30
    usd_per_mtoken:        float = USD_PER_MTOKEN
    cache_hit_rate:        float = 0.0   # 0.0 = no cache; 0.4 = 40% hits

    def tokens_per_ticket(self) -> int:
        return self.tokens_per_call * self.loops_per_ticket

    def tickets_per_month(self) -> int:
        return int(self.dau * self.tickets_per_dau_per_day * self.days_per_month)

    def effective_tickets_hitting_llm(self) -> int:
        return int(self.tickets_per_month() * (1.0 - self.cache_hit_rate))

    def monthly_tokens(self) -> int:
        return self.effective_tickets_hitting_llm() * self.tokens_per_ticket()

    def monthly_usd(self) -> float:
        return self.monthly_tokens() / 1_000_000 * self.usd_per_mtoken


def print_finops_table() -> None:
    """Print the token multiplier table across DAU tiers. Session 14.1."""
    print("\nTOKEN MULTIPLIER COST TABLE")
    print("=" * 80)
    print(f"  {'DAU':>10} {'tickets/day':>12} {'tokens/ticket':>14} {'monthly USD':>14}")
    print("-" * 80)
    for dau in [1_000, 10_000, 100_000, 1_000_000]:
        m = TokenCostModel(dau=dau)
        print(f"  {dau:>10,} {int(dau*m.tickets_per_dau_per_day):>12,} "
              f"{m.tokens_per_ticket():>14,} ${m.monthly_usd():>13,.0f}")

    print()
    print("WITH 40% SEMANTIC CACHE HIT RATE — 1M DAU:")
    m_raw    = TokenCostModel(dau=1_000_000, cache_hit_rate=0.0)
    m_cached = TokenCostModel(dau=1_000_000, cache_hit_rate=0.4)
    print(f"  Without cache: ${m_raw.monthly_usd():,.0f}/mo")
    print(f"  With cache:    ${m_cached.monthly_usd():,.0f}/mo")
    print(f"  Monthly saving: ${m_raw.monthly_usd()-m_cached.monthly_usd():,.0f}")
    print()


# ── SECTION 2: Gemini Embedder ──────────────────────────────────────────────

class GeminiEmbedder:
    """
    Wraps genai.embed_content with L2 normalisation and an LRU.

    L2 normalisation makes cosine similarity a plain dot product —
    both faster and what Redis Stack's HNSW expects with COSINE metric.

    The LRU prevents double-embedding the same literal text within one tick.
    It is an orthogonal optimisation to the semantic cache — the semantic
    cache matches on vector similarity; this LRU matches on literal equality.

    Introduced: Session 14.1. Permanent.
    """

    def __init__(self, model: str = EMBED_MODEL, lru_capacity: int = 1024):
        self.model        = model
        self._lru: OrderedDict = OrderedDict()
        self._capacity    = lru_capacity
        self.api_calls    = 0

    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised embedding vector for text."""
        if text in self._lru:
            self._lru.move_to_end(text)
            return self._lru[text]

        self.api_calls += 1
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
        )
        vec = np.asarray(result["embedding"], dtype=np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n

        self._lru[text] = vec
        if len(self._lru) > self._capacity:
            self._lru.popitem(last=False)
        return vec


# ── SECTION 3: Cosine Similarity and CacheEntry ─────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot-product cosine similarity for L2-normalised vectors.
    Assumes both inputs are already L2-normalised (GeminiEmbedder guarantees this).
    If that invariant is violated the result is not a valid cosine score.
    Introduced: Session 14.1. Permanent.
    """
    return float(np.dot(a, b))


class CacheEntry(BaseModel):
    """
    One entry in the semantic cache.
    The key (vector) is stored separately in SemanticCache._store.
    This object is the value — returned to the gateway on a hit.

    is_fresh() enforces three independent gates.
    All three must pass or the entry is treated as a miss.
    Introduced: Session 14.1. Permanent.
    """
    ticket_text_preview: str      # first 200 chars — for debugging only
    consensus_response:  str      # the user-facing reply
    internal_notes:      str = "" # operator-only context from Triage
    created_at:          float    # epoch seconds at write time
    ttl_seconds:         int      # expiry horizon
    policy_version:      str      # must match current APP_POLICY_VERSION
    model_version:       str      # must match current chat model id
    token_count_at_write: int = 0 # tokens the Swarm used — for cache_usd_saved

    def is_fresh(
        self,
        now:            float,
        current_policy: str,
        current_model:  str,
    ) -> bool:
        """
        Three-gate freshness check.
        A pass on one and a miss on another is still a miss overall.
        Introduced: Session 14.1. Permanent.
        """
        if now - self.created_at > self.ttl_seconds:
            return False   # TTL expired
        if self.policy_version != current_policy:
            return False   # policy was bumped since this entry was written
        if self.model_version != current_model:
            return False   # model was upgraded since this entry was written
        return True


# ── SECTION 4: SemanticCache ───────────────────────────────────────────────

class SemanticCache:
    """
    In-memory semantic cache with a dict-backed Redis shim.

    In production: replace self._store with a Redis Stack client.
    get() becomes an FT.SEARCH call against an HNSW index (sub-ms at 1M vectors).
    put() becomes a HSET call.
    The put/get surface is identical in both environments.

    The cache is placed at the FastAPI Gateway, upstream of the LangGraph swarm.
    A hit returns before app.invoke() is ever called.
    Zero graph compilation. Zero vLLM calls. Zero MCP fetches.

    Introduced: Session 14.1. Permanent.
    """

    def __init__(
        self,
        embedder:       GeminiEmbedder,
        threshold:      float = DEFAULT_COSINE_THRESHOLD,
        ttl_seconds:    int   = DEFAULT_TTL_SECONDS,
        policy_version: str   = POLICY_VERSION,
        model_version:  str   = MODEL_VERSION,
    ):
        self.embedder       = embedder
        self.threshold      = threshold
        self.ttl_seconds    = ttl_seconds
        self.policy_version = policy_version
        self.model_version  = model_version

        self._store: Dict[Tuple, CacheEntry] = {}

        # Counters — exposed as OTel span attributes
        self.served_hits    = 0
        self.attempted_hits = 0  # hits above threshold but failed is_fresh
        self.misses         = 0
        self.tokens_saved   = 0  # accumulates token_count_at_write from hits

    def _key_of(self, vector: np.ndarray) -> tuple:
        return tuple(float(x) for x in vector)

    def get(self, ticket_text: str) -> Optional[CacheEntry]:
        """
        Return the best-matching fresh entry, or None on miss.

        Colab: O(N) linear scan (no HNSW index in the dict shim).
        Production: FT.SEARCH against HNSW index — sub-millisecond at 1M vectors.

        Returns None on:
          - no stored entry with cosine >= threshold
          - best match failed is_fresh() (TTL, policy, or model mismatch)

        Introduced: Session 14.1. Permanent.
        """
        q = self.embedder.embed(ticket_text)

        best_score = -1.0
        best_entry: Optional[CacheEntry] = None

        for stored_key, entry in self._store.items():
            stored_vec = np.asarray(stored_key, dtype=np.float32)
            score      = cosine_similarity(q, stored_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.threshold:
            self.misses += 1
            return None

        self.attempted_hits += 1
        if not best_entry.is_fresh(time.time(), self.policy_version, self.model_version):
            return None   # stale hit counts as a miss for serving

        self.served_hits  += 1
        self.tokens_saved += best_entry.token_count_at_write
        return best_entry

    def put(
        self,
        ticket_text:        str,
        consensus_response: str,
        internal_notes:     str = "",
        token_count:        int = 0,
    ) -> CacheEntry:
        """
        Insert a new entry keyed on the embedding of ticket_text.
        Called by the gateway on every Swarm miss, after the Swarm returns.
        Introduced: Session 14.1. Permanent.
        """
        vec   = self.embedder.embed(ticket_text)
        entry = CacheEntry(
            ticket_text_preview  = ticket_text[:200],
            consensus_response   = consensus_response,
            internal_notes       = internal_notes,
            created_at           = time.time(),
            ttl_seconds          = self.ttl_seconds,
            policy_version       = self.policy_version,
            model_version        = self.model_version,
            token_count_at_write = token_count,
        )
        self._store[self._key_of(vec)] = entry
        return entry

    def evict_expired(self) -> int:
        """Sweep expired entries. Returns eviction count. Introduced: S14.1."""
        now   = time.time()
        stale = [k for k, e in self._store.items()
                 if now - e.created_at > e.ttl_seconds]
        for k in stale:
            del self._store[k]
        return len(stale)

    def bump_policy_version(self, new_version: str) -> int:
        """
        Explicit invalidation lever — cheaper than FLUSHDB.
        All entries with old version fail is_fresh() on next read.
        Returns count of entries that will become stale.
        Introduced: Session 14.1. Permanent.
        """
        old   = self.policy_version
        self.policy_version = new_version
        return sum(1 for e in self._store.values() if e.policy_version == old)

    def stats(self) -> dict:
        """Counters for OTel span attributes and 13.3 daemon. Introduced: S14.1."""
        total   = self.served_hits + self.misses
        usd_saved = self.tokens_saved / 1_000_000 * USD_PER_MTOKEN
        return {
            "served_hits":       self.served_hits,
            "attempted_hits":    self.attempted_hits,
            "misses":            self.misses,
            "served_hit_rate":   self.served_hits / total if total else 0.0,
            "tokens_saved":      self.tokens_saved,
            "usd_saved":         round(usd_saved, 4),
            "size":              len(self._store),
        }


# ── SECTION 5: Gateway Dependency ─────────────────────────────────────────

class TicketRequest(BaseModel):
    ticket_id:   str
    user_id:     str
    ticket_text: str


class TicketResponse(BaseModel):
    ticket_id:          str
    consensus_response: str
    internal_notes:     str
    served_from_cache:  bool
    cache_bypass_reason: str
    wallclock_ms:       float


def swarm_invoke_stub(ticket: TicketRequest) -> Tuple[str, str, int]:
    """
    Stand-in for the Chronicle LangGraph swarm (agent.py graph.ainvoke).
    Calls Gemini once to generate a plausible consensus_response.
    Returns (consensus_response, internal_notes, token_count).

    In production: replace with graph.ainvoke(build_initial_state(request, ...)).
    The caching architecture is identical regardless of what is inside the graph.
    Introduced: Session 14.1. Permanent.
    """
    model  = genai.GenerativeModel(MODEL)
    prompt = (
        f"You are a customer-support agent. "
        f"Given the ticket below, write a concise response (≤100 words) "
        f"that a human agent can send as-is.\n\nTicket: {ticket.ticket_text}"
    )
    resp      = model.generate_content(prompt)
    consensus = resp.text.strip()
    notes     = f"[internal] stub_triage; user={ticket.user_id}; len={len(consensus)}"
    # Approximate token count — use response.usage_metadata in production
    tokens    = len(prompt.split()) + len(consensus.split())
    return consensus, notes, tokens


def semantic_cache_dep(cache: SemanticCache, ticket: TicketRequest) -> TicketResponse:
    """
    FastAPI dependency body. Runs at the gateway, upstream of LangGraph.

    On a hit:  return CacheEntry.consensus_response. Graph never touched.
    On a miss: invoke Swarm, write response back, return Swarm answer.

    Stamps cache_hit and cache_bypass_reason before returning.
    The gateway knows, before returning, whether it hit or missed.
    These two attributes are what the 13.3 daemon reads as SLIs.

    NOTE on production wiring (see api.py): this convenience function's miss
    path calls swarm_invoke_stub() directly, which makes its own blocking
    Gemini call — fine for the standalone demo below, but NOT what api.py
    uses for /analyze/async. That endpoint must ack in <100ms, so it calls
    cache.get() directly and populates the cache asynchronously from the
    real LangGraph swarm's completed output instead of calling this
    function on the miss path. See api.py's _run_chronicle_analysis_and_cache().

    Introduced: Session 14.1. Permanent.
    """
    t0  = time.time()
    hit = cache.get(ticket.ticket_text)

    if hit is not None:
        # ── CACHE HIT ─────────────────────────────────────────────────────
        # Stamp cache_hit=True on the OTel span here in production:
        #   span.set_attribute("cache_hit", True)
        #   span.set_attribute("cache_bypass_reason", "none")
        #   span.set_attribute("cache_usd_saved", tokens_saved_this_hit / 1M * 3.0)
        return TicketResponse(
            ticket_id           = ticket.ticket_id,
            consensus_response  = hit.consensus_response,
            internal_notes      = hit.internal_notes,
            served_from_cache   = True,
            cache_bypass_reason = "none",
            wallclock_ms        = (time.time() - t0) * 1000,
        )

    # ── CACHE MISS — fall through to Swarm ────────────────────────────────
    # Stamp cache_hit=False + reason on OTel span in production:
    #   span.set_attribute("cache_hit", False)
    #   span.set_attribute("cache_bypass_reason", "below_threshold")
    consensus, notes, tokens = swarm_invoke_stub(ticket)
    cache.put(ticket.ticket_text, consensus, notes, token_count=tokens)

    return TicketResponse(
        ticket_id           = ticket.ticket_id,
        consensus_response  = consensus,
        internal_notes      = notes,
        served_from_cache   = False,
        cache_bypass_reason = "below_threshold",
        wallclock_ms        = (time.time() - t0) * 1000,
    )


# ── SECTION 6: Threshold Calibration ──────────────────────────────────────

class LabeledPair(BaseModel):
    query:  str
    stored: str
    label:  str   # 'dup' or 'nope'


LABELED_PAIRS: List[LabeledPair] = [
    # true duplicates — good hits
    LabeledPair(query="How do I reset my password?",           stored="I forgot my password, help me log back in",             label="dup"),
    LabeledPair(query="My password is not working",            stored="I need to reset my password",                           label="dup"),
    LabeledPair(query="Refund for damaged item",               stored="My item arrived broken, I want my money back",          label="dup"),
    LabeledPair(query="Where is my order?",                    stored="Tracking shows no updates on my package",               label="dup"),
    LabeledPair(query="Cancel my subscription",                stored="I want to stop my monthly plan",                        label="dup"),
    LabeledPair(query="Update my billing address",             stored="Change the address on my card",                         label="dup"),
    LabeledPair(query="Two-factor code not arriving",          stored="2FA SMS never comes",                                   label="dup"),
    LabeledPair(query="Export my data",                        stored="GDPR data download request",                            label="dup"),
    LabeledPair(query="Why was I charged twice?",              stored="Duplicate charge on my credit card",                    label="dup"),
    LabeledPair(query="Login fails on mobile",                 stored="I cannot sign in from my phone app",                    label="dup"),
    # genuinely different — wrong if they hit
    LabeledPair(query="How do I reset my password?",           stored="How do I request a refund?",                            label="nope"),
    LabeledPair(query="Where is my order?",                    stored="Where is the admin console?",                           label="nope"),
    LabeledPair(query="Cancel my subscription",                stored="Add a new seat to my team",                             label="nope"),
    LabeledPair(query="Export my data",                        stored="Import data from CSV",                                  label="nope"),
    LabeledPair(query="Update billing address",                stored="Update shipping address",                               label="nope"),
    LabeledPair(query="Login fails on mobile",                 stored="Mobile push notifications stopped",                     label="nope"),
    LabeledPair(query="Refund for damaged item",               stored="Return policy on opened boxes",                         label="nope"),
    LabeledPair(query="Two-factor code not arriving",          stored="Marketing emails not arriving",                         label="nope"),
]


def run_threshold_sweep(embedder: GeminiEmbedder) -> None:
    """
    Embed all labeled pairs and sweep thresholds 0.80–0.99.
    Prints the confusion matrix at each threshold.
    Run once per month; record chosen threshold + calibration_date.
    Introduced: Session 14.1. Permanent.
    """
    print("THRESHOLD CALIBRATION SWEEP")
    print("Embedding labeled pairs...")

    scored = []
    for p in LABELED_PAIRS:
        a     = embedder.embed(p.query)
        b     = embedder.embed(p.stored)
        score = cosine_similarity(a, b)
        scored.append((p, score))

    print(f"\nCOSINE SCORES ({len(scored)} pairs)")
    print("-" * 72)
    for p, s in scored:
        marker = "DUP " if p.label == "dup" else "NOPE"
        print(f"  {marker} {s:.4f}  {p.query[:32]:<32} <-> {p.stored[:30]}")

    dup_total  = sum(1 for p, _ in scored if p.label == "dup")
    nope_total = sum(1 for p, _ in scored if p.label == "nope")

    print(f"\n{'thr':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'hit_rate':>10} {'wrong_ans':>10}")
    print("-" * 44)
    for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.87]:
        tp = sum(1 for p, s in scored if p.label == "dup"  and s >= t)
        fp = sum(1 for p, s in scored if p.label == "nope" and s >= t)
        fn = dup_total - tp
        print(f"  {t:>4.2f} {tp:>4} {fp:>4} {fn:>4} "
              f"{tp/dup_total*100:>9.1f}% {fp/nope_total*100:>9.1f}%")
    print()


# ── SECTION 7: Hit Rate Simulation ────────────────────────────────────────

TRAFFIC_TEMPLATES = [
    ["How do I reset my password?",                 "I forgot my password, help me log back in",
     "Password reset link never arrived",            "My password is not working",
     "Reset password flow is broken",               "Cannot sign in, tried reset, still locked"],
    ["I want a refund for my damaged item",          "My order arrived broken, please refund",
     "How do I get my money back for a defective product?", "Refund for a defective unit",
     "Return a damaged package",                    "My item was broken in shipping, give me a refund"],
    ["Where is my order?",                          "Shipment tracking shows no updates",
     "Package has not moved in 5 days",             "Order status stuck on pending",
     "Delivery is late by a week",                  "Tracking link says label created but nothing else"],
    ["How do I cancel my subscription?",            "I want to stop my monthly plan",
     "Cancel my annual subscription effective immediately", "Turn off auto-renew",
     "End my membership please",                    "Unsubscribe me from the paid plan"],
    ["Why was I charged twice?",                    "Duplicate charge on my credit card",
     "I see two identical line items on my bill",   "Billed twice for the same invoice",
     "Double-charged this month",                   "Two charges for one order"],
]


def build_synthetic_traffic(n: int, seed: int = 42) -> List[str]:
    """
    80% template paraphrases + 20% unique one-off tickets.
    Simulates the empirical 40%-hit-rate target.
    Introduced: Session 14.1. Permanent.
    """
    rng     = random.Random(seed)
    tickets = []
    for _ in range(n):
        if rng.random() < 0.80:
            tmpl = rng.choice(TRAFFIC_TEMPLATES)
            tickets.append(rng.choice(tmpl))
        else:
            tickets.append(
                f"Unique ticket #{rng.randint(0, 10**9)} about "
                f"an edge case {rng.choice(['widget','gadget','doohickey'])}"
            )
    rng.shuffle(tickets)
    return tickets


def run_hit_rate_simulation(cache: SemanticCache) -> None:
    """
    Run 60 synthetic tickets through the cache. Print hit/miss counts.
    Introduced: Session 14.1. Permanent.
    """
    print(f"HIT RATE SIMULATION (60 tickets, 80/20 template/unique, threshold {cache.threshold})")
    tickets = build_synthetic_traffic(60, seed=7)
    hits = misses = 0

    for t in tickets:
        entry = cache.get(t)
        if entry is not None:
            hits += 1
        else:
            misses += 1
            cache.put(t, f"[stub] response for: {t[:60]}")

    print(f"  hits:     {hits}")
    print(f"  misses:   {misses}")
    print(f"  HIT RATE: {hits/(hits+misses)*100:.1f}%  (target band: 35–50%)")
    print()


# ── SECTION 8: 15s → 40ms End-to-End Demo ─────────────────────────────────

FIRST_TICKET = (
    "Hi team, I tried to log in this morning and my password no longer works. "
    "I clicked reset but the email never arrived in my inbox or spam folder. "
    "My customer ID is CUST-99421. I urgently need dashboard access for a "
    "CFO report due today. Please help me regain access as soon as possible."
)

REPHRASED_TICKET = (
    "Hello, I need urgent help getting into my account. The password that used "
    "to work is suddenly rejected, and I requested a reset link but have not "
    "seen it. Customer ID CUST-99421. I have a report due to my CFO today. "
    "Can you please help me get back in immediately?"
)


def run_end_to_end_demo(embedder: GeminiEmbedder) -> None:
    """
    First ticket: cold cache → full Swarm invoke (~15s).
    Rephrased ticket: warm cache → cosine hit (~40ms).
    Prints the session-promised [CACHE HIT] banner.
    Introduced: Session 14.1. Permanent.
    """
    cache = SemanticCache(embedder, threshold=DEFAULT_COSINE_THRESHOLD)

    print("=" * 72)
    print("END-TO-END DEMO — semantic cache in front of the Chronicle swarm")
    print("=" * 72)
    print()

    # Ticket 1 — cold cache
    t0   = time.time()
    req1 = TicketRequest(ticket_id="T-001", user_id="alice", ticket_text=FIRST_TICKET)
    r1   = semantic_cache_dep(cache, req1)
    dt1  = time.time() - t0

    print(f">>> TICKET 1 (cold cache)")
    print(f"    served_from_cache = {r1.served_from_cache}")
    print(f"    wallclock         = {dt1*1000:.0f}ms")
    print(f"    response preview  : {r1.consensus_response[:80]}...")
    print()

    # Ticket 2 — rephrased → should hit
    t0   = time.time()
    req2 = TicketRequest(ticket_id="T-002", user_id="bob", ticket_text=REPHRASED_TICKET)
    r2   = semantic_cache_dep(cache, req2)
    dt2  = time.time() - t0

    print(f">>> TICKET 2 (rephrased)")
    print(f"    served_from_cache = {r2.served_from_cache}")
    print(f"    wallclock         = {dt2*1000:.0f}ms")

    if r2.served_from_cache:
        print(f"    [CACHE HIT] served in {dt2*1000:.0f}ms, bypassing the AI engine")
    else:
        print(f"    cache MISS — threshold may be too tight for this paraphrase pair")

    print()
    print("-" * 72)
    if dt2 > 0:
        print(f"SPEEDUP:     {dt1/dt2:.0f}×")
    print(f"CACHE STATS: {cache.stats()}")
    print("-" * 72)
    print()


# ── SECTION 9: Verification ────────────────────────────────────────────────

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 14.1 — VERIFICATION TEST                           │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │    - TokenCostModel arithmetic is correct                   │
    │    - CacheEntry.is_fresh() enforces all three gates         │
    │    - SemanticCache.get() returns None below threshold       │
    │    - bump_policy_version() invalidates old entries on read  │
    │    - GeminiEmbedder produces L2-normalised vectors          │
    │    - cosine_similarity(v, v) ≈ 1.0                         │
    ├─────────────────────────────────────────────────────────────┤
    │  PASS CRITERIA:                                             │
    │    ✓ TokenCostModel(dau=100000).monthly_usd() ≈ $90,000     │
    │    ✓ Expired CacheEntry → is_fresh() returns False          │
    │    ✓ Old policy_version → is_fresh() returns False          │
    │    ✓ get() returns None for empty cache                     │
    │    ✓ bump_policy_version() causes subsequent get() to miss  │
    │    ✓ L2-normalised vector has norm ≈ 1.0                    │
    └─────────────────────────────────────────────────────────────┘
    """
    import time as _time
    checks = []
    start  = _time.monotonic()

    # CHECK 1: TokenCostModel arithmetic
    m      = TokenCostModel(dau=100_000, cache_hit_rate=0.0)
    cost   = m.monthly_usd()
    cost_ok = 85_000 <= cost <= 95_000   # ≈ $90,000
    checks.append({
        "label":  "TokenCostModel(dau=100K).monthly_usd() ≈ $90,000",
        "passed": cost_ok,
        "note":   f"Got ${cost:,.0f}",
    })

    # CHECK 2: is_fresh() — TTL gate
    e = CacheEntry(
        ticket_text_preview="test", consensus_response="test",
        created_at=_time.time() - 90_000,   # 25 hours ago — past 24h TTL
        ttl_seconds=86400, policy_version=POLICY_VERSION, model_version=MODEL_VERSION,
    )
    ttl_ok = not e.is_fresh(_time.time(), POLICY_VERSION, MODEL_VERSION)
    checks.append({
        "label":  "CacheEntry.is_fresh() returns False when TTL expired",
        "passed": ttl_ok,
        "note":   "Entry 25 hours old correctly treated as stale",
    })

    # CHECK 3: is_fresh() — policy gate
    e2 = CacheEntry(
        ticket_text_preview="test", consensus_response="test",
        created_at=_time.time(), ttl_seconds=86400,
        policy_version="v_old", model_version=MODEL_VERSION,
    )
    policy_ok = not e2.is_fresh(_time.time(), "v_new", MODEL_VERSION)
    checks.append({
        "label":  "CacheEntry.is_fresh() returns False on policy_version mismatch",
        "passed": policy_ok,
        "note":   "policy_version=v_old vs current=v_new → stale",
    })

    # CHECK 4: get() returns None on empty cache (no API calls needed)
    embedder_mock = type("M", (), {
        "embed": lambda self, t: np.ones(4, dtype=np.float32) / 2.0
    })()
    cache_empty = SemanticCache(embedder_mock, threshold=0.95)
    none_ok = cache_empty.get("anything") is None
    checks.append({
        "label":  "SemanticCache.get() returns None on empty cache",
        "passed": none_ok,
        "note":   "Empty cache correctly returns None",
    })

    # CHECK 5: bump_policy_version causes miss on next get
    cache2 = SemanticCache(embedder_mock, threshold=0.0)  # threshold=0 → always hits on cosine
    cache2.put("test ticket", "some response")
    hit_before = cache2.get("test ticket")
    cache2.bump_policy_version("bumped_v")
    hit_after  = cache2.get("test ticket")
    bump_ok    = hit_before is not None and hit_after is None
    checks.append({
        "label":  "bump_policy_version() causes old entries to miss on next read",
        "passed": bump_ok,
        "note":   f"Before bump: {'hit' if hit_before else 'miss'} · After bump: {'hit' if hit_after else 'miss'}",
    })

    # CHECK 6: L2-norm invariant (uses a fixed vector — no API call)
    v = np.array([3.0, 4.0], dtype=np.float32)
    v = v / np.linalg.norm(v)
    norm_ok = abs(np.linalg.norm(v) - 1.0) < 1e-5
    cos_self = abs(cosine_similarity(v, v) - 1.0) < 1e-5
    checks.append({
        "label":  "L2-normalised vector has norm ≈ 1.0 and cosine(v,v) ≈ 1.0",
        "passed": norm_ok and cos_self,
        "note":   f"||v||={np.linalg.norm(v):.6f}  cosine(v,v)={cosine_similarity(v,v):.6f}",
    })

    duration_ms = round((_time.monotonic() - start) * 1000)
    passed      = sum(1 for c in checks if c["passed"])
    total       = len(checks)
    return {
        "passed":      passed == total,
        "checks":      checks,
        "summary":     f"{passed}/{total} checks passed in {duration_ms}ms",
        "duration_ms": duration_ms,
    }


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Chronicle — Session 14.1 Verification               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    result = run_session_verification()
    print(f"  Verification: {result['summary']}\n")
    for check in result["checks"]:
        icon = "✓" if check["passed"] else "✗"
        print(f"  {icon} {check['label']}")
        print(f"      {check['note']}")
    print()

    if not result["passed"]:
        print("  ✗ Fix failing checks before running the demo.")
        sys.exit(1)

    print("  ✓ Session 14.1 VERIFIED.\n")

    embedder = GeminiEmbedder()

    # FinOps table
    print_finops_table()

    # Threshold sweep (requires Gemini API key)
    if GEMINI_API_KEY:
        run_threshold_sweep(embedder)

        # Hit rate simulation
        sim_cache = SemanticCache(embedder, threshold=DEFAULT_COSINE_THRESHOLD)
        run_hit_rate_simulation(sim_cache)

        # End-to-end demo
        run_end_to_end_demo(embedder)
    else:
        print("  Set GEMINI_API_KEY to run the embedding-dependent sections.")

    print("  ✓ Session 14.1 COMPLETE.")
    print()
    print("  Next: wire the cache into api.py (see FILE 2 below).")
    print()


# ══════════════════════════════════════════════════════════════════
# SESSION 14.2 HANDOFF — LiteLLM + Model Routing
# ══════════════════════════════════════════════════════════════════
#
# What gets ADDED in Session 14.2 (extend, never remove):
#   model_router.py: LiteLLM-based router that selects the cheapest
#     viable model for each ticket tier (simple/complex/critical)
#   The miss path in semantic_cache_dep() calls model_router.route()
#     instead of swarm_invoke_stub() directly
#   cache_usd_saved and routing_model attributes added to OTel span
#
# What stays UNCHANGED from Session 14.1:
#   SemanticCache.get() / put() / evict_expired() / bump_policy_version()
#   CacheEntry schema
#   GeminiEmbedder
#   cosine_similarity()
#   TicketRequest / TicketResponse Pydantic models
#   All threshold calibration logic
#   run_session_verification()
# ══════════════════════════════════════════════════════════════════