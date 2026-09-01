"""Retrieval telemetry contracts for the holographic memory store.

``retrieval_count`` exists to tell the store which facts are earning their
place, and it is the only usage signal ``trust_score`` and
``temporal_decay_half_life`` could ever learn from.

The counter was written by exactly one method, ``MemoryStore.search_facts``,
which no caller in the tree ever invoked. Every real read — the per-turn
``<memory-context>`` prefetch and the explicit ``fact_store action=search``
tool — goes through ``FactRetriever``, which never recorded anything. The
observable result on a live store was 313 of 317 facts sitting at
``retrieval_count = 0`` while 36 of them carried ``helpful_count > 0``: the
store believed those facts had been rated helpful without ever having been
retrieved.

These are behaviour contracts, not snapshots. They assert how usage and the
counter must relate, so they stay meaningful as the retrieval pipeline is
re-tuned.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """A real on-disk store.

    NOT ``":memory:"`` — MemoryStore shares one process-wide connection per
    resolved path, so ``":memory:"`` becomes a literal ``./:memory:`` file
    that leaks state across runs.
    """
    s = MemoryStore(str(tmp_path / "telemetry.db"))
    yield s
    s.close()


@pytest.fixture
def seeded(store):
    """Distinct facts with no shared content tokens, so a query can select
    a known subset and leave the rest unmatched."""
    ids = {
        "rollback": store.add_fact(
            content="The Thursday deployment rollback failed on stale migration state.",
            category="project",
        ),
        "compaction": store.add_fact(
            content="Compaction threshold tuned to 0.85 for long sessions.",
            category="tool",
        ),
        "kayak": store.add_fact(
            content="Kayak paddle feathering angle preference is sixty degrees.",
            category="user_pref",
        ),
    }
    return store, ids


def _counts(store) -> dict[int, int]:
    return {
        row["fact_id"]: row["retrieval_count"]
        for row in store._conn.execute(
            "SELECT fact_id, retrieval_count FROM facts"
        ).fetchall()
    }


# ---------------------------------------------------------------------------
# The core contract: the path the agent actually reads through must record.
# ---------------------------------------------------------------------------

def test_retriever_search_records_usage_for_returned_facts(seeded):
    """``FactRetriever.search`` is the real read path and must count.

    This is the regression that mattered: the counter was only written by
    ``MemoryStore.search_facts``, which nothing called.
    """
    store, ids = seeded
    before = _counts(store)

    results = FactRetriever(store=store).search("deployment rollback")

    assert results, "fixture query must match at least one fact"
    after = _counts(store)
    for fact in results:
        fid = fact["fact_id"]
        assert after[fid] == before[fid] + 1, (
            f"fact {fid} was returned to the caller but its retrieval_count "
            f"did not move ({before[fid]} -> {after[fid]})"
        )


def test_retriever_search_does_not_record_unreturned_facts(seeded):
    """Telemetry must be attributable: only what was returned is counted.

    A counter that increments on every fact in the table is as useless as one
    that never increments — neither ranks anything.
    """
    store, ids = seeded
    before = _counts(store)

    results = FactRetriever(store=store).search("deployment rollback")
    returned = {f["fact_id"] for f in results}

    assert ids["kayak"] not in returned, "fixture no longer isolates a non-match"
    after = _counts(store)
    for fid in set(before) - returned:
        assert after[fid] == before[fid], (
            f"fact {fid} was never returned but its retrieval_count moved"
        )


def test_empty_result_records_nothing(store):
    """A query that matches nothing must leave every counter alone."""
    store.add_fact(content="Solitary unrelated fact about kayaks.", category="general")
    before = _counts(store)

    assert FactRetriever(store=store).search("zzzznonexistenttoken") == []

    assert _counts(store) == before


def test_repeated_retrieval_accumulates(seeded):
    """Counts must accumulate across calls — that is the whole signal."""
    store, _ = seeded
    retriever = FactRetriever(store=store)

    first = retriever.search("deployment rollback")
    fid = first[0]["fact_id"]
    start = _counts(store)[fid]

    for _ in range(3):
        retriever.search("deployment rollback")

    assert _counts(store)[fid] == start + 3


# ---------------------------------------------------------------------------
# The store-level invariant that made the defect visible on a live corpus.
# ---------------------------------------------------------------------------

def test_helpful_feedback_implies_prior_retrieval(seeded):
    """A fact rated helpful must have been retrieved at least once.

    On the live store this invariant was violated by 36 facts. It is the
    cheapest reproduction of the defect: no timing, no pipeline knowledge,
    just two columns that cannot both be true.
    """
    store, _ = seeded
    results = FactRetriever(store=store).search("deployment rollback")
    store.record_feedback(results[0]["fact_id"], helpful=True)

    incoherent = store._conn.execute(
        "SELECT fact_id FROM facts WHERE helpful_count > 0 AND retrieval_count = 0"
    ).fetchall()
    assert not incoherent, (
        "facts rated helpful without ever being retrieved: "
        f"{[r['fact_id'] for r in incoherent]}"
    )


def test_returned_rows_agree_with_database(seeded):
    """The ``retrieval_count`` handed back must match what was persisted.

    Rows are SELECTed before the counter is incremented, so returning them
    untouched would report a count the database already disagrees with —
    stale by exactly the retrieval the caller just performed.
    """
    store, _ = seeded

    results = FactRetriever(store=store).search("deployment rollback")
    assert results

    persisted = _counts(store)
    for fact in results:
        assert fact["retrieval_count"] == persisted[fact["fact_id"]], (
            f"fact {fact['fact_id']} returned retrieval_count="
            f"{fact['retrieval_count']} but the database holds "
            f"{persisted[fact['fact_id']]}"
        )


# ---------------------------------------------------------------------------
# Every read path, not just the one that was easiest to test.
#
# The first cut of this fix wrapped search/probe/related/reason and still left
# probe's category-bank branch unrecorded — a public path that delegates its
# result set to a private scorer. The suite was green because every test drove
# ``search``. Parametrising over the real read surface is what catches an
# unplugged branch, so these must stay exhaustive as methods are added.
# ---------------------------------------------------------------------------

RETRIEVAL_CALLS = {
    "search": lambda r: r.search("deployment rollback"),
    "probe": lambda r: r.probe("rollback"),
    # probe's OTHER branch: an existing category bank routes through
    # _score_facts_by_vector. Banks are rebuilt on every add_fact, so on a
    # live store this is probe's common case, not an edge.
    "probe_category_bank": lambda r: r.probe("rollback", category="project"),
    "related": lambda r: r.related("rollback"),
    "reason": lambda r: r.reason(["rollback", "migration"]),
}


@pytest.mark.parametrize("name", sorted(RETRIEVAL_CALLS))
def test_every_read_path_records_usage(seeded, name):
    """Any method that hands facts to the agent must count them exactly once."""
    store, _ = seeded
    retriever = FactRetriever(store=store)
    before = _counts(store)

    results = RETRIEVAL_CALLS[name](retriever)

    assert results, f"{name} returned nothing — fixture no longer exercises it"
    after = _counts(store)
    for fact in results:
        fid = fact["fact_id"]
        assert after[fid] == before[fid] + 1, (
            f"{name} returned fact {fid} but its retrieval_count went "
            f"{before[fid]} -> {after[fid]} (expected exactly +1; "
            f"0 means the path is unrecorded, +2 means double-counted)"
        )


def test_numpy_absent_fallback_counts_once(seeded, monkeypatch):
    """Delegating fallbacks must not double-count.

    Without numpy, probe/related/reason fall back to ``search``, which has
    already recorded. A well-meaning wrapper around the delegated call would
    count those facts twice — and would overstate exactly the retrievals that
    were least direct.
    """
    from plugins.memory.holographic import holographic as hrr

    store, _ = seeded
    monkeypatch.setattr(hrr, "_HAS_NUMPY", False)
    retriever = FactRetriever(store=store)
    before = _counts(store)

    results = retriever.probe("rollback")

    assert results, "fallback returned nothing — test proves nothing"
    after = _counts(store)
    for fact in results:
        fid = fact["fact_id"]
        assert after[fid] == before[fid] + 1, (
            f"fact {fid} counted {after[fid] - before[fid]}x through the "
            "numpy-absent fallback (expected exactly 1)"
        )


# ---------------------------------------------------------------------------
# Wiring: unit tests on a pure function cannot catch an unplugged feature.
# This drives the real provider entry point that builds <memory-context>.
# ---------------------------------------------------------------------------

def test_provider_prefetch_records_usage(tmp_path):
    """The per-turn injection path must record, not just the search tool.

    ``HolographicMemoryProvider.prefetch`` is what fills the
    ``<memory-context>`` block on every turn. It was the single largest
    source of unrecorded reads.
    """
    from plugins.memory.holographic import HolographicMemoryProvider

    provider = HolographicMemoryProvider()
    provider._config = {"db_path": str(tmp_path / "prefetch.db")}
    provider.initialize(session_id="test-session")
    try:
        provider._store.add_fact(
            content="The Thursday deployment rollback failed on stale migration state.",
            category="project",
        )

        context = provider.prefetch("deployment rollback")

        assert context, "prefetch returned no context for a matching query"
        counts = _counts(provider._store)
        recorded = {fid: c for fid, c in counts.items() if c}
        assert recorded, (
            "prefetch injected facts into the turn but recorded no retrievals"
        )
        # Pin the exact count, not merely non-zero: `any(c > 0)` would pass
        # just as happily if the injection path counted every fact twice.
        assert all(c == 1 for c in recorded.values()), (
            f"prefetch recorded {recorded} — expected exactly 1 per fact"
        )
    finally:
        provider._store.close()


# ---------------------------------------------------------------------------
# Telemetry is bookkeeping: it must never cost the caller its results.
# ---------------------------------------------------------------------------

def test_recording_failure_does_not_break_retrieval(seeded, monkeypatch):
    """If the counter write fails, the caller still gets its facts.

    Memory recall is load-bearing for the turn; usage statistics are not.
    A telemetry fault must degrade to missing statistics, never to a lost
    ``<memory-context>`` block.
    """
    store, _ = seeded

    def boom(*args, **kwargs):
        raise RuntimeError("simulated telemetry failure")

    monkeypatch.setattr(store, "record_retrievals", boom)

    results = FactRetriever(store=store).search("deployment rollback")

    assert results, "a telemetry failure must not swallow retrieval results"
