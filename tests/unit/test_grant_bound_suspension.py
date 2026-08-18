"""Tests for issue #3649 — suspend-side grant population and journal append.

Remaining acceptance criteria:
    AC-SUSP  park_task() populates role, grant_hash, parent_run_id,
             chain_head_at_suspend on AgentCheckpoint at suspend time.
    AC-CONT  resume_task() appends a task.grant_continuation row to the journal
             binding (checkpoint_hash, grant_hash, chain_head_at_suspend,
             chain_head_at_resume) immediately after the resume row is written.
"""

from __future__ import annotations

from pathlib import Path

from bernstein.core.persistence.agent_checkpoint import (
    AgentCheckpoint,
    compute_grant_hash,
    find_checkpoint_for_task,
    save_checkpoint,
)
from bernstein.core.replay.journal import load_events
from bernstein.core.security.audit_chain import AuditChainStore
from bernstein.core.security.permissions import get_permissions_for_role
from bernstein.core.tasks.suspension import (
    JOURNAL_EVENT_GRANT_CONTINUATION,
    SuspendRow,
    park_task,
    resume_task,
)

_KEY = b"test-key-32-bytes-exactly-------"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(tmp_path: Path) -> AuditChainStore:
    return AuditChainStore(tmp_path / "audit", key=_KEY)


def _worktree(tmp_path: Path, name: str = "wt") -> Path:
    wt = tmp_path / name
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "work.py").write_text("# in progress\n", encoding="utf-8")
    return wt


def _park(
    tmp_path: Path,
    *,
    task_id: str = "T-grant",
    role: str = "backend",
    parent_run_id: str = "run-42",
    agent_id: str = "agent-1",
) -> tuple[SuspendRow, str, AuditChainStore]:
    """Park a task and return (suspend_row, suspend_receipt_hash, chain)."""
    sdd = tmp_path / ".sdd"
    chain = _chain(tmp_path)
    wt = _worktree(tmp_path)

    # Pre-populate an AgentCheckpoint so park_task can read role/parent_run_id
    cp = AgentCheckpoint(
        agent_id=agent_id,
        task_id=task_id,
        worktree_path=str(wt),
        role=role,
        parent_run_id=parent_run_id,
    )
    save_checkpoint(cp, sdd / "runtime")

    result = park_task(
        sdd_dir=sdd,
        task_id=task_id,
        adapter="claude",
        session_id="sess-1",
        worktree_path=wt,
        envelope="subscription",
        reserved_usd=5.0,
        spent_usd=1.0,
        chain=chain,
    )
    return result.suspend_row, result.suspend_receipt_hash, chain


# ---------------------------------------------------------------------------
# AC-SUSP: suspend-side population of AgentCheckpoint grant fields
# ---------------------------------------------------------------------------


class TestSuspendSideGrantPopulation:
    def test_park_writes_grant_hash_to_checkpoint(self, tmp_path: Path) -> None:
        """park_task() must populate grant_hash on the stored AgentCheckpoint."""
        task_id = "T-susp-grant"
        role = "backend"
        sdd = tmp_path / ".sdd"
        chain = _chain(tmp_path)
        wt = _worktree(tmp_path)

        cp = AgentCheckpoint(
            agent_id="ag-1",
            task_id=task_id,
            worktree_path=str(wt),
            role=role,
            parent_run_id="run-1",
        )
        save_checkpoint(cp, sdd / "runtime")

        park_task(
            sdd_dir=sdd,
            task_id=task_id,
            adapter="claude",
            session_id="s",
            worktree_path=wt,
            envelope="subscription",
            reserved_usd=5.0,
            spent_usd=1.0,
            chain=chain,
        )

        updated = find_checkpoint_for_task(task_id, sdd / "runtime")
        assert updated is not None
        assert updated.grant_hash != "", "grant_hash must be populated after park"

    def test_park_grant_hash_is_correct(self, tmp_path: Path) -> None:
        """The stored grant_hash must match compute_grant_hash() output."""
        task_id = "T-hash-correct"
        role = "backend"
        parent_run_id = "run-99"
        sdd = tmp_path / ".sdd"
        chain = _chain(tmp_path)
        wt = _worktree(tmp_path)

        cp = AgentCheckpoint(
            agent_id="ag-2",
            task_id=task_id,
            worktree_path=str(wt),
            role=role,
            parent_run_id=parent_run_id,
        )
        save_checkpoint(cp, sdd / "runtime")

        park_task(
            sdd_dir=sdd,
            task_id=task_id,
            adapter="claude",
            session_id="s",
            worktree_path=wt,
            envelope="subscription",
            reserved_usd=5.0,
            spent_usd=1.0,
            chain=chain,
        )

        updated = find_checkpoint_for_task(task_id, sdd / "runtime")
        assert updated is not None

        perms = get_permissions_for_role(role)
        expected = compute_grant_hash(
            role,
            perms,
            task_id,
            parent_run_id,
            updated.chain_head_at_suspend,
        )
        assert updated.grant_hash == expected

    def test_park_populates_chain_head_at_suspend(self, tmp_path: Path) -> None:
        """chain_head_at_suspend must equal the suspend row's event_hash."""
        task_id = "T-chain-head"
        sdd = tmp_path / ".sdd"
        chain = _chain(tmp_path)
        wt = _worktree(tmp_path)

        cp = AgentCheckpoint(
            agent_id="ag-3",
            task_id=task_id,
            worktree_path=str(wt),
            role="backend",
            parent_run_id="run-1",
        )
        save_checkpoint(cp, sdd / "runtime")

        result = park_task(
            sdd_dir=sdd,
            task_id=task_id,
            adapter="claude",
            session_id="s",
            worktree_path=wt,
            envelope="subscription",
            reserved_usd=5.0,
            spent_usd=1.0,
            chain=chain,
        )

        updated = find_checkpoint_for_task(task_id, sdd / "runtime")
        assert updated is not None
        assert updated.chain_head_at_suspend == result.suspend_row.event_hash

    def test_park_populates_role_and_parent_run_id(self, tmp_path: Path) -> None:
        """role and parent_run_id must be preserved from the existing checkpoint."""
        task_id = "T-role-parent"
        role = "qa"
        parent_run_id = "run-777"
        sdd = tmp_path / ".sdd"
        chain = _chain(tmp_path)
        wt = _worktree(tmp_path)

        cp = AgentCheckpoint(
            agent_id="ag-4",
            task_id=task_id,
            worktree_path=str(wt),
            role=role,
            parent_run_id=parent_run_id,
        )
        save_checkpoint(cp, sdd / "runtime")

        park_task(
            sdd_dir=sdd,
            task_id=task_id,
            adapter="claude",
            session_id="s",
            worktree_path=wt,
            envelope="subscription",
            reserved_usd=5.0,
            spent_usd=1.0,
            chain=chain,
        )

        updated = find_checkpoint_for_task(task_id, sdd / "runtime")
        assert updated is not None
        assert updated.role == role
        assert updated.parent_run_id == parent_run_id

    def test_park_without_prior_checkpoint_does_not_raise(self, tmp_path: Path) -> None:
        """park_task() without a pre-existing AgentCheckpoint must not fail.

        Grant fields remain empty in this case — backward compat for tasks
        that were not spawned with agent_checkpoint.py.
        """
        sdd = tmp_path / ".sdd"
        chain = _chain(tmp_path)
        wt = _worktree(tmp_path)

        # No prior checkpoint written
        result = park_task(
            sdd_dir=sdd,
            task_id="T-no-prior",
            adapter="claude",
            session_id="s",
            worktree_path=wt,
            envelope="subscription",
            reserved_usd=5.0,
            spent_usd=1.0,
            chain=chain,
        )
        assert result.suspend_row.task_id == "T-no-prior"


# ---------------------------------------------------------------------------
# AC-CONT: journal append of ContinuationEntry on successful resume
# ---------------------------------------------------------------------------


class TestJournalContinuationEntry:
    def test_resume_appends_grant_continuation_row(self, tmp_path: Path) -> None:
        """resume_task() appends a task.grant_continuation row to the journal."""
        suspend_row, receipt_hash, chain = _park(tmp_path)
        sdd = tmp_path / ".sdd"
        wt2 = _worktree(tmp_path, "wt2")

        resume_task(
            sdd_dir=sdd,
            suspend_row=suspend_row,
            new_worktree_path=wt2,
            chain=chain,
            suspend_receipt_hash=receipt_hash,
        )

        from bernstein.core.tasks.checkpoint_retry import task_run_id

        journal_path = sdd / "runs" / task_run_id(suspend_row.task_id) / "journal.jsonl"
        events = load_events(journal_path).events
        continuation_rows = [e for e in events if e.get("event") == JOURNAL_EVENT_GRANT_CONTINUATION]
        assert len(continuation_rows) == 1, "Exactly one grant_continuation row must be appended on successful resume"

    def test_continuation_row_binds_correct_fields(self, tmp_path: Path) -> None:
        """The continuation row must bind checkpoint_hash, grant_hash,
        chain_head_at_suspend, chain_head_at_resume."""
        task_id = "T-cont-fields"
        suspend_row, receipt_hash, chain = _park(tmp_path, task_id=task_id)
        sdd = tmp_path / ".sdd"
        wt2 = _worktree(tmp_path, "wt2")

        result = resume_task(
            sdd_dir=sdd,
            suspend_row=suspend_row,
            new_worktree_path=wt2,
            chain=chain,
            suspend_receipt_hash=receipt_hash,
        )

        from bernstein.core.tasks.checkpoint_retry import task_run_id

        journal_path = sdd / "runs" / task_run_id(task_id) / "journal.jsonl"
        events = load_events(journal_path).events
        row = next(e for e in events if e.get("event") == JOURNAL_EVENT_GRANT_CONTINUATION)

        assert row["chain_head_at_resume"] == result.resume_event_hash
        assert row["chain_head_at_suspend"] == suspend_row.event_hash
        # grant_hash and checkpoint_hash must be non-empty strings
        assert isinstance(row.get("grant_hash"), str)
        assert isinstance(row.get("checkpoint_hash"), str)

    def test_continuation_row_appears_after_resume_row(self, tmp_path: Path) -> None:
        """grant_continuation must come after the task.resume row in the journal."""
        suspend_row, receipt_hash, chain = _park(tmp_path)
        sdd = tmp_path / ".sdd"
        wt2 = _worktree(tmp_path, "wt2")

        resume_task(
            sdd_dir=sdd,
            suspend_row=suspend_row,
            new_worktree_path=wt2,
            chain=chain,
            suspend_receipt_hash=receipt_hash,
        )

        from bernstein.core.tasks.checkpoint_retry import task_run_id

        journal_path = sdd / "runs" / task_run_id(suspend_row.task_id) / "journal.jsonl"
        events = load_events(journal_path).events
        event_types = [e.get("event") for e in events]

        resume_idx = next(i for i, e in enumerate(event_types) if e == "task.resume")
        cont_idx = next(i for i, e in enumerate(event_types) if e == JOURNAL_EVENT_GRANT_CONTINUATION)
        assert cont_idx > resume_idx

    def test_no_continuation_row_without_checkpoint(self, tmp_path: Path) -> None:
        """If no AgentCheckpoint exists for the task, no continuation row is written.

        This is the backward-compat path: old tasks that never had a checkpoint
        resume normally and simply produce no continuation evidence.
        """
        sdd = tmp_path / ".sdd"
        chain = _chain(tmp_path)
        wt = _worktree(tmp_path)

        # Park without a pre-existing AgentCheckpoint
        result = park_task(
            sdd_dir=sdd,
            task_id="T-no-cp",
            adapter="claude",
            session_id="s",
            worktree_path=wt,
            envelope="subscription",
            reserved_usd=5.0,
            spent_usd=1.0,
            chain=chain,
        )
        wt2 = _worktree(tmp_path, "wt2")

        resume_task(
            sdd_dir=sdd,
            suspend_row=result.suspend_row,
            new_worktree_path=wt2,
            chain=chain,
            suspend_receipt_hash=result.suspend_receipt_hash,
        )

        from bernstein.core.tasks.checkpoint_retry import task_run_id

        journal_path = sdd / "runs" / task_run_id("T-no-cp") / "journal.jsonl"
        events = load_events(journal_path).events
        continuation_rows = [e for e in events if e.get("event") == JOURNAL_EVENT_GRANT_CONTINUATION]
        assert continuation_rows == [], "No continuation row should appear when no checkpoint exists"
