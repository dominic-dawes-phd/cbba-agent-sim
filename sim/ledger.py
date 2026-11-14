"""The replicated task ledger: each agent's copy of who owns what and what is done (M2).

Each agent holds its own `Ledger` and changes only its own copy. Copies converge by gossip: an agent
broadcasts the rows it changed, and every so often its whole ledger (anti-entropy), over the same
lossy channel as everything else. Merging is code, never an LLM.

**Merge rule.** Of two copies of a row, the one that wins is: done over not done; then the higher
version; then the higher status (unclaimed < claimed < in_progress); then the higher bid; then the
lower owner id; then the later lease. That is a total order, and a merge keeps the larger row, so
merging is commutative, associative, and idempotent: copies agree once they have seen the same rows,
in whatever order the messages arrived. Done is absorbing. Every local change raises the version,
so a release (back to unclaimed) or a re-claim after a lease ran out beats the row it replaces.

**Leases.** An owner renews its lease while it works. A lease that runs out makes the task
claimable again, which is how a crashed owner's task gets picked up; a healthy owner cut off by a
partition can lose its task the same way (a lease false positive, counted by the analyzer).

The ledger records shared state only. Agreement among agents that can reach each other comes from
the allocator (CBBA); nothing here is a guarantee across partitions (hard rule 10).
"""

import json
from dataclasses import asdict, dataclass

STATUSES = ("unclaimed", "claimed", "in_progress", "done")


@dataclass(frozen=True)
class LedgerRow:
    """One task's entry.

    Attributes:
        status: One of `STATUSES`.
        owner: The agent that claimed it, or None.
        bid: The owner's bid when it claimed (an allocator's score; 0 without one).
        version: Raised by every local change, so later decisions win merges.
        lease_expiry: The tick after which the claim lapses unless renewed; None when unowned.
    """

    status: str
    owner: str | None
    bid: float
    version: int
    lease_expiry: int | None


UNCLAIMED = LedgerRow("unclaimed", None, 0.0, 0, None)


def wins(row: LedgerRow, other: LedgerRow) -> bool:
    """Whether `row` beats `other` in a merge (see the module docstring); False when equal."""
    if (row.status == "done") != (other.status == "done"):
        return row.status == "done"
    if row.version != other.version:
        return row.version > other.version
    rank, other_rank = STATUSES.index(row.status), STATUSES.index(other.status)
    if rank != other_rank:
        return rank > other_rank
    if row.bid != other.bid:
        return row.bid > other.bid
    if row.owner != other.owner:
        return other.owner is None or (row.owner is not None and row.owner < other.owner)
    return (row.lease_expiry or -1) > (other.lease_expiry or -1)


def merged(row: LedgerRow, other: LedgerRow) -> LedgerRow:
    """The row both copies agree on after a merge."""
    return row if wins(row, other) or row == other else other


class Ledger:
    """One agent's copy of the ledger."""

    def __init__(self, agent_id: str, lease_ticks: int) -> None:
        """Sets up an empty copy.

        Args:
            agent_id: The agent holding this copy.
            lease_ticks: How long a claim or renewal lasts.
        """
        self.agent_id = agent_id
        self.rows: dict[str, LedgerRow] = {}
        self._lease_ticks = lease_ticks
        self._changed: set[str] = set()

    def row(self, task_id: str) -> LedgerRow:
        """A task's row (unclaimed if never seen)."""
        return self.rows.get(task_id, UNCLAIMED)

    def add(self, task_id: str) -> None:
        """Records a task this agent learned of, unclaimed, so gossip spreads it."""
        if task_id not in self.rows:
            self._set(task_id, UNCLAIMED)

    def claim(self, task_id: str, bid: float, tick: int) -> None:
        """Claims a task for this agent."""
        row = self.row(task_id)
        self._set(
            task_id,
            LedgerRow("claimed", self.agent_id, bid, row.version + 1, tick + self._lease_ticks),
        )

    def start(self, task_id: str, tick: int) -> None:
        """Marks this agent's task in progress (and renews its lease)."""
        row = self.row(task_id)
        lease = tick + self._lease_ticks
        self._set(task_id, LedgerRow("in_progress", self.agent_id, row.bid, row.version + 1, lease))

    def renew(self, task_id: str, tick: int) -> None:
        """Extends this agent's lease on a task it owns; the version stays, the later lease wins."""
        row = self.row(task_id)
        if row.owner == self.agent_id and row.status in ("claimed", "in_progress"):
            self._set(
                task_id,
                LedgerRow(row.status, row.owner, row.bid, row.version, tick + self._lease_ticks),
            )

    def done(self, task_id: str) -> None:
        """Marks a task done. Done is absorbing."""
        row = self.row(task_id)
        self._set(
            task_id, LedgerRow("done", row.owner or self.agent_id, row.bid, row.version + 1, None)
        )

    def release(self, task_id: str) -> None:
        """Gives a task back: unclaimed, with a higher version so the release wins."""
        row = self.row(task_id)
        if row.status != "done":
            self._set(task_id, LedgerRow("unclaimed", None, 0.0, row.version + 1, None))

    def lapsed(self, task_id: str, tick: int) -> bool:
        """Whether a task's claim has run out, so it may be claimed again."""
        row = self.row(task_id)
        return (
            row.status in ("claimed", "in_progress")
            and row.lease_expiry is not None
            and row.lease_expiry < tick
        )

    def merge(self, incoming: dict[str, LedgerRow]) -> list[str]:
        """Merges rows from a teammate.

        Returns:
            The tasks whose row changed here, in id order.
        """
        changed = []
        for task_id in sorted(incoming):
            current = self.rows.get(task_id)
            result = incoming[task_id] if current is None else merged(current, incoming[task_id])
            if result != current:
                self.rows[task_id] = result
                changed.append(task_id)
        return changed

    def take_changes(self) -> dict[str, LedgerRow]:
        """The rows this agent changed since the last call (for broadcast-on-change)."""
        changes = {task_id: self.rows[task_id] for task_id in sorted(self._changed)}
        self._changed.clear()
        return changes

    def _set(self, task_id: str, row: LedgerRow) -> None:
        self.rows[task_id] = row
        self._changed.add(task_id)


def encode_rows(rows: dict[str, LedgerRow]) -> dict[str, str]:
    """A ledger message payload (the channel carries string fields)."""
    return {"rows": json.dumps({task_id: asdict(row) for task_id, row in sorted(rows.items())})}


def decode_rows(payload: dict[str, str]) -> dict[str, LedgerRow]:
    """The rows a ledger message carries."""
    return {task_id: LedgerRow(**fields) for task_id, fields in json.loads(payload["rows"]).items()}
