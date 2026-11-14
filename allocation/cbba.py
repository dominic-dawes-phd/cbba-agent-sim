"""CBBA, the consensus-based bundle algorithm (Choi, Brunet, How 2009), for C2 and C3 (M3).

Each agent runs CBBA in code every tick (`on_tick`), never in its LLM (hard rule 2):

1. **Bundle building.** The agent adds, best first, the tasks whose bid beats the winning bid it
   knows of, up to `bundle_size`. A bid is `s_ij * discount ** steps_ij`: a suitability `s_ij`
   times a travel discount on the static steps from the agent to the task (hard rule 8). The
   score does not depend on the rest of the bundle, which keeps CBBA's convergence argument
   intact. C2 and C3 differ only in `s_ij`: 1 for any capable agent (C2), or the agent's LLM
   score, asked once per task and cached (C3).
2. **Consensus.** Agents broadcast their winning bids, winners, and timestamps (`cbba` messages)
   over the lossy channel, when they change and every `gossip_every` ticks, and resolve conflicts
   with the paper's update table. An agent that is outbid on a task drops it and every task it
   added after it.

The agent's current task is the first in its bundle. Once it starts that task its bid becomes
`COMMITTED_BID`, so no teammate outbids work already under way. The replicated ledger
(`sim.ledger`) mirrors the bundle as claims with leases and records progress and completion; a
lease that runs out (a crashed owner) reopens the task for bidding.

Conflict-freedom holds only among agents that can reach each other. Across a partition two
agents can win the same task (hard rule 10).
"""

import json
import math
from dataclasses import asdict

from allocation.llm_policy import Assignment
from llm.prompts import PRIMITIVE_TOOLS
from sim.agent import DecisionRequest, TickReport, TickView
from sim.config import CbbaConfig
from sim.ledger import Ledger, decode_rows, encode_rows
from sim.scenario import TaskSpec

NO_WINNER = ""
COMMITTED_BID = 1e9
# Ledger rows listed in the prompt, at most.
LEDGER_PROMPT_ROWS = 20


def beats(bid: float, agent: str, other_bid: float, other_agent: str) -> bool:
    """Whether one bid beats another: higher wins, ties go to the lower agent id."""
    if other_agent == NO_WINNER:
        return bid > 0
    if bid != other_bid:
        return bid > other_bid
    return agent < other_agent


class CbbaState:
    """One agent's CBBA state: bundle, winning bids `y`, winners `z`, and timestamps `s`."""

    def __init__(self, agent_id: str, bundle_size: int) -> None:
        self.agent_id = agent_id
        self.bundle_size = bundle_size
        self.bundle: list[str] = []
        self.bids: dict[str, float] = {}
        self.winners: dict[str, str] = {}
        self.stamps: dict[str, int] = {}

    def build(self, scores: dict[str, float]) -> bool:
        """Bundle building: add the best task this agent outbids on, until the bundle is full.

        Returns:
            Whether the bundle changed.
        """
        changed = False
        while len(self.bundle) < self.bundle_size:
            best: tuple[float, str] | None = None
            for task, score in sorted(scores.items()):
                if task in self.bundle or not self._outbids(task, score):
                    continue
                if best is None or score > best[0]:
                    best = (score, task)
            if best is None:
                break
            score, task = best
            self.bundle.append(task)
            self.bids[task] = score
            self.winners[task] = self.agent_id
            changed = True
        return changed

    def commit(self, task: str) -> None:
        """Marks the agent's current task as under way, so no teammate outbids it."""
        if task in self.bundle:
            self.bids[task] = COMMITTED_BID

    def merge(
        self,
        sender: str,
        bids: dict[str, float],
        winners: dict[str, str],
        stamps: dict[str, int],
        tick: int,
    ) -> bool:
        """Consensus with one teammate's message (Choi et al. 2009, Table 1).

        Returns:
            Whether this agent's view of any winner changed.
        """
        changed = False
        for task in sorted(set(bids) | set(winners)):
            action = self._rule(
                sender, task, bids.get(task, 0.0), winners.get(task, NO_WINNER), stamps
            )
            if action == "update" and (self.bids.get(task), self.winners.get(task)) != (
                bids.get(task, 0.0),
                winners.get(task, NO_WINNER),
            ):
                self.bids[task] = bids.get(task, 0.0)
                self.winners[task] = winners.get(task, NO_WINNER)
                changed = True
            elif action == "reset" and self.winners.get(task, NO_WINNER) != NO_WINNER:
                self.bids[task] = 0.0
                self.winners[task] = NO_WINNER
                changed = True
        self.stamps[sender] = tick
        for agent, stamp in stamps.items():
            if agent != self.agent_id:
                self.stamps[agent] = max(self.stamps.get(agent, -1), stamp)
        return self._drop_outbid() or changed

    def drop(self, task: str) -> None:
        """Forgets a finished task entirely."""
        if task in self.bundle:
            self.bundle.remove(task)
        self.bids.pop(task, None)
        self.winners.pop(task, None)

    def release(self, task: str) -> None:
        """Gives up a task: no winner, and the tasks added after it go too (they built on it)."""
        if task not in self.bundle:
            return
        index = self.bundle.index(task)
        for later in self.bundle[index:]:
            self.bids[later] = 0.0
            self.winners[later] = NO_WINNER
        self.bundle = self.bundle[:index]

    def reset(self, task: str) -> None:
        """Clears a task's winner (its owner's lease ran out)."""
        self.release(task)
        self.bids[task] = 0.0
        self.winners[task] = NO_WINNER

    def payload(self, tick: int) -> dict[str, str]:
        """A `cbba` message: winning bids, winners, and timestamps (own stamp = now)."""
        stamps = {**self.stamps, self.agent_id: tick}
        return {
            "bids": json.dumps(dict(sorted(self.bids.items()))),
            "winners": json.dumps(dict(sorted(self.winners.items()))),
            "stamps": json.dumps(dict(sorted(stamps.items()))),
        }

    def _outbids(self, task: str, score: float) -> bool:
        return score > 0 and beats(
            score, self.agent_id, self.bids.get(task, 0.0), self.winners.get(task, NO_WINNER)
        )

    def _drop_outbid(self) -> bool:
        """Drops the first bundle task this agent no longer wins, and every task after it."""
        for index, task in enumerate(self.bundle):
            if self.winners.get(task) != self.agent_id:
                for later in self.bundle[index + 1 :]:
                    if self.winners.get(later) == self.agent_id:
                        self.bids[later] = 0.0
                        self.winners[later] = NO_WINNER
                self.bundle = self.bundle[:index]
                return True
        return False

    def _rule(self, sender: str, task: str, bid: float, winner: str, stamps: dict[str, int]) -> str:
        """The update table's action for one task: "update", "reset", or "leave"."""
        me = self.agent_id
        mine = self.winners.get(task, NO_WINNER)
        my_bid = self.bids.get(task, 0.0)
        # The sender has more recent news of each agent than this agent does.
        newer = {agent for agent, stamp in stamps.items() if stamp > self.stamps.get(agent, -1)}
        if winner == sender:
            if mine == me:
                return "update" if beats(bid, sender, my_bid, me) else "leave"
            if mine in (sender, NO_WINNER):
                return "update"
            return "update" if mine in newer or beats(bid, sender, my_bid, mine) else "leave"
        if winner == me:
            if mine == sender:
                return "reset"
            if mine in (me, NO_WINNER):
                return "leave"
            return "reset" if mine in newer else "leave"
        if winner == NO_WINNER:
            if mine == sender:
                return "update"
            if mine in (me, NO_WINNER):
                return "leave"
            return "update" if mine in newer else "leave"
        if mine == me:
            return "update" if winner in newer and beats(bid, winner, my_bid, me) else "leave"
        if mine == sender:
            return "update" if winner in newer else "reset"
        if mine in (winner, NO_WINNER):
            return "update" if winner in newer else "leave"
        if winner in newer and mine in newer:
            return "update"
        if winner in newer and beats(bid, winner, my_bid, mine):
            return "update"
        if mine in newer and self.stamps.get(winner, -1) > stamps.get(winner, -1):
            return "reset"
        return "leave"


def decode_cbba(payload: dict[str, str]) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    """The bids, winners, and timestamps a `cbba` message carries."""
    return (
        json.loads(payload["bids"]),
        json.loads(payload["winners"]),
        json.loads(payload["stamps"]),
    )


class CbbaAllocator:
    """CBBA plus the replicated ledger, for one agent; the LLM executes the bundle's first task."""

    tools = (*PRIMITIVE_TOOLS, "mark_done", "give_up")
    rules = (
        "Your team divides tasks by auction. Work only on YOUR TASK, the first in your bundle.",
        "When your task's goal holds, call mark_done; if you cannot finish it, call give_up so it "
        "is auctioned again.",
    )

    def __init__(
        self,
        agent_id: str,
        task_pool: list[TaskSpec],
        config: CbbaConfig,
        posted: bool,
        scored_by_llm: bool,
    ) -> None:
        """Sets up the allocator.

        Args:
            agent_id: The agent this allocator works for.
            task_pool: The episode's task list, shared with the sim (it grows as tasks are found).
            config: Bundle size, discount, gossip period, and leases.
            posted: Every mission task is known from the start (posted mode).
            scored_by_llm: Suitability comes from the agent's LLM (C3) rather than code (C2).
        """
        self._agent_id = agent_id
        self._task_pool = task_pool
        self._config = config
        self._posted = tuple(task.task_id for task in task_pool) if posted else ()
        self._scored_by_llm = scored_by_llm
        self._ledger = Ledger(agent_id, config.lease_ticks)
        self._state = CbbaState(agent_id, config.bundle_size)
        self._scores: dict[str, float] = {}
        self._given_up: dict[str, int] = {}
        self._history_seen = 0
        self._head: str | None = None
        self._dirty = False  # a change made at a decision, to broadcast at the next tick
        self._last_gossip = -config.gossip_every

    def on_tick(self, view: TickView) -> TickReport:
        """Reads messages and own ledger tools, rebuilds the bundle, and gossips changes."""
        tick = view.tick
        for _, result in view.history[self._history_seen :]:
            if result.ok and result.call.name == "mark_done":
                self._ledger.done(result.call.args[0])
            elif result.ok and result.call.name == "give_up":
                task = result.call.args[0]
                self._ledger.release(task)
                self._state.release(task)
                self._given_up[task] = tick + self._config.give_up_ticks
        self._history_seen = len(view.history)
        changed = self._dirty
        self._dirty = False
        merged_ids: set[str] = set()
        for message in view.messages:
            if message.kind == "ledger":
                merged_ids.update(self._ledger.merge(decode_rows(message.payload)))
            elif message.kind == "cbba":
                changed |= self._state.merge(message.sender, *decode_cbba(message.payload), tick)
        for task_id in (*self._posted, *view.known_task_ids):
            self._ledger.add(task_id)
        for task_id, row in sorted(self._ledger.rows.items()):
            if row.status == "done" and (
                task_id in self._state.bids or task_id in self._state.bundle
            ):
                self._state.drop(task_id)
                changed = True
            elif self._ledger.lapsed(task_id, tick) and row.owner != self._agent_id:
                self._state.reset(task_id)
                self._ledger.release(task_id)
                changed = True
        changed |= self._state.build(self._bid_scores(view))
        self._mirror_claims(tick)
        if tick % max(1, self._config.lease_ticks // 3) == 0:
            for task_id in self._state.bundle:
                self._ledger.renew(task_id, tick)
        head = self._state.bundle[0] if self._state.bundle else None
        decision_due = head != self._head or bool(self._unscored_ids(view.known_task_ids))
        self._head = head
        rows = self._ledger.take_changes()
        broadcasts = []
        if rows:
            broadcasts.append(("ledger", encode_rows(rows)))
        if changed or tick - self._last_gossip >= self._config.gossip_every:
            broadcasts.append(("cbba", self._state.payload(tick)))
            if not rows:
                broadcasts.append(("ledger", encode_rows(self._ledger.rows)))
            self._last_gossip = tick
        # The log holds every change to this copy, merges included, for divergence metrics.
        logged = {
            task_id: asdict(self._ledger.rows[task_id])
            for task_id in sorted(merged_ids | set(rows))
        } or None
        return TickReport(tuple(broadcasts), decision_due, logged)

    def assign(self, request: DecisionRequest) -> Assignment:
        """The bundle's first task, committed once work on it starts, with the bundle and ledger."""
        head = self._state.bundle[0] if self._state.bundle else None
        if head is not None and self._ledger.row(head).status == "claimed":
            self._ledger.start(head, request.tick)
            self._state.commit(head)
            self._dirty = True
        tasks = {task.task_id: task for task in self._task_pool}
        lines = [f"YOUR BUNDLE (in order): {', '.join(self._state.bundle) or 'empty'}"]
        lines.append("TEAM LEDGER:")
        for task_id, row in sorted(self._ledger.rows.items())[:LEDGER_PROMPT_ROWS]:
            owner = f" by {row.owner}" if row.owner else ""
            lines.append(f"  {task_id}: {row.status}{owner}")
        return Assignment(tasks.get(head), tuple(lines), ())

    def unscored(self, request: DecisionRequest) -> list[TaskSpec]:
        """Known open tasks the LLM has not scored yet (C3 only)."""
        tasks = {task.task_id: task for task in self._task_pool}
        return [tasks[task_id] for task_id in self._unscored_ids(request.known_task_ids)]

    def wants_observation(self, tick: int) -> bool:
        """Its coordination never needs the perception text."""
        return False

    def set_scores(self, scores: dict[str, float]) -> None:
        """Stores the LLM's suitability scores, used for every later bid."""
        self._scores.update(scores)

    def _unscored_ids(self, known_task_ids: tuple[str, ...]) -> list[str]:
        if not self._scored_by_llm:
            return []
        known = {*self._posted, *known_task_ids}
        tasks = {task.task_id for task in self._task_pool}
        return sorted(
            task_id
            for task_id in known & tasks
            if task_id not in self._scores and self._ledger.row(task_id).status != "done"
        )

    def _bid_scores(self, view: TickView) -> dict[str, float]:
        """This agent's bid on every task it may take now."""
        done = {task_id for task_id, row in self._ledger.rows.items() if row.status == "done"}
        scores = {}
        for task in self._task_pool:
            task_id = task.task_id
            if (
                task_id not in self._ledger.rows
                or task_id in done
                or self._given_up.get(task_id, -1) >= view.tick
                or not set(task.after) <= done
            ):
                continue
            suitability = self._suitability(task, view)
            steps = task_steps(task, view)
            if suitability > 0 and steps is not None:
                scores[task_id] = suitability * self._config.discount**steps
        return scores

    def _suitability(self, task: TaskSpec, view: TickView) -> float:
        """s_ij: the LLM's cached score (C3), or 1 for a capable agent (C2)."""
        if self._scored_by_llm:
            return self._scores.get(task.task_id, 0.0)
        scans = any(step.name == "scan" for step in task.steps)
        return 0.0 if scans and not view.card.can_scan else 1.0

    def _mirror_claims(self, tick: int) -> None:
        """Claims in the ledger every task in the bundle, and releases ones no longer in it."""
        for task_id in self._state.bundle:
            if self._ledger.row(task_id).owner != self._agent_id:
                self._ledger.claim(task_id, self._state.bids[task_id], tick)
        for task_id, row in sorted(self._ledger.rows.items()):
            owned = row.owner == self._agent_id and row.status in ("claimed", "in_progress")
            if owned and task_id not in self._state.bundle:
                self._ledger.release(task_id)


def task_steps(task: TaskSpec, view: TickView) -> int | None:
    """Static steps from the agent to where a task happens (hard rule 8), or None if unknown.

    The place is the first goal object the agent believes in, else the first step target it
    believes in (furniture is known from the map).
    """
    candidates = [goal.obj_id for goal in task.goals] + [
        step.args[0].split("[")[0] for step in task.steps if step.args
    ]
    for obj_id in candidates:
        belief = view.beliefs.get(obj_id)
        if belief is not None and not belief.missing:
            steps = view.planner.steps_to_any(view.cell, belief.cells())
            return None if steps is None or math.isinf(steps) else steps
    return None
