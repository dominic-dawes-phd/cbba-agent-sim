"""C1: RoCo-style dialogue rounds, where the team divides work only by talking (conditions.md).

One `DialogueRounds` object holds the team's rounds; each agent's `DialogueAllocator` drives its own
part every tick (`on_tick`), and its executor does the work the latest plan gave it.

- **Trigger.** An agent with no task (at the start, after a WAIT, or after it calls
  `talk_to_team`) asks for a round. The round opens unless someone in its radio component is
  already in one, in which case it waits for that round's plan.
- **Members** are the online agents connected to the trigger at that moment (within `comm_range`
  of each other, transitively). Partitioned and crashed agents are not members, so `k`
  partitions talk in `k` separate rounds.
- **Turns** go in order, the trigger first, then the others by id. A turn is one LLM call (latency
  `L`, hard rule 3), and its message goes out `L` ticks later to every member separately over the
  lossy channel, so each member's transcript holds only what reached it. The next member speaks
  after the one-tick hop.
- **Ending.** A turn ends with PROCEED, or EXECUTE and a joint plan (a task, WAIT, or CONTINUE per
  member), allowed only once every member has spoken. Code checks a plan (members, known tasks)
  before it is sent; a failed check goes back to the members as feedback and the round goes on, up
  to `replans` failures. A round also ends after `turns_per_member` x members turns.
- **Adoption.** A member adopts the plan only if the EXECUTE message reached it. Members in the
  middle of a skill keep executing while the team talks, and the trigger holds still until a plan
  arrives.

Nothing here is structured shared state: there are no claims, reports, ledger rows, or
`task_found` messages (they carry task ids). A plan is advice each receiver takes or misses.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from allocation.llm_policy import Assignment
from llm.client import ChatRequest, LLMClient
from llm.latency import latency_ticks
from llm.prompts import PRIMITIVE_TOOLS, PROMPTS_DIR, capability_line, history_lines
from sim.agent import DecisionRequest, LLMCallRecord, TickReport, TickView
from sim.config import DialogueConfig, LatencyConfig, LLMConfig
from sim.logger import JsonValue
from sim.scenario import TaskSpec
from sim.world import Cell

DIALOGUE_SYSTEM_PATH = PROMPTS_DIR / "dialogue_system.txt"
WAIT = "WAIT"
CONTINUE = "CONTINUE"
PROCEED = "PROCEED"
EXECUTE = "EXECUTE"


@dataclass
class Round:
    """One dialogue round among the agents of one radio component.

    Attributes:
        speaker_index: Whose turn it is, as an index into `members`.
        next_turn_tick: When the current speaker may speak (after the last message's hop).
        pending: The current speaker's message, waiting for its deliberation to end: (send tick,
            payload).
    """

    round_id: int
    members: list[str]
    next_turn_tick: int
    turns: int = 0
    replans: int = 0
    speaker_index: int = 0
    spoken: set[str] = field(default_factory=set)
    pending: tuple[int, dict[str, str]] | None = None

    @property
    def speaker(self) -> str:
        """The member whose turn it is."""
        return self.members[self.speaker_index]


class DialogueRounds:
    """The team's dialogue rounds, shared by every agent's allocator in a C1 episode."""

    def __init__(self, config: DialogueConfig, comm_range: float | None) -> None:
        self._config = config
        self._comm_range = comm_range
        self._rounds: dict[int, Round] = {}
        self._member_of: dict[str, int] = {}
        self._next_id = 0
        self.feedback: dict[str, list[str]] = {}  # per agent: plan feedback from the current round

    def round_of(self, agent_id: str) -> Round | None:
        """The open round an agent is a member of, if any."""
        round_id = self._member_of.get(agent_id)
        return self._rounds.get(round_id) if round_id is not None else None

    def try_open(
        self, trigger: str, tick: int, cells: dict[str, Cell], online: dict[str, bool]
    ) -> Round | None:
        """Opens a round for a trigger's radio component, unless a member is already in one."""
        members = self._component(trigger, cells, online)
        if any(member in self._member_of for member in members):
            return None
        order = [trigger, *sorted(member for member in members if member != trigger)]
        opened = Round(self._next_id, order, tick)
        self._rounds[opened.round_id] = opened
        for member in order:
            self._member_of[member] = opened.round_id
            self.feedback[member] = []
        self._next_id += 1
        return opened

    def max_turns(self, opened: Round) -> int:
        """M: the round's turn limit."""
        return self._config.turns_per_member * len(opened.members)

    def close(self, closed: Round) -> None:
        """Ends a round; its members are free to join another."""
        for member in closed.members:
            if self._member_of.get(member) == closed.round_id:
                del self._member_of[member]
        del self._rounds[closed.round_id]

    def _component(
        self, trigger: str, cells: dict[str, Cell], online: dict[str, bool]
    ) -> list[str]:
        """The online agents connected to the trigger by radio range, transitively."""
        reached = {trigger}
        frontier = [trigger]
        while frontier:
            agent = frontier.pop()
            for other in sorted(cells):
                in_range = (
                    self._comm_range is None
                    or math.dist(cells[agent], cells[other]) <= self._comm_range
                )
                if other not in reached and online[other] and in_range:
                    reached.add(other)
                    frontier.append(other)
        return sorted(reached)


class DialogueAllocator:
    """One agent's part in C1: speak in rounds, adopt plans that reach it, execute its task."""

    tools = (*PRIMITIVE_TOOLS, "talk_to_team")
    rules = (
        "Your team divides work by talking. YOUR TASK comes from the latest plan you heard.",
        "When you finish your task, or cannot continue it, call talk_to_team to agree a new plan.",
    )

    def __init__(
        self,
        agent_id: str,
        slot: int,
        rounds: DialogueRounds,
        task_pool: list[TaskSpec],
        client: LLMClient,
        llm: LLMConfig,
        latency: LatencyConfig,
        config: DialogueConfig,
        posted: bool,
    ) -> None:
        """Sets up the allocator.

        Args:
            agent_id: The agent this allocator speaks for.
            slot: The agent's server slot.
            rounds: The team's shared rounds.
            task_pool: The episode's task list (it grows as tasks are found).
            client: The shared model client, for dialogue turns.
            llm: How the model is queried, and how much history a prompt shows.
            latency: Reference rates turning a turn's tokens into ticks.
            config: Turn and re-plan limits.
            posted: Every mission task is known from the start (posted mode).
        """
        self._agent_id = agent_id
        self._slot = slot
        self._rounds = rounds
        self._task_pool = task_pool
        self._client = client
        self._llm = llm
        self._latency = latency
        self._config = config
        self._current: str | None = None
        self._plan_note = "no plan yet"
        self._wants_round = True
        self._retry_after = 0
        self._transcript: list[str] = []
        self._transcript_round = -1
        self._joined: set[int] = set()  # rounds this agent has been a member of
        self._posted = tuple(task.task_id for task in task_pool) if posted else ()
        self._history_seen = 0

    def assign(self, request: DecisionRequest) -> Assignment:
        """The task the latest plan gave this agent; with none, it holds still and asks to talk."""
        tasks = {task.task_id: task for task in self._task_pool}
        lines = (f"TEAM PLAN (as you last heard it): {self._plan_note}",)
        if self._current is None:
            self._wants_round = True
            # Holding still costs no LLM call, but an agent still reserving furniture must get the
            # chance to unlock it first, or it blocks teammates for the whole round.
            return Assignment(None, lines, (), hold=not request.own_reservations)
        return Assignment(tasks.get(self._current), lines, ())

    def wants_observation(self, tick: int) -> bool:
        """A turn this tick needs the agent's perception text."""
        current = self._rounds.round_of(self._agent_id)
        return (
            current is not None
            and current.speaker == self._agent_id
            and current.pending is None
            and tick >= current.next_turn_tick
        )

    def on_tick(self, view: TickView) -> TickReport:
        """Hears dialogue, adopts plans, asks for rounds, and takes its turn when due."""
        tick = view.tick
        decision_due = False
        for _, result in view.history[self._history_seen :]:
            if result.ok and result.call.name == "talk_to_team":
                self._current = None
                self._wants_round = True
                decision_due = True
        self._history_seen = len(view.history)
        current = self._rounds.round_of(self._agent_id)
        if current is not None and current.round_id != self._transcript_round:
            self._transcript = []
            self._transcript_round = current.round_id
            self._joined.add(current.round_id)
        for message in view.messages:
            # A round can close (its EXECUTE sent) before the message reaches its members.
            if message.kind != "dialogue" or int(message.payload["round"]) not in self._joined:
                continue
            self._transcript.append(f"{message.sender}: {message.payload['text']}")
            if message.payload["end"] == EXECUTE:
                plan = json.loads(message.payload["plan"])
                decision_due |= self._adopt(plan, message.sender, tick)
        broadcasts: list[tuple[str, dict[str, str]]] = []
        calls: list[LLMCallRecord] = []
        logged: list[dict[str, JsonValue]] = []
        current = self._rounds.round_of(self._agent_id)
        if current is None and self._wants_round and tick >= self._retry_after:
            current = self._rounds.try_open(self._agent_id, tick, view.team_cells, view.team_online)
            if current is not None:
                self._transcript = []
                self._transcript_round = current.round_id
                self._joined.add(current.round_id)
                logged.append(
                    {"round": current.round_id, "event": "open", "members": current.members}
                )
        if (
            current is not None
            and current.pending is None
            and not view.team_online[current.speaker]
        ):
            # A member that crashed mid-round loses its turn instead of stalling everyone.
            current.speaker_index = (current.speaker_index + 1) % len(current.members)
        if current is not None and current.speaker == self._agent_id:
            if current.pending is None and tick >= current.next_turn_tick:
                calls.append(self._speak(current, view))
            if current.pending is not None and tick >= current.pending[0]:
                payload = current.pending[1]
                current.pending = None
                broadcasts.append(("dialogue", payload))
                self._transcript.append(f"{self._agent_id} (you): {payload['text']}")
                logged.append(
                    {
                        "round": current.round_id,
                        "event": "turn",
                        "members": current.members,
                        "text": payload["text"],
                        "end": payload["end"],
                        "plan": payload["plan"],
                        "problem": payload["problem"],
                    }
                )
                decision_due |= self._finish_turn(current, payload, tick)
        return TickReport(tuple(broadcasts), decision_due, None, tuple(calls), tuple(logged))

    def unscored(self, request: DecisionRequest) -> list[TaskSpec]:
        """C1 has no utilities."""
        return []

    def set_scores(self, scores: dict[str, float]) -> None:
        """C1 takes no scores."""
        return

    def _speak(self, current: Round, view: TickView) -> LLMCallRecord:
        """One turn: an LLM call whose message, checked if it is a plan, goes out L ticks later."""
        known = sorted(
            {*self._posted, *view.known_task_ids} | ({self._current} if self._current else set())
        )
        tasks = {task.task_id: task for task in self._task_pool}
        may_execute = set(current.members) <= current.spoken | {self._agent_id}
        system = (
            Path(DIALOGUE_SYSTEM_PATH)
            .read_text(encoding="utf-8")
            .format(agent_id=self._agent_id, members=", ".join(current.members))
            .rstrip()
        )
        lines = ["TASK CONTEXT (tasks you know of):"]
        lines += [
            f"  {task_id}: {tasks[task_id].description}" for task_id in known if task_id in tasks
        ]
        lines.append(f"YOUR TASK NOW: {self._current or 'none'}")
        lines.append(capability_line(view.card))
        lines.append("YOUR RECENT ACTIONS (oldest first):")
        lines += history_lines(view.history, self._llm.history_keep) or ["  none"]
        lines.append("CURRENT OBSERVATION:")
        lines.append(view.observation or "")
        lines.append("THIS ROUND SO FAR (what reached you):")
        lines += [f"  {line}" for line in self._transcript] or ["  nothing yet"]
        feedback = self._rounds.feedback.get(self._agent_id, [])
        if feedback:
            lines.append("PLAN FEEDBACK: " + "; ".join(feedback))
        lines.append(f"YOU MAY EXECUTE: {'yes' if may_execute else 'no'}")
        user = "\n".join(lines)
        reply = self._client.complete(
            ChatRequest(
                ({"role": "system", "content": system}, {"role": "user", "content": user}),
                dialogue_schema(
                    current.members, [*known, WAIT, CONTINUE], self._config.message_chars
                ),
                "dialogue",
                self._slot,
            )
        )
        delay = latency_ticks(reply.prompt_tokens, reply.output_tokens, self._latency)
        current.pending = (
            view.tick + delay,
            self._payload(current, reply.text, known, may_execute),
        )
        current.turns += 1
        current.spoken.add(self._agent_id)
        return LLMCallRecord(
            "dialogue",
            f"{system}\n\n{user}",
            reply.text,
            reply.prompt_tokens,
            reply.output_tokens,
            reply.wall_seconds,
            reply.cached,
        )

    def _payload(
        self, current: Round, text: str, known: list[str], may_execute: bool
    ) -> dict[str, str]:
        """The turn's message, with its plan checked in code (parse, members, known tasks)."""
        try:
            reply = json.loads(text)
        except json.JSONDecodeError:
            reply = {"message": text[: self._config.message_chars], "end": PROCEED, "plan": []}
        message = str(reply.get("message", ""))[: self._config.message_chars]
        end = reply.get("end") if reply.get("end") in (PROCEED, EXECUTE) else PROCEED
        plan = reply.get("plan") if isinstance(reply.get("plan"), list) else []
        problem = None
        if end == EXECUTE:
            problem = check_plan(plan, current.members, set(known), may_execute)
        entries = {} if problem else {str(e["agent"]): str(e["assignment"]) for e in plan}
        return {
            "round": str(current.round_id),
            "text": message,
            "end": EXECUTE if end == EXECUTE and problem is None else PROCEED,
            "plan": json.dumps(entries),
            "problem": problem or "",
        }

    def _finish_turn(self, current: Round, payload: dict[str, str], tick: int) -> bool:
        """After the speaker's message goes out: adopt its own plan, re-plan, or pass the turn."""
        decision_due = False
        if payload["problem"]:
            current.replans += 1
            for member in current.members:
                self._rounds.feedback.setdefault(member, []).append(
                    f"{self._agent_id}'s plan was rejected: {payload['problem']}"
                )
        if payload["end"] == EXECUTE:
            decision_due = self._adopt(json.loads(payload["plan"]), self._agent_id, tick)
            self._rounds.close(current)
            return decision_due
        if current.replans > self._config.replans or current.turns >= self._rounds.max_turns(
            current
        ):
            self._rounds.close(current)
            self._retry_after = tick + self._config.retry_ticks
            return False
        current.speaker_index = (current.speaker_index + 1) % len(current.members)
        current.next_turn_tick = tick + 1
        return False

    def _adopt(self, plan: dict[str, str], speaker: str, tick: int) -> bool:
        """Takes this agent's part of a plan that reached it; True when its task changed.

        A WAIT leaves it without a task, and it asks to talk again only after `retry_ticks`.
        """
        assignment = plan.get(self._agent_id)
        if assignment is None:
            return False
        before = self._current
        if assignment == WAIT:
            self._current = None
        elif assignment != CONTINUE:
            self._current = assignment
        self._wants_round = self._current is None
        if self._wants_round:
            self._retry_after = tick + self._config.retry_ticks
        self._plan_note = f"from {speaker}: " + ", ".join(
            f"{agent}: {task}" for agent, task in sorted(plan.items())
        )
        return self._current != before


def check_plan(
    plan: list[JsonValue], members: list[str], known: set[str], may_execute: bool
) -> str | None:
    """Why a joint plan is invalid, or None: parse, then members, then known tasks (beliefs)."""
    if not may_execute:
        return "every member must speak before EXECUTE"
    seen = set()
    for entry in plan:
        if not isinstance(entry, dict) or "agent" not in entry or "assignment" not in entry:
            return "each plan entry needs an agent and an assignment"
        agent, assignment = entry["agent"], entry["assignment"]
        if agent not in members:
            return f"{agent} is not in this round"
        if agent in seen:
            return f"{agent} appears twice"
        seen.add(agent)
        if assignment not in (WAIT, CONTINUE) and assignment not in known:
            return f"{assignment} is not a task you know of"
    if not seen:
        return "the plan is empty"
    return None


def dialogue_schema(
    members: list[str], assignments: list[str], message_chars: int
) -> dict[str, JsonValue]:
    """The JSON schema a dialogue turn must match."""
    return {
        "type": "object",
        "properties": {
            "message": {"type": "string", "maxLength": message_chars},
            "end": {"type": "string", "enum": [PROCEED, EXECUTE]},
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string", "enum": members},
                        "assignment": {"type": "string", "enum": assignments},
                    },
                    "required": ["agent", "assignment"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["message", "end", "plan"],
        "additionalProperties": False,
    }
