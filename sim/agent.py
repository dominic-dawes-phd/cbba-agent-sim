"""Agent interface shared by every condition (hard rule 7).

Conditions differ only in the `Policy` they plug in. The sim owns the agent's body, beliefs, and
action queue. A policy only sees a frozen prompt and returns tool calls.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sim.beliefs import Belief, serialize_beliefs
from sim.channel import Message
from sim.pathing import PathPlanner
from sim.perception import Viewpoint
from sim.world import Cell, World

COMPASS_NAMES = (
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
    "north",
    "northeast",
)


@dataclass(frozen=True)
class CapabilityCard:
    """Static capabilities of one agent. `success` maps primitive name to success probability."""

    hands: int
    speed: int
    view_range: float
    view_angle: float
    can_scan: bool
    success: dict[str, float]

    def success_prob(self, primitive: str) -> float:
        """Probability that a primitive succeeds once its preconditions hold. Defaults to 1."""
        return self.success.get(primitive, 1.0)


@dataclass(frozen=True)
class ToolCall:
    """One tool call as returned by a policy, e.g. ToolCall("pick", ("plate_2",))."""

    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class ActionResult:
    """Outcome of a tool call.

    Attributes:
        outcome: "ok"; "hallucination" (named an object the agent never observed);
            "invalid_call" (unknown tool, bad arguments, or the agent's own state forbids it);
            "stale_belief" (the world no longer matches the agent's belief); or
            "skill_failure" (preconditions held but the success roll failed).
    """

    call: ToolCall
    outcome: str
    text: str

    @property
    def ok(self) -> bool:
        """Whether the call succeeded."""
        return self.outcome == "ok"


@dataclass
class Skill:
    """A tool call in progress. It applies its effect when `ticks_left` reaches zero.

    `move_to` is open-ended instead: it ends when the agent is next to `targets`, or fails after
    too many ticks without getting closer (`stuck_ticks`; closer means fewer static steps than
    `best_steps`, so shuffling sideways is not progress). `drop_at_step` is the step on which a held
    object slips out of the gripper, or None. `started` is the tick the skill began, for logs.
    """

    call: ToolCall
    ticks_left: int
    started: int = 0
    interruptible: bool = False
    targets: tuple[Cell, ...] = ()
    stuck_ticks: int = 0
    best_steps: int | None = None  # fewest static steps to the target reached so far
    steps_taken: int = 0
    drop_at_step: int | None = None
    detouring: bool = False  # on an obstacle-aware path until the static distance improves


@dataclass(frozen=True)
class DecisionRequest:
    """What a policy receives at a decision point, frozen at `tick`.

    `prompt` is the observation as text for LLM policies. `beliefs` is the same observation as a
    copy of the belief graph, and `object_steps` the static step counts the prompt shows, for coded
    policies. `known_task_ids` are the tasks the agent learned during the episode (revealed or
    discovered by its own perception, or from teammates' messages). `own_reservations` is the
    furniture whose workspace this agent holds; like what it is holding, an agent knows this about
    itself. `frontier_steps` is the static step count to the nearest floor cell its team coverage
    map says is unsearched (where `explore` goes), and `own_frontier_steps` to the nearest floor it
    has not seen itself (where `search` goes); None when there is none.
    """

    agent_id: str
    tick: int
    prompt: str
    inbox: list[Message]
    last_result: ActionResult | None
    beliefs: dict[str, Belief]
    object_steps: dict[str, int | None]
    known_task_ids: tuple[str, ...]
    own_reservations: tuple[str, ...]
    frontier_steps: int | None
    own_frontier_steps: int | None
    card: CapabilityCard
    history: tuple[tuple[int, ActionResult], ...]


@dataclass(frozen=True)
class DecisionResponse:
    """A policy's tool calls plus the token counts that the latency model charges.

    `task_id` names the task these calls work on (None for idle calls), so every logged action can
    be attributed to a task and duplicated work measured afterwards. An LLM policy lists each model
    call it made in `llm_calls` (each is logged and charged its own latency) and sets `error` when
    its reply could not be used, which counts like a rejected call.
    """

    tool_calls: list[ToolCall]
    raw_text: str
    prompt_tokens: int
    output_tokens: int
    task_id: str | None = None
    llm_calls: tuple["LLMCallRecord", ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class LLMCallRecord:
    """One model call made during a decision, as logged (hard rule 9).

    Attributes:
        purpose: What it was for: "act" (choose primitives), "utility", or "dialogue".
        prompt: The full prompt, every message joined.
        cached: Answered from the response cache instead of the server.
    """

    purpose: str
    prompt: str
    raw_text: str
    prompt_tokens: int
    output_tokens: int
    wall_seconds: float
    cached: bool


@dataclass(frozen=True)
class TickView:
    """What an agent's coordination code sees every tick, between decisions.

    Allocators that run in code (CBBA, ledger gossip) read their messages and update their state
    here; nothing here is an LLM call (hard rule 2). `beliefs` is the agent's live belief graph,
    to be read, not changed; `planner` gives the static steps every condition shares (hard rule
    8). `team_cells` and `team_online` are for the protocol's own bookkeeping (who is in radio
    reach of whom when a dialogue round starts), never shown to an LLM. `observation` is the
    agent's perception text, built only when the policy asks for it (`wants_observation`).
    """

    agent_id: str
    tick: int
    cell: Cell
    beliefs: dict[str, Belief]
    known_task_ids: tuple[str, ...]
    messages: tuple[Message, ...]
    history: tuple[tuple[int, ActionResult], ...]
    card: CapabilityCard
    planner: PathPlanner
    team_cells: dict[str, Cell]
    team_online: dict[str, bool]
    observation: str | None


@dataclass(frozen=True)
class TickReport:
    """What coordination code asks of the loop after a tick.

    Attributes:
        broadcasts: (message kind, payload) pairs to send to the team over the lossy channel.
        decision_due: The agent's assignment changed, or it has something to decide (a decision
            point, hard rule 2).
        ledger_rows: Ledger rows that changed in this agent's copy, for the `ledgers` log.
        llm_calls: Model calls made in coordination (a dialogue turn), each logged and charged.
        dialogue: Dialogue events to log (`dialogues` stream).
    """

    broadcasts: tuple[tuple[str, dict[str, str]], ...] = ()
    decision_due: bool = False
    ledger_rows: dict[str, dict[str, str | int | float | None]] | None = None
    llm_calls: tuple["LLMCallRecord", ...] = ()
    dialogue: tuple[dict[str, str | int | bool | list[str] | None], ...] = ()


@runtime_checkable
class TickingPolicy(Protocol):
    """A policy whose coordination runs in code every tick (CBBA, ledger gossip)."""

    def on_tick(self, view: TickView) -> TickReport:
        """Reads this tick's coordination messages and updates the allocator's state."""
        ...

    def wants_observation(self, tick: int) -> bool:
        """Whether this tick's view needs the agent's perception text (a dialogue turn)."""
        ...


class Policy(Protocol):
    """Coordination and execution logic for one agent. The only code that differs by condition."""

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Returns the tool calls to run, in order, starting L ticks after the request."""
        ...


@dataclass
class AgentState:
    """One agent's body, beliefs, and decision state.

    Attributes:
        thinking_until: First tick at which the agent may act after its last LLM call.
        decision_due: A decision point is pending (episode start, queue finished, call failed,
            rejoin).
        unread: Messages delivered since the last decision.
        last_result: Result of the most recent tool call, shown at the next decision.
        known_tasks: Ids of tasks this agent learned during the episode (revealed, discovered, or
            heard about).
        blocked_ticks: Ticks spent unable to move (congestion metric).
        yields: Times this agent, idle, stepped aside for a blocked teammate.
        move_conflicts: Times this agent lost a same-cell conflict in the movement phase.
        current_task: The task the queued calls work on, per the policy's last response.
        searched: Team coverage map: cell -> tick it was in view (first sight while search_ttl is
            off, last sight while it is on), from own perception and teammates' coverage messages.
        own_searched: The same, from this agent's own perception only. Coverage messages say where
            teammates looked, not what they saw, so finding an object takes own sight.
        unsent_coverage: Own coverage entries not yet broadcast to teammates.
        shards: An object this agent just shattered, awaiting its `clean_up` reflex.
        task_area: The tile a reflex returns to once the shards are in the trash.
        pick_places: For each queued pick, where the agent believed the item was when it decided
            ((relation, furniture), or None for loose). A pick whose item is seen elsewhere is
            cancelled; the pick itself fails if the item moved unseen.
        picked_from: Where each held item was picked up from (furniture, slot cell), so a reflex
            can put it back.
        history: Every finished tool call with its tick: the action history an LLM prompt shows.
    """

    agent_id: str
    card: CapabilityCard
    cell: Cell
    heading: float
    beliefs: dict[str, Belief]
    held: list[str] = field(default_factory=list)
    online: bool = True
    queue: list[ToolCall] = field(default_factory=list)
    skill: Skill | None = None
    thinking_until: int = 0
    decision_due: bool = True
    unread: list[Message] = field(default_factory=list)
    last_result: ActionResult | None = None
    known_tasks: set[str] = field(default_factory=set)
    blocked_ticks: int = 0
    yields: int = 0
    move_conflicts: int = 0
    current_task: str | None = None
    searched: dict[Cell, int] = field(default_factory=dict)
    own_searched: dict[Cell, int] = field(default_factory=dict)
    unsent_coverage: dict[Cell, int] = field(default_factory=dict)
    shards: str | None = None
    task_area: Cell | None = None
    pick_places: dict[str, tuple[str, str] | None] = field(default_factory=dict)
    picked_from: dict[str, tuple[str, Cell]] = field(default_factory=dict)
    history: list[tuple[int, ActionResult]] = field(default_factory=list)

    def viewpoint(self) -> Viewpoint:
        """The agent's current viewpoint for perception."""
        return Viewpoint(self.cell, self.heading, self.card.view_range, self.card.view_angle)

    def hands_free(self, world: World) -> int:
        """Hands not occupied by held objects."""
        return self.card.hands - sum(world.objects[obj_id].hands_required for obj_id in self.held)

    def at_decision_point(self, tick: int) -> bool:
        """Whether the agent queries its policy this tick.

        A new message triggers a decision only when the agent is idle or in an interruptible skill
        (wait). Otherwise it is read when the current skill finishes.
        """
        if not self.online or tick < self.thinking_until:
            return False
        if self.decision_due:
            return True
        if not self.unread:
            return False
        return self.can_be_interrupted()

    def can_be_interrupted(self) -> bool:
        """Whether news may trigger a decision now: idle, or in an interruptible skill (wait)."""
        idle = self.skill is None and not self.queue
        return idle or (self.skill is not None and self.skill.interruptible)


def compass_name(heading: float) -> str:
    """Nearest of the eight compass directions for a heading in degrees."""
    return COMPASS_NAMES[round(heading / 45) % 8]


def build_prompt(
    agent: AgentState,
    world: World,
    planner: PathPlanner,
    tick: int,
    frontier_steps: int | None,
    own_frontier_steps: int | None,
) -> str:
    """Serializes the agent's frozen observation: pose, beliefs, search state, inbox, result."""
    holding = ", ".join(agent.held) if agent.held else "nothing"
    team = "none" if frontier_steps is None else f"{frontier_steps} steps away"
    own = "none" if own_frontier_steps is None else f"{own_frontier_steps} steps away"
    lines = [
        f"TICK {tick}. You are {agent.agent_id} in {world.rooms[agent.cell]}, "
        f"facing {compass_name(agent.heading)}. Holding: {holding}.",
        serialize_beliefs(agent.beliefs, tick, agent.cell, planner),
        f"UNSEARCHED: nearest floor nobody has searched: {team}; "
        f"nearest floor you have not seen yourself: {own}.",
        "MESSAGES:",
    ]
    for message in agent.unread:
        payload = ", ".join(f"{key}={message.payload[key]}" for key in sorted(message.payload))
        lines.append(f"  from {message.sender}: {message.kind} {payload}")
    if not agent.unread:
        lines.append("  none")
    if agent.last_result is not None:
        call = agent.last_result.call
        lines.append(
            f"LAST ACTION: {call.name}({', '.join(call.args)}) -> "
            f"{agent.last_result.outcome}: {agent.last_result.text}"
        )
    return "\n".join(lines)
