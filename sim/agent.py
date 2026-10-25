"""Agent interface shared by every condition (hard rule 7).

Conditions differ only in the `Policy` they plug in. The sim owns the agent's body, beliefs, and
action queue. A policy only sees a frozen prompt and returns tool calls.
"""

from dataclasses import dataclass, field
from typing import Protocol

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
    """A tool call in progress. It applies its effect when `ticks_left` reaches zero."""

    call: ToolCall
    ticks_left: int
    interruptible: bool = False
    target_cell: Cell | None = None


@dataclass(frozen=True)
class DecisionRequest:
    """What a policy receives at a decision point. The prompt is frozen at `tick`."""

    agent_id: str
    tick: int
    prompt: str
    inbox: list[Message]
    last_result: ActionResult | None


@dataclass(frozen=True)
class DecisionResponse:
    """A policy's tool calls plus the token counts that the latency model charges."""

    tool_calls: list[ToolCall]
    raw_text: str
    prompt_tokens: int
    output_tokens: int


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
        idle = self.skill is None and not self.queue
        return idle or (self.skill is not None and self.skill.interruptible)


def compass_name(heading: float) -> str:
    """Nearest of the eight compass directions for a heading in degrees."""
    return COMPASS_NAMES[round(heading / 45) % 8]


def build_prompt(agent: AgentState, world: World, planner: PathPlanner, tick: int) -> str:
    """Serializes the agent's frozen observation: pose, beliefs, inbox, and last result."""
    holding = ", ".join(agent.held) if agent.held else "nothing"
    lines = [
        f"TICK {tick}. You are {agent.agent_id} in {world.rooms[agent.cell]}, "
        f"facing {compass_name(agent.heading)}. Holding: {holding}.",
        serialize_beliefs(agent.beliefs, tick, agent.cell, planner),
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
