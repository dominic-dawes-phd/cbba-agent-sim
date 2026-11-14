"""The LLM policy: an allocator decides who does what, the agent's LLM decides each primitive.

Every LLM condition is this policy with a different allocator (hard rule 7): G's nearest-task rule,
CBBA (C2, C3), or dialogue rounds (C1). The allocator names the agent's current task, adds its own
lines to the prompt, may issue calls itself (G's claim), and sets which allocation tools the LLM
may use. The LLM then picks up to `LLM_MAX_CALLS_PER_REPLY` tool calls for that task. Before that,
an allocator that wants LLM utilities (C3) gets the agent's tasks scored in one more call.

A workspace belongs to the task it was locked for. At every decision code unlocks the workspaces
the agent holds that its current task's recipe does not lock, before anything else, as the coded
executor always unlocks. A lock the LLM forgets then cannot strand teammates (a drop's clean-up
reflex never gives up on a blocked can), and an unlock lost with a rejected reply is issued again.
A task tool (report, release, mark_done, give_up) may name only the agent's own current task.
"""

from dataclasses import dataclass
from typing import Protocol

from allocation.scripted import all_items_shattered
from allocation.utility import parse_scores, utility_messages, utility_schema
from llm.client import ChatRequest, LLMClient
from llm.prompts import executor_observation, executor_schema, executor_system, parse_calls
from sim.agent import (
    DecisionRequest,
    DecisionResponse,
    LLMCallRecord,
    TickReport,
    TickView,
    ToolCall,
)
from sim.config import TASK_TOOLS, LLMConfig
from sim.scenario import TaskSpec

# A holding agent waits this long, interruptibly, before its next decision (no LLM call).
HOLD_TICKS = 5
# The tool each condition ends a task with: G reports, C2 and C3 mark it done, C1 talks.
FINISH_TOOLS = ("report", "mark_done", "talk_to_team")


@dataclass(frozen=True)
class Assignment:
    """What the allocator hands the executor at one decision.

    Attributes:
        task: The agent's current task, or None.
        lines: The allocator's section of the prompt (a bundle, ledger rows, a plan).
        prefix: Calls the allocator issues itself before the LLM's (an announcement).
        hold: The agent should hold still with no LLM call (C1's trigger, waiting for a plan).
    """

    task: TaskSpec | None
    lines: tuple[str, ...] = ()
    prefix: tuple[ToolCall, ...] = ()
    hold: bool = False


class Allocator(Protocol):
    """Who does what, for one agent. The part of an LLM condition that is not the executor."""

    tools: tuple[str, ...]
    rules: tuple[str, ...]

    def assign(self, request: DecisionRequest) -> Assignment:
        """The agent's current task at this decision, after reading the request's news."""
        ...

    def on_tick(self, view: TickView) -> TickReport:
        """Coordination in code between decisions (CBBA, ledger gossip); nothing for G."""
        ...

    def unscored(self, request: DecisionRequest) -> list[TaskSpec]:
        """Tasks to have the LLM score now (C3); empty when utilities come from code."""
        ...

    def set_scores(self, scores: dict[str, float]) -> None:
        """Receives the LLM's suitability scores, in [0, 1]."""
        ...

    def wants_observation(self, tick: int) -> bool:
        """Whether `on_tick` needs the agent's perception text this tick (a dialogue turn)."""
        ...


class LLMPolicy:
    """One agent's LLM executor working for an allocator."""

    def __init__(
        self, slot: int, allocator: Allocator, client: LLMClient, config: LLMConfig
    ) -> None:
        """Sets up the policy.

        Args:
            slot: The server slot this agent uses, so its prompt prefix stays cached.
            allocator: Decides the agent's current task.
            client: The shared model client.
            config: How the model is queried and how much history the prompt shows.
        """
        self._slot = slot
        self._allocator = allocator
        self._client = client
        self._config = config

    def on_tick(self, view: TickView) -> TickReport:
        """The allocator's coordination step for this tick."""
        return self._allocator.on_tick(view)

    def wants_observation(self, tick: int) -> bool:
        """Whether the allocator needs the perception text this tick."""
        return self._allocator.wants_observation(tick)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Scores new tasks if the allocator wants that, then asks for the next calls."""
        records = []
        pending = self._allocator.unscored(request)
        if pending:
            records.append(self._score(request, pending))
        assignment = self._allocator.assign(request)
        task_id = assignment.task.task_id if assignment.task is not None else None
        needed = set()
        if assignment.task is not None:
            needed = {step.args[0] for step in assignment.task.steps if step.name == "lock"}
        releases = [
            ToolCall("unlock", (target,))
            for target in request.own_reservations
            if target not in needed
        ]
        if assignment.hold:
            hold = ToolCall("wait", (str(HOLD_TICKS),))
            return DecisionResponse([hold], "holding for a plan", 0, 0, None, tuple(records))
        # A tool the agent's own body cannot use is not offered (no scanner, no scan).
        tools = tuple(t for t in self._allocator.tools if t != "scan" or request.card.can_scan)
        system = executor_system(request.agent_id, tools, self._allocator.rules)
        lines = assignment.lines
        if assignment.task is not None and all_items_shattered(assignment.task, request.beliefs):
            # The coded executor reads this from the task list as well: nothing is left to do.
            finish = next(tool for tool in FINISH_TOOLS if tool in tools)
            lines = (
                *lines,
                f"YOUR TASK is over: every item it moves has shattered. End it: {finish}.",
            )
        user = executor_observation(
            request, assignment.task, lines, self._config.history_keep, "report" in tools
        )
        reply = self._client.complete(
            ChatRequest(
                ({"role": "system", "content": system}, {"role": "user", "content": user}),
                executor_schema(tools),
                "act",
                self._slot,
                request,
                assignment.task,
            )
        )
        calls, error = parse_calls(reply.text, tools)
        for call in calls:
            if error is None and call.name in TASK_TOOLS and call.args[0] != task_id:
                error = f"{call.name} takes your own task ({task_id or 'you have none'})"
        record = LLMCallRecord(
            "act",
            f"{system}\n\n{user}",
            reply.text,
            reply.prompt_tokens,
            reply.output_tokens,
            reply.wall_seconds,
            reply.cached,
        )
        return DecisionResponse(
            [*releases, *assignment.prefix, *calls],
            reply.text,
            reply.prompt_tokens,
            reply.output_tokens,
            task_id,
            (*records, record),
            error,
        )

    def _score(self, request: DecisionRequest, tasks: list[TaskSpec]) -> LLMCallRecord:
        """Has the LLM score tasks for this agent (C3) and hands the scores to the allocator."""
        task_ids = [task.task_id for task in tasks]
        system, user = utility_messages(request.agent_id, request.card, tasks)
        reply = self._client.complete(
            ChatRequest(
                ({"role": "system", "content": system}, {"role": "user", "content": user}),
                utility_schema(task_ids),
                "utility",
                self._slot,
                request,
                None,
            )
        )
        self._allocator.set_scores(parse_scores(reply.text, task_ids))
        return LLMCallRecord(
            "utility",
            f"{system}\n\n{user}",
            reply.text,
            reply.prompt_tokens,
            reply.output_tokens,
            reply.wall_seconds,
            reply.cached,
        )
