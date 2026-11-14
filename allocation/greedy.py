"""Greedy baseline (condition G): nearest available task, announced by a claim, no consensus.

Each agent, whenever it is free, takes the task closest to it (static BFS steps, the same numbers
its prompt shows; hard rule 8) among the tasks it knows of that no teammate has claimed or reported
done and whose predecessors it has heard are done. It broadcasts a claim and does the task. Nobody
arbitrates: two agents that choose the same task before hearing each other's claim both do it, and
a lost claim message has the same effect. That duplicated work is what consensus (CBBA) is meant to
remove, so this is the reference for "any coordination versus consensus".

Execution is the shared scripted executor (`ScriptedPolicy`); only the task choice differs.
"""

from allocation.llm_policy import Assignment
from allocation.scripted import ScriptedPolicy
from llm.prompts import PRIMITIVE_TOOLS
from sim.agent import DecisionRequest, TickReport, TickView, ToolCall
from sim.config import RELEASE_RETAKE_TICKS, ScriptedConfig
from sim.scenario import TaskSpec
from sim.world import split_slot


class GreedyPolicy(ScriptedPolicy):
    """Takes the nearest available task each time it is free. Starts with no tasks of its own.

    With `posted` it knows every mission task from the start (they are posted to the whole team);
    otherwise it knows only the tasks it reveals or discovers itself or hears about. A task it
    cannot do (too few hands) or that a teammate turns out to be holding is released for others.
    """

    def __init__(self, config: ScriptedConfig, task_pool: list[TaskSpec], posted: bool) -> None:
        super().__init__([], config, task_pool)
        self._posted = tuple(task.task_id for task in task_pool) if posted else ()

    def _known_ids(self, request: DecisionRequest) -> tuple[str, ...]:
        """The posted mission tasks plus the ones this agent learned of."""
        return (*self._posted, *request.known_task_ids)

    def _choose_next_task(self, request: DecisionRequest) -> None:
        known = self._known_ids(request)
        best: tuple[int, int, TaskSpec] | None = None
        for order, task in enumerate(self._task_pool):
            if not self._available(task, known, request.beliefs):
                continue
            steps = task_steps(task, request)
            if steps is not None and (best is None or (steps, order) < best[:2]):
                best = (steps, order, task)
        if best is not None:
            self._tasks.append(best[2])
            self._parked = False


def task_steps(task: TaskSpec, request: DecisionRequest) -> int | None:
    """Static steps from the agent to where a task happens.

    The place is the first goal object the agent believes in, else the first step target it
    believes in (furniture is known from the map, so a dishwasher task always has a place).

    Returns:
        The step count, or None if the agent knows of nowhere to go for this task.
    """
    candidates = [goal.obj_id for goal in task.goals] + [
        split_slot(step.args[0])[0] for step in task.steps if step.args
    ]
    for obj_id in candidates:
        belief = request.beliefs.get(obj_id)
        if belief is not None and not belief.missing:
            return request.object_steps.get(obj_id)
    return None


class GreedyAllocator:
    """Baseline G's allocation for the LLM executor: the nearest available task, claimed in code.

    The same rule as `GreedyPolicy`: whenever free, take the nearest known task that no teammate
    has claimed or reported done and whose predecessors are known done, and announce a claim. The
    claim is issued by code (it is part of G's protocol, not a judgment), and the LLM executes the
    task, then reports it done or releases it.
    """

    tools = (*PRIMITIVE_TOOLS, "report", "release")
    rules = (
        "When your task's goal holds, report it; teammates then know it is done.",
        "If you cannot finish your task, release it so a teammate can take it.",
    )

    def __init__(self, task_pool: list[TaskSpec], posted: bool) -> None:
        """Sets up the allocator.

        Args:
            task_pool: The episode's task list, shared with the sim (it grows as tasks are found).
            posted: Every mission task is known from the start (posted mode).
        """
        self._task_pool = task_pool
        self._posted = tuple(task.task_id for task in task_pool) if posted else ()
        self._current: str | None = None
        self._announced = False
        self._known_done: set[str] = set()
        self._claimed_by_others: dict[str, str] = {}
        self._released: dict[str, int] = {}  # task -> tick from which this agent may retake it
        self._history_seen = 0  # action-history entries already read

    def assign(self, request: DecisionRequest) -> Assignment:
        """Reads the news, keeps the current task while it is open, else takes the nearest one."""
        for message in request.inbox:
            task_id = message.payload.get("task_id")
            if message.kind == "report":
                self._known_done.add(task_id)
            elif message.kind == "claim":
                self._claimed_by_others[task_id] = message.sender
            elif (
                message.kind == "release" and self._claimed_by_others.get(task_id) == message.sender
            ):
                del self._claimed_by_others[task_id]
        # Own announcements since the last decision, wherever they sat in the queue.
        for _, result in request.history[self._history_seen :]:
            if not result.ok or not result.call.args:
                continue
            task_id = result.call.args[0]
            if result.call.name == "report":
                self._known_done.add(task_id)
            elif result.call.name == "claim" and task_id == self._current:
                self._announced = True
            elif result.call.name == "release" and task_id == self._current:
                self._released[task_id] = request.tick + RELEASE_RETAKE_TICKS
                self._current = None
        self._history_seen = len(request.history)
        if self._current in self._known_done:
            self._current = None
        if self._current is None:
            self._current = self._nearest_available(request)
            self._announced = False
        tasks = {task.task_id: task for task in self._task_pool}
        task = tasks.get(self._current)
        prefix = (ToolCall("claim", (task.task_id,)),) if task and not self._announced else ()
        # What G's code knows from announcements, which the coded executor uses too.
        lines = []
        if self._claimed_by_others:
            claims = ", ".join(
                f"{task_id} ({agent})" for task_id, agent in sorted(self._claimed_by_others.items())
            )
            lines.append(f"CLAIMED BY TEAMMATES: {claims}")
        if self._known_done:
            lines.append(f"REPORTED DONE: {', '.join(sorted(self._known_done))}")
        return Assignment(task, tuple(lines), prefix)

    def on_tick(self, view: TickView) -> TickReport:
        """G coordinates only through announcements, which arrive as decision-time messages."""
        return TickReport()

    def unscored(self, request: DecisionRequest) -> list[TaskSpec]:
        """G's utility is distance, computed in code."""
        return []

    def wants_observation(self, tick: int) -> bool:
        """Its coordination never needs the perception text."""
        return False

    def set_scores(self, scores: dict[str, float]) -> None:
        """G takes no LLM scores."""
        return

    def _nearest_available(self, request: DecisionRequest) -> str | None:
        known = {*self._posted, *request.known_task_ids}
        best: tuple[int, int, str] | None = None
        for order, task in enumerate(self._task_pool):
            available = (
                task.task_id in known
                and task.task_id not in self._known_done
                and task.task_id not in self._claimed_by_others
                and self._released.get(task.task_id, 0) <= request.tick
                and set(task.after) <= self._known_done
            )
            steps = task_steps(task, request) if available else None
            if steps is not None and (best is None or (steps, order) < best[:2]):
                best = (steps, order, task.task_id)
        return best[2] if best is not None else None
