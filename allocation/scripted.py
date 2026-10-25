"""Scripted test policy: executes a fixed task list with no LLM.

This is scaffolding for exercising the sim loop, not one of the five conditions. It walks to each
step's target, runs the step, reports the task done, and waits for predecessors it has not heard
about. On a failure it restarts the current task from its first step. While waiting it
re-broadcasts its own reports, which keeps cross-agent precedence from deadlocking when reports
are lost.
"""

import math

from sim.agent import DecisionRequest, DecisionResponse, ToolCall
from sim.config import CHARS_PER_TOKEN, SCRIPTED_TOKENS_PER_CALL, ScriptedConfig
from sim.scenario import TaskSpec

# Index of the argument a step must walk next to before it runs.
TARGET_ARG: dict[str, int] = {"pick": 0, "place": 1, "open": 0, "close": 0, "scan": 0}


class ScriptedPolicy:
    """Runs one agent's scripted task list in order."""

    def __init__(self, tasks: list[TaskSpec], config: ScriptedConfig) -> None:
        self._tasks = tasks
        self._wait_ticks = config.wait_ticks
        self._task_index = 0
        self._step_index = 0
        self._known_done: set[str] = set()
        self._own_done: list[str] = []

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Returns the calls for the current step, a report, or a wait."""
        for message in request.inbox:
            if message.kind == "report":
                self._known_done.add(message.payload["task_id"])
        result = request.last_result
        if result is not None and self._task_index < len(self._tasks):
            if result.ok and result.call == self._expected_call():
                self._advance()
            elif not result.ok:
                self._step_index = 0
        calls = self._next_calls(result is not None and result.ok and result.call.name == "wait")
        raw_text = "; ".join(f"{call.name}({', '.join(call.args)})" for call in calls)
        return DecisionResponse(
            tool_calls=calls,
            raw_text=raw_text,
            prompt_tokens=math.ceil(len(request.prompt) / CHARS_PER_TOKEN),
            output_tokens=SCRIPTED_TOKENS_PER_CALL * len(calls),
        )

    def _expected_call(self) -> ToolCall:
        task = self._tasks[self._task_index]
        if self._step_index < len(task.steps):
            return task.steps[self._step_index]
        return ToolCall("report", (task.task_id,))

    def _advance(self) -> None:
        task = self._tasks[self._task_index]
        self._step_index += 1
        if self._step_index > len(task.steps):
            self._own_done.append(task.task_id)
            self._known_done.add(task.task_id)
            self._task_index += 1
            self._step_index = 0

    def _next_calls(self, finished_wait: bool) -> list[ToolCall]:
        wait = ToolCall("wait", (str(self._wait_ticks),))
        reports = [ToolCall("report", (task_id,)) for task_id in self._own_done]
        if self._task_index >= len(self._tasks):
            return [*reports, wait] if finished_wait else [wait]
        task = self._tasks[self._task_index]
        if self._step_index == 0 and not set(task.after) <= self._known_done:
            return [*reports, wait] if finished_wait else [wait]
        step = self._expected_call()
        if step.name in TARGET_ARG:
            return [ToolCall("move_to", (step.args[TARGET_ARG[step.name]],)), step]
        return [step]
