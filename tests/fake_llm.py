"""Test doubles for the LLM client, so the whole LLM pipeline runs without a GPU."""

import json
import threading

from allocation.scripted import ScriptedPolicy
from llm.client import ChatReply, ChatRequest
from sim.agent import DecisionRequest, ToolCall
from sim.config import CHARS_PER_TOKEN, ScriptedConfig
from sim.scenario import TaskSpec, goal_holds
from sim.world import World

# The coded executor's announcements, as the ledger tools of the auction conditions.
LEDGER_RENAMES = {"report": "mark_done", "release": "give_up"}


# C1 has no announcements: finishing or giving up a task means asking the team to talk.
DIALOGUE_RENAMES = {"report": "talk_to_team", "release": "talk_to_team"}


class FollowerBrain(ScriptedPolicy):
    """The coded executor on one given task, never adopting another (the allocator decides)."""

    def _choose_next_task(self, request: DecisionRequest) -> None:
        return

    def _give_back_current_task(self) -> list[ToolCall]:
        """Always announces a task it hands back, so the allocator hears of it as from an LLM."""
        task_id = self._tasks[self._task_index].task_id
        super()._give_back_current_task()
        return [ToolCall("release", (task_id,))]


class CodedBrainClient:
    """Answers executor prompts with what the coded executor would do on the agent's current task.

    Replies are real JSON in the executor schema, and token counts come from the prompt's length,
    so prompting, parsing, latency, logging, and batching are all exercised. Allocation tools the
    condition does not allow are dropped from the coded calls.
    """

    def __init__(
        self, config: ScriptedConfig, task_pool: list[TaskSpec], world: World | None = None
    ) -> None:
        """Sets up the fake.

        Args:
            config: The coded executor's parameters.
            task_pool: The episode's task list.
            world: Ground truth, which only dialogue turns read (to propose open tasks). A test
                double may cheat; no condition ever does.
        """
        self._config = config
        self._task_pool = task_pool
        self._world = world
        self._done: set[str] = set()  # done is sticky, as in the sim
        self._brains: dict[str, tuple[str | None, FollowerBrain]] = {}
        self._lock = threading.Lock()

    def complete(self, request: ChatRequest) -> ChatReply:
        """The coded executor's next calls for this decision, as an executor reply.

        A utility request gets every task scored 100. The coded executor's announcements become
        the ledger tools when the condition offers those instead (`report` -> `mark_done`).
        """
        prompt_chars = sum(len(message["content"]) for message in request.messages)
        if request.purpose == "dialogue":
            text = self._dialogue_turn(request)
            return ChatReply(
                text, prompt_chars // CHARS_PER_TOKEN, len(text) // CHARS_PER_TOKEN, 0.0, False
            )
        if request.purpose == "utility":
            task_ids = request.schema["properties"]["scores"]["items"]["properties"]["task"]["enum"]
            text = json.dumps({"scores": [{"task": task_id, "score": 100} for task_id in task_ids]})
            return ChatReply(
                text, prompt_chars // CHARS_PER_TOKEN, len(text) // CHARS_PER_TOKEN, 0.0, False
            )
        decision = request.decision
        task_id = request.task.task_id if request.task is not None else None
        with self._lock:
            current = self._brains.get(decision.agent_id)
            finished = current is not None and current[1]._task_index >= len(current[1]._tasks)
            if current is None or current[0] != task_id or finished:
                brain = FollowerBrain([], self._config, self._task_pool)
                if request.task is not None:
                    # Adopted, not its own: it can hand the task back with release, as an LLM can.
                    brain._tasks.append(request.task)
                    # The allocator assigns a task only once its predecessors are known done.
                    brain._known_done.update(request.task.after)
                current = (task_id, brain)
                self._brains[decision.agent_id] = current
        variants = request.schema["properties"]["calls"]["items"]["anyOf"]
        allowed = [variant["properties"]["tool"]["const"] for variant in variants]
        renames = {}
        if "mark_done" in allowed:
            renames = LEDGER_RENAMES
        elif "talk_to_team" in allowed:
            renames = DIALOGUE_RENAMES
        response = current[1].decide(decision)
        calls = []
        for call in response.tool_calls:
            name = renames.get(call.name, call.name)
            if name in allowed:
                calls.append(
                    {"tool": name, "args": [] if name == "talk_to_team" else list(call.args)}
                )
        calls = calls[:3] or [{"tool": "wait", "args": ["5"]}]
        text = json.dumps({"thought": "coded", "calls": calls})
        return ChatReply(
            text, prompt_chars // CHARS_PER_TOKEN, len(text) // CHARS_PER_TOKEN, 0.0, False
        )

    def _dialogue_turn(self, request: ChatRequest) -> str:
        """PROCEED until EXECUTE is allowed, then a plan giving open known tasks out in turn."""
        properties = request.schema["properties"]["plan"]["items"]["properties"]
        members = properties["agent"]["enum"]
        options = [
            option
            for option in properties["assignment"]["enum"]
            if option not in ("WAIT", "CONTINUE")
        ]
        if "YOU MAY EXECUTE: yes" not in request.messages[-1]["content"]:
            return json.dumps({"message": "my status", "end": "PROCEED", "plan": []})
        tasks = {task.task_id: task for task in self._task_pool}
        with self._lock:
            self._done |= {
                task_id
                for task_id, task in tasks.items()
                if all(goal_holds(self._world, goal) for goal in task.goals)
                and set(task.after) <= self._done
            }
            done = set(self._done)
        open_tasks = [
            task_id
            for task_id in options
            if task_id in tasks and task_id not in done and set(tasks[task_id].after) <= done
        ]
        plan = [
            {"agent": agent, "assignment": open_tasks[index] if index < len(open_tasks) else "WAIT"}
            for index, agent in enumerate(members)
        ]
        return json.dumps({"message": "plan", "end": "EXECUTE", "plan": plan})


class FixedReplyClient:
    """Answers every request with the same text (malformed replies, for error paths)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, request: ChatRequest) -> ChatReply:
        """The fixed reply."""
        return ChatReply(self._text, 100, 10, 0.0, False)
