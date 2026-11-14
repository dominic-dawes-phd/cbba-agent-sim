"""Scripted test policy: executes a fixed task list with no LLM.

This is scaffolding for exercising the sim loop, not one of the five conditions. It claims each task
as it starts, walks to each step's target, runs the step, reports the task done, and waits for
predecessors it has not heard about. On a failure it retries the failed step (see `_recover`). While
waiting it re-broadcasts its own reports, which keeps cross-agent precedence from deadlocking when
reports are lost. With nothing to do it explores (`explore`, nearest unsearched floor) while any
floor is unsearched, then parks next to its dock (`dock_<agent id>`, if the map has one) so it does
not stand in a doorway. A step that picks up an object it has not seen searches (`search`, nearest
floor it has not seen itself: teammates' coverage says where they looked, not what they saw).

Two stand-ins for later machinery:
- Allocation: when its own list is used up, it adopts the first discovered task it knows of that
  nobody has claimed or reported done (the finder usually knows first, so it usually takes it, which
  is what a proximity-driven auction would do). Duplicates are possible; a duplicate that finds a
  teammate already holding the task's object drops the task. The ledger replaces this in M2.
  `GreedyPolicy` (allocation/greedy.py) overrides only this choice.
- Storage choice: a place into furniture goes to a slot its beliefs say has room. If none has room,
  a dish goes to the drying rack (unpacking) or the agent looks inside a cupboard it has not seen
  (`store_dish`).

Rules that keep teammates from undoing each other's work:
- A task in a teammate's fixed assignment is never adopted, and an item a teammate is holding is
  never targeted (a mission task waits for it; an adopted task is left to the holder).
- An open or close step whose target is already believed in that state is skipped without a call.
- A wash whose dishwasher is believed mid-cycle, with nothing of the task loaded in it yet, moves to
  another dishwasher believed free. With none, the agent puts back any dish it holds where it got
  it, gives the task up (an adopted one is released; an own one goes to the end of its list), and
  does not take a wash on while its machine is believed busy.
- The agent re-reads its tasks from the pool at every decision, because the sim cuts a task when an
  item breaks or a cycle start splits it.
"""

import dataclasses
import math

from sim.agent import ActionResult, DecisionRequest, DecisionResponse, ToolCall
from sim.beliefs import Belief
from sim.config import (
    CHARS_PER_TOKEN,
    DISH_TYPES,
    EXPLORING_TOOLS,
    SCRIPTED_TOKENS_PER_CALL,
    WASH_KINDS,
    ScriptedConfig,
)
from sim.scenario import TaskSpec
from sim.triggers import DISHWASHER_TYPE, DRYING_RACK_TYPE
from sim.world import SHARDS_TYPE, split_slot

# Index of the argument a step must walk next to before it runs.
TARGET_ARG: dict[str, int] = {
    "pick": 0,
    "place": 1,
    "open": 0,
    "close": 0,
    "scan": 0,
    "start": 0,
    "lock": 0,
    "unlock": 0,
    "tie_bag": 0,
}


class ScriptedPolicy:
    """Runs one agent's scripted task list in order, then adopts discovered tasks.

    `task_pool` is the episode's task list (`scenario.tasks`), which grows as tasks are discovered.
    The policy only reads tasks whose ids reached it in `DecisionRequest.known_task_ids`, plus the
    mission tasks it was given.
    """

    # Whether a wash whose dishwasher is busy moves elsewhere or is given up (see the module
    # docstring). A replayed plan (C0) waits instead: its schedule already accounts for the cycle.
    reroutes_busy_washes = True

    def __init__(
        self,
        tasks: list[TaskSpec],
        config: ScriptedConfig,
        task_pool: list[TaskSpec],
        reserved: frozenset[str] = frozenset(),
    ) -> None:
        """Sets up the policy.

        Args:
            tasks: This agent's own task list, in order.
            config: The scripted policy's parameters.
            task_pool: The episode's task list, shared with the sim.
            reserved: Tasks in teammates' fixed assignments, never adopted.
        """
        self._tasks = list(tasks)
        self._mission_ids = {task.task_id for task in tasks}
        self._reserved = reserved
        self._retargets: dict[str, str] = {}  # wash task -> the dishwasher it moved to
        self._picked_from: dict[str, str] = {}  # item -> furniture it was believed on when picked
        self._put_back: dict[str, str] = {}  # dishes owed a return to where they were picked up
        self._now = 0  # the tick of the decision being made
        self._me = ""  # this agent's id
        # Static steps to each believed object, and the task ids known, at this decision.
        self._steps: dict[str, int | None] = {}
        self._known: tuple[str, ...] = ()
        self._task_pool = task_pool
        self._wait_ticks = config.wait_ticks
        self._task_index = 0
        self._step_index = 0
        self._known_done: set[str] = set()
        self._claimed: set[str] = set()  # tasks this agent has announced it is doing
        self._claimed_by_others: dict[str, str] = {}  # task -> teammate, from claim messages
        self._dropped: set[str] = set()  # tasks left to a teammate or beyond this agent
        self._own_done: list[str] = []
        self._sent_step: ToolCall | None = None  # the current step as last sent, targets resolved
        self._released: list[str] = []  # claims given up, re-announced while idle like reports
        self._pending: list[ToolCall] = []  # announcements for a left task: report or release
        self._parked = False

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Returns the calls for the current step, a report, or a wait."""
        for message in request.inbox:
            task_id = message.payload.get("task_id")
            if message.kind == "report":
                self._known_done.add(task_id)
            elif message.kind == "claim":
                self._claimed_by_others[task_id] = message.sender
            elif message.kind == "release" and self._claimed_by_others.get(task_id) == (
                message.sender
            ):
                del self._claimed_by_others[task_id]
        holding = {
            obj_id
            for obj_id, belief in request.beliefs.items()
            if (obj_id, "held_by", request.agent_id) in belief.relations
        }
        self._now = request.tick
        self._me = request.agent_id
        self._steps = request.object_steps
        self._known = self._known_ids(request)
        self._refresh_tasks()
        result = request.last_result
        backoff = False
        if (
            result is not None
            and result.ok
            and result.call == ToolCall("move_to", (f"dock_{request.agent_id}",))
        ):
            self._parked = True
        if result is not None and self._task_index < len(self._tasks):
            # A door already in the wanted state (a teammate left it so) means the step is met.
            already = result.call.name in ("open", "close") and " is already " in result.text
            if (result.ok or already) and result.call == self._sent_step:
                self._advance()
            elif not result.ok:
                backoff = self._recover(result)
        if self._task_index < len(self._tasks) and all_items_shattered(
            self._tasks[self._task_index], request.beliefs
        ):
            # Nothing left to do: the sim voided the task. Reporting lets anyone waiting move on.
            self._leave_current_task(report=True)
        self._skip_tasks_done_by_others(holding)
        if self._task_index >= len(self._tasks):
            self._choose_next_task(request)
        calls, task_id = self._next_calls(
            result is not None and result.ok and result.call.name in ("wait", *EXPLORING_TOOLS),
            backoff,
            request.beliefs,
            holding,
            request.agent_id,
            request.own_reservations,
            request.frontier_steps,
            request.own_frontier_steps,
            request.object_steps,
        )
        raw_text = "; ".join(f"{call.name}({', '.join(call.args)})" for call in calls)
        return DecisionResponse(
            tool_calls=calls,
            raw_text=raw_text,
            prompt_tokens=math.ceil(len(request.prompt) / CHARS_PER_TOKEN),
            output_tokens=SCRIPTED_TOKENS_PER_CALL * len(calls),
            task_id=task_id,
        )

    def _refresh_tasks(self) -> None:
        """Re-reads this agent's tasks from the pool, keeping its own dishwasher retargets.

        A changed step list (a cycle start split the task) restarts the current task from its first
        step; steps already met are then skipped.
        """
        current = {task.task_id: task for task in self._task_pool}
        for index, task in enumerate(self._tasks):
            fresh = self._retargeted(current.get(task.task_id, task))
            if fresh.steps != task.steps and index == self._task_index:
                self._step_index = 0
                self._sent_step = None
            self._tasks[index] = fresh

    def _retargeted(self, task: TaskSpec) -> TaskSpec:
        """The task with its default dishwasher swapped for the one this agent moved it to."""
        target = self._retargets.get(task.task_id)
        if target is None:
            return task
        default = _appliance_of(task)
        steps = tuple(
            ToolCall(step.name, tuple(target if arg == default else arg for arg in step.args))
            for step in task.steps
        )
        return dataclasses.replace(task, steps=steps)

    def _recover(self, result: ActionResult) -> bool:
        """Repositions the script after a failed step.

        A shattered object needs nothing here: the next step skips that item. A non-mission task
        whose object a teammate is holding, or that needs more hands than this agent has, is left.
        A full target is re-resolved from beliefs at the next step. Any other failure retries the
        same step. Leaving a task gives up its workspace reservations and its claim.

        Returns:
            Whether the retry should wait first (the blocker may clear, like a lock or a wash).
        """
        task_id = self._tasks[self._task_index].task_id
        if "shattered" in result.text:
            return False
        beyond_me = (
            " is held by " in result.text
            or " free hands" in result.text
            or result.text.endswith(" no longer exists")
        )
        if beyond_me and task_id not in self._mission_ids:
            self._dropped.add(task_id)
            self._leave_current_task(report=False)
            return False
        lost = (
            result.outcome == "hallucination"
            or result.text.endswith((" is not here", " has moved"))
            or " is no longer " in result.text
        )
        if lost and task_id not in self._mission_ids:
            # Not dropped for good: once the object is seen again the task is available again.
            self._pending += self._give_back_current_task()
            return False
        return not result.text.endswith(" full")

    def _leave_current_task(self, report: bool) -> None:
        """Stops the current task, announcing its report (if voided) or the release of its claim.

        Workspace reservations are not queued here: a decision mid-queue replaces the rest of the
        queue, so `_next_calls` releases whatever the agent still holds at every decision instead.
        Reports and releases are also re-announced while idle, in case one was cut off the same way.
        """
        task_id = self._tasks[self._task_index].task_id
        if report:
            self._pending.append(ToolCall("report", (task_id,)))
        elif task_id in self._claimed:
            self._pending.append(ToolCall("release", (task_id,)))
            self._released.append(task_id)
        self._finish_current_task(report=report)

    def _finish_current_task(self, report: bool) -> None:
        if report:
            task_id = self._tasks[self._task_index].task_id
            self._own_done.append(task_id)
            self._known_done.add(task_id)
        self._task_index += 1
        self._step_index = 0
        # A re-broadcast of the old task's report must not count as a step of the next task.
        self._sent_step = None

    def _advance(self) -> None:
        task = self._tasks[self._task_index]
        self._step_index += 1
        if self._step_index > len(task.steps):
            self._finish_current_task(report=True)

    def _skip_item(self, item_id: str) -> None:
        """Skips the current task's steps for one item, through the step that places it."""
        task = self._tasks[self._task_index]
        while self._step_index < len(task.steps):
            step = task.steps[self._step_index]
            self._advance()
            if step.name == "place" and step.args[0] == item_id:
                return

    def _skip_tasks_done_by_others(self, holding: set[str]) -> None:
        """Leaves tasks a teammate reported finished, giving up any workspace they reserved.

        A task whose item this agent is holding is not left: the report may be wrong (it may
        have taken the item out after the teammate loaded the rest), so the item goes where the
        task wants it instead of staying in the gripper forever.
        """
        while (
            self._task_index < len(self._tasks)
            and self._tasks[self._task_index].task_id in self._known_done
            and self._tasks[self._task_index].task_id not in self._own_done
            and not any(goal.obj_id in holding for goal in self._tasks[self._task_index].goals)
        ):
            self._task_index += 1
            self._step_index = 0
            self._sent_step = None

    def _available(
        self, task: TaskSpec, known_task_ids: tuple[str, ...], beliefs: dict[str, Belief]
    ) -> bool:
        """Whether a task is open for this agent to take: known, not taken, not done, doable.

        Doable means every object it picks up is believed, or sits behind a door the task opens
        first (the bowls inside the dishwasher it unpacks). The agent does not search for objects
        it has never seen; it waits until perception or a teammate's message reveals them.
        """
        shattered = all_items_shattered(task, beliefs)
        opens_first = False
        unseen_pick = False
        for step in task.steps:
            opens_first = opens_first or step.name == "open"
            if step.name == "pick" and not _located(step.args[0], beliefs) and not opens_first:
                unseen_pick = True
        return (
            task.task_id in known_task_ids
            and task.task_id not in {mine.task_id for mine in self._tasks}
            and task.task_id not in self._known_done
            and task.task_id not in self._dropped
            and task.task_id not in self._claimed_by_others
            and task.task_id not in self._reserved
            and set(task.after) <= self._known_done
            and not shattered
            and not unseen_pick
            and not self._machine_busy(self._retargeted(task), beliefs)
            and not any(
                _holder(goal.obj_id, beliefs) not in (None, self._me) for goal in task.goals
            )
        )

    def _machine_busy(self, task: TaskSpec, beliefs: dict[str, Belief]) -> bool:
        """Whether a wash's machine would still be mid-cycle on arrival, with no other one free."""
        if task.kind not in WASH_KINDS:
            return False
        machine = _appliance_of(task)
        arrival = self._steps.get(machine) or 0
        return _cycle_left(machine, beliefs, self._now) > arrival and not _free_dishwashers(
            machine, beliefs, self._now
        )

    def _known_ids(self, request: DecisionRequest) -> tuple[str, ...]:
        """Task ids this agent knows of (GreedyPolicy adds the posted ones)."""
        return request.known_task_ids

    def _other_work(self, task: TaskSpec, beliefs: dict[str, Belief]) -> bool:
        """Whether this agent has something useful to do besides the given task.

        Putting a dish back only pays when the agent can work on something else meanwhile.
        """
        own = any(
            mine.task_id != task.task_id and mine.task_id not in self._known_done
            for mine in self._tasks[self._task_index :]
        )
        return own or any(
            other.task_id != task.task_id and self._available(other, self._known, beliefs)
            for other in self._task_pool
        )

    def _cheapest_way_out(
        self,
        task: TaskSpec,
        holding: set[str],
        reachable: list[tuple[int, str]],
        object_steps: dict[str, int | None],
    ) -> int:
        """Steps the cheapest alternative to waiting at a busy machine costs.

        Moving to another free dishwasher costs the walk there. Putting the held dishes back
        costs the round trip to where they came from (nothing when holding none of them).
        Static steps (hard rule 8): a faster agent covers them in fewer ticks, so it errs toward
        waiting.
        """
        items = {goal.obj_id for goal in task.goals}
        sources = [self._picked_from.get(item) for item in holding & items]
        put_back = 2 * max(
            (object_steps.get(source) or 0 for source in sources if source is not None), default=0
        )
        return min([put_back, *(steps for steps, _ in reachable)])

    def _choose_next_task(self, request: DecisionRequest) -> None:
        """Adopts the first available discovered task, in discovery order."""
        for task in self._task_pool:
            if self._available(task, request.known_task_ids, request.beliefs):
                self._tasks.append(task)
                self._parked = False
                return

    def _expected_call(self) -> ToolCall:
        task = self._tasks[self._task_index]
        if self._step_index < len(task.steps):
            return task.steps[self._step_index]
        return ToolCall("report", (task.task_id,))

    def _next_calls(
        self,
        finished_idle: bool,
        backoff: bool,
        beliefs: dict[str, Belief],
        holding: set[str],
        agent_id: str,
        own_reservations: tuple[str, ...],
        frontier_steps: int | None,
        own_frontier_steps: int | None,
        object_steps: dict[str, int | None],
    ) -> tuple[list[ToolCall], str | None]:
        """The next calls and the task they work on (None for idle and wrap-up calls).

        Between tasks (none in progress, or the next one not yet begun) the agent first gives up
        any workspace it still holds, whichever way the last task ended. With no task it explores
        floor nobody has searched (hands empty), then parks. A pick of an object it has never seen
        becomes a search of floor it has not seen itself.
        """
        wait = ToolCall("wait", (str(self._wait_ticks),))
        explore = ToolCall("explore", ())
        can_explore = frontier_steps is not None and not holding
        can_search = own_frontier_steps is not None and not holding
        if self._pending:
            pending = self._pending
            self._pending = []
            return pending, None
        # News can cut a queue short, so a dish owed a put-back is asked for again until it is down.
        self._put_back = {
            item: source for item, source in self._put_back.items() if item in holding
        }
        if self._put_back:
            return self._put_back_calls(beliefs), None
        between_tasks = self._task_index >= len(self._tasks) or self._step_index == 0
        if between_tasks and own_reservations:
            return [ToolCall("unlock", (furniture_id,)) for furniture_id in own_reservations], None
        reports = [ToolCall("report", (task_id,)) for task_id in self._own_done]
        releases = [ToolCall("release", (task_id,)) for task_id in self._released]
        announcements = [*reports, *releases] if finished_idle else []
        idle_calls = [*announcements, wait]
        dock = f"dock_{agent_id}"
        if self._task_index >= len(self._tasks):
            if can_explore:
                self._parked = False
                return [*announcements, explore], None
            if not self._parked and dock in beliefs:
                return [ToolCall("move_to", (dock,))], None
            return idle_calls, None
        task = self._tasks[self._task_index]
        if self._step_index == 0 and not set(task.after) <= self._known_done:
            return idle_calls, None
        step = self._expected_call()
        # A shattered item is simply gone from the task. Already holding what this step would pick
        # up (a pick whose result was not seen) moves on. An item already where the task wants it
        # (a teammate loaded it) is skipped: never take it back out. A door already in the wanted
        # state needs no call.
        while _step_met(step, task, beliefs, holding):
            if step.name in ("place", "open", "close") or step.args[0] in holding:
                self._advance()
            else:
                self._skip_item(step.args[0])
            if self._task_index >= len(self._tasks) or self._step_index == 0:
                return [wait], None
            step = self._expected_call()
        holder = _holder(step.args[0], beliefs) if step.name == "pick" else None
        if holder not in (None, agent_id):
            if task.task_id in self._mission_ids:
                return [wait], task.task_id
            return [*self._give_back_current_task(), wait], None
        machine = _appliance_of(task)
        if (
            task.kind in WASH_KINDS
            and _running(machine, beliefs, self._now)
            and not _loaded_into(task, machine, beliefs)
        ):
            left = _cycle_left(machine, beliefs, self._now)
            reachable = [
                (object_steps[obj_id], obj_id)
                for obj_id in _free_dishwashers(machine, beliefs, self._now)
                if object_steps.get(obj_id) is not None
            ]
            soon = left <= self._cheapest_way_out(task, holding, reachable, object_steps)
            idle_otherwise = not reachable and not self._other_work(task, beliefs)
            if not self.reroutes_busy_washes or soon or idle_otherwise:
                return [ToolCall("wait", (str(left),))], task.task_id
            if not reachable:
                return self._give_up_wash(task, holding, beliefs), None
            self._retargets[task.task_id] = min(reachable)[1]
            self._tasks[self._task_index] = self._retargeted(task)
            task = self._tasks[self._task_index]
            step = self._expected_call()
        if step.name == "pick" and not _located(step.args[0], beliefs):
            return ([ToolCall("search", ())] if can_search else [wait]), task.task_id
        calls = [wait] if backoff else []
        if task.task_id not in self._claimed and task.task_id not in self._mission_ids:
            # Only tasks chosen at run time are claimed; a fixed assignment is agreed in advance,
            # and every message costs its receivers a decision. A claim is always the first call,
            # so it runs before any interruption can drop it.
            calls.insert(0, ToolCall("claim", (task.task_id,)))
            self._claimed.add(task.task_id)
            if task.task_id in self._released:
                self._released.remove(task.task_id)  # its old release must not undo this claim
        if step.name == "place":
            target = resolve_place_target(step.args[0], step.args[1], task.kind, beliefs)
            if target is None:
                return [*calls, wait], task.task_id
            step = ToolCall("place", (step.args[0], target))
            base = split_slot(target)[0]
            if beliefs[base].visual_attrs.get("state") == "closed":
                calls += [ToolCall("move_to", (base,)), ToolCall("open", (base,))]
        if step.name == "pick":
            source = _parent_of(step.args[0], beliefs)
            if source is not None:
                self._picked_from[step.args[0]] = source
        self._sent_step = step
        if step.name in TARGET_ARG:
            calls.append(ToolCall("move_to", (step.args[TARGET_ARG[step.name]],)))
        return [*calls, step], task.task_id

    def _give_up_wash(
        self, task: TaskSpec, holding: set[str], beliefs: dict[str, Belief]
    ) -> list[ToolCall]:
        """Its dishwasher is busy and no other is free: put dishes back and do something else.

        Each held dish of the task goes back where it was picked up. An adopted task is released
        (and not taken again while the machine is believed busy); an own task moves to the end of
        this agent's list.
        """
        items = {goal.obj_id for goal in task.goals}
        for item in sorted(holding & items):
            source = self._picked_from.get(item)
            if source is not None and source in beliefs:
                self._put_back[item] = source
        calls = self._put_back_calls(beliefs)
        if task.task_id in self._mission_ids:
            self._tasks.append(self._tasks.pop(self._task_index))
            self._step_index = 0
            self._sent_step = None
            return [*calls, ToolCall("wait", (str(self._wait_ticks),))]
        return [*calls, *self._give_back_current_task(), ToolCall("wait", (str(self._wait_ticks),))]

    def _put_back_calls(self, beliefs: dict[str, Belief]) -> list[ToolCall]:
        """Walks each dish owed a put-back to where it was picked up and places it there."""
        calls = []
        for item, source in sorted(self._put_back.items()):
            target = resolve_place_target(item, source, None, beliefs) or source
            calls += [
                ToolCall("move_to", (split_slot(target)[0],)),
                ToolCall("place", (item, target)),
            ]
        return calls

    def _give_back_current_task(self) -> list[ToolCall]:
        """Hands an adopted task back to the team, free to be taken again (by this agent too).

        Unlike leaving it, the task comes off this agent's list, so it can be adopted again once
        whatever stood in the way (a busy dishwasher, a teammate holding its item) has cleared.

        Returns:
            The release announcement, if the task was claimed.
        """
        task_id = self._tasks.pop(self._task_index).task_id
        self._step_index = 0
        self._sent_step = None
        if task_id not in self._claimed:
            return []
        self._claimed.discard(task_id)
        self._released.append(task_id)
        return [ToolCall("release", (task_id,))]


def resolve_place_target(
    item_id: str, target: str, kind: str | None, beliefs: dict[str, Belief]
) -> str | None:
    """Chooses where a place step really goes, from the agent's beliefs.

    A furniture id stays as it is while its fill is unknown, becomes one slot with room when its
    fill is known, and falls back when it is believed full. A furniture type ("cabinet") means any
    furniture of that type. Fallback: another furniture of the same type believed to have room;
    for `store_dish`, a cupboard not yet looked inside; for any other dish, the drying rack.

    Returns:
        The target to send, or None to wait.
    """
    if target in beliefs:
        slot = _slot_with_room(target, beliefs[target])
        if slot is not None or "slots" not in beliefs[target].visual_attrs:
            return slot or target
        furniture_type = beliefs[target].obj_type
    else:
        furniture_type = target
    candidates = sorted(
        obj_id for obj_id, belief in beliefs.items() if belief.obj_type == furniture_type
    )
    for candidate in candidates:
        slot = _slot_with_room(candidate, beliefs[candidate])
        if slot is not None:
            return slot
    if kind == "store_dish":
        unseen = [obj_id for obj_id in candidates if "slots" not in beliefs[obj_id].visual_attrs]
        return unseen[0] if unseen else None
    item = beliefs.get(item_id)
    racks = sorted(
        obj_id for obj_id, belief in beliefs.items() if belief.obj_type == DRYING_RACK_TYPE
    )
    if item is not None and item.obj_type in DISH_TYPES and racks:
        return racks[0]
    return None


def _already_placed(item_id: str, task: TaskSpec, beliefs: dict[str, Belief]) -> bool:
    """Whether the agent believes an item already meets its goal in this task.

    That is in or on the goal furniture, or (an `in_kind` goal) in or on furniture of a listed type,
    or (a `washed` goal) loaded into any dishwasher or already clean.
    """
    belief = beliefs.get(item_id)
    if belief is None or belief.missing:
        return False
    for goal in task.goals:
        if goal.obj_id != item_id:
            continue
        if goal.relation in ("in", "on"):
            return (item_id, goal.relation, goal.value) in belief.relations
        if goal.relation == "washed":
            return belief.visual_attrs.get("dirty") == "no" or any(
                relation == "in"
                and parent in beliefs
                and beliefs[parent].obj_type == DISHWASHER_TYPE
                for _, relation, parent in belief.relations
            )
        if goal.relation == "in_kind":
            kinds = goal.value.split("|")
            return any(
                relation in ("in", "on") and parent in beliefs and beliefs[parent].obj_type in kinds
                for _, relation, parent in belief.relations
            )
    return False


def _step_met(
    step: ToolCall, task: TaskSpec, beliefs: dict[str, Belief], holding: set[str]
) -> bool:
    """Whether a step needs no call.

    A pick or place needs none when its item shattered, and a pick when the item is in hand or
    already where the task wants it. An open or close needs none when the door is already believed
    in the state it asks for.
    """
    if step.name in ("open", "close"):
        wanted = "open" if step.name == "open" else "closed"
        belief = beliefs.get(step.args[0])
        return belief is not None and belief.visual_attrs.get("state") == wanted
    if step.name not in ("pick", "place"):
        return False
    item = step.args[0]
    if _is_shards(item, beliefs):
        return True
    return step.name == "pick" and (item in holding or _already_placed(item, task, beliefs))


def _appliance_of(task: TaskSpec) -> str | None:
    """The dishwasher a wash task's steps start (None for other kinds)."""
    for step in task.steps:
        if step.name == "start":
            return step.args[0]
    return None


def _running(machine: str | None, beliefs: dict[str, Belief], tick: int) -> bool:
    """Whether the agent believes a dishwasher is mid-cycle now."""
    return _cycle_left(machine, beliefs, tick) > 0


def _cycle_left(machine: str | None, beliefs: dict[str, Belief], tick: int) -> int:
    """Ticks the agent believes a dishwasher's cycle still has to run (0 when not running).

    A sighting of a running cycle comes with its countdown, so the cycle is believed over once the
    ticks left at that sighting have passed, without seeing the machine again.
    """
    belief = beliefs.get(machine)
    if belief is None or belief.visual_attrs.get("cycle") != "running":
        return 0
    return max(0, belief.last_seen + int(belief.visual_attrs.get("cycle_left", "0")) - tick)


def _free_dishwashers(busy: str | None, beliefs: dict[str, Belief], tick: int) -> list[str]:
    """Other dishwashers the agent knows of and does not believe mid-cycle."""
    return [
        obj_id
        for obj_id, belief in sorted(beliefs.items())
        if belief.obj_type == DISHWASHER_TYPE
        and obj_id != busy
        and not _running(obj_id, beliefs, tick)
    ]


def _loaded_into(task: TaskSpec, machine: str | None, beliefs: dict[str, Belief]) -> bool:
    """Whether the agent believes any of the task's items is inside the machine."""
    return any(
        goal.obj_id in beliefs and (goal.obj_id, "in", machine) in beliefs[goal.obj_id].relations
        for goal in task.goals
    )


def _holder(item: str, beliefs: dict[str, Belief]) -> str | None:
    """The agent believed to be holding an item, if any."""
    belief = beliefs.get(item)
    if belief is None:
        return None
    for _, relation, other in belief.relations:
        if relation == "held_by":
            return other
    return None


def _parent_of(item: str, beliefs: dict[str, Belief]) -> str | None:
    """The furniture an item is believed in or on, if any."""
    belief = beliefs.get(item)
    if belief is None:
        return None
    for _, relation, parent in belief.relations:
        if relation in ("in", "on"):
            return parent
    return None


def _located(obj_id: str, beliefs: dict[str, Belief]) -> bool:
    """Whether the agent believes it knows where an object is (seen, and not since missing)."""
    return obj_id in beliefs and not beliefs[obj_id].missing


def _is_shards(obj_id: str, beliefs: dict[str, Belief]) -> bool:
    """Whether the agent believes an object has shattered."""
    return obj_id in beliefs and beliefs[obj_id].obj_type == SHARDS_TYPE


def all_items_shattered(task: TaskSpec, beliefs: dict[str, Belief]) -> bool:
    """Whether the task has no item left: every one it is about shattered, or the sim voided it.

    A break removes the item's goals, so a task whose goals no longer mention anything it picks up
    has nothing to do. Goals about furniture (a door) are not items, and a task with only those is
    not over.
    """
    picked = {step.args[0] for step in task.steps if step.name == "pick"}
    if picked and not picked & {goal.obj_id for goal in task.goals}:
        return True
    items = [
        goal.obj_id
        for goal in task.goals
        if goal.obj_id not in beliefs or not beliefs[goal.obj_id].footprint
    ]
    return bool(items) and all(_is_shards(obj_id, beliefs) for obj_id in items)


def _slot_with_room(furniture_id: str, belief: Belief) -> str | None:
    """The first slot believed to have room, as "id[i]" (or "id" for one-cell furniture)."""
    slots = belief.visual_attrs.get("slots")
    if slots is None:
        return None
    for index, fill in enumerate(slots.split(",")):
        count, capacity = fill.split("/")
        if int(count) < int(capacity):
            return furniture_id if len(belief.cells()) == 1 else f"{furniture_id}[{index}]"
    return None
