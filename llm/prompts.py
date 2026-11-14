"""Prompts and replies for the LLM executor.

The system prompt states the agent's role, the tools its condition allows, and the reply format.
The observation adds to the shared perception text (`sim.agent.build_prompt`) what an LLM needs to
act: its capabilities, its current task with the task's recipe (task know-how is the same for every
condition), the allocator's own section, and the action history, compacted in code (the last
`history_keep` actions verbatim, earlier ones in one summary line).

Replies are JSON constrained by a schema: a short thought, logged and never acted on, and the tool
calls. A reply that does not parse is not acted on; the policy reports it as an error, which costs
the decision's latency like any rejected call.
"""

import json
from pathlib import Path

import yaml

from sim.agent import ActionResult, CapabilityCard, DecisionRequest, ToolCall
from sim.config import COMPLETION_TOOLS, LLM_MAX_CALLS_PER_REPLY, LLM_THOUGHT_CHARS
from sim.logger import JsonValue
from sim.primitives import TOOL_ARITY
from sim.scenario import TaskSpec

PROMPTS_DIR = Path("configs/prompts")
TOOL_LINES_PATH = PROMPTS_DIR / "tools.yaml"
EXECUTOR_SYSTEM_PATH = PROMPTS_DIR / "executor_system.txt"

# Primitives every LLM executor may call. Allocation tools are added per condition.
PRIMITIVE_TOOLS = (
    "move_to",
    "explore",
    "search",
    "pick",
    "place",
    "open",
    "close",
    "start",
    "lock",
    "unlock",
    "tie_bag",
    "scan",
    "wait",
)


def tool_descriptions() -> dict[str, str]:
    """The one-line description of every tool, as the prompt shows it."""
    with TOOL_LINES_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def executor_system(agent_id: str, tools: tuple[str, ...], rules: tuple[str, ...]) -> str:
    """The system prompt: role, the condition's tools and rules, and the reply format."""
    descriptions = tool_descriptions()
    template = EXECUTOR_SYSTEM_PATH.read_text(encoding="utf-8")
    return template.format(
        agent_id=agent_id,
        max_calls=LLM_MAX_CALLS_PER_REPLY,
        tools="\n".join(f"- {descriptions[tool]}" for tool in tools),
        rules="\n".join(f"- {rule}" for rule in rules),
    ).rstrip()


def executor_observation(
    request: DecisionRequest,
    task: TaskSpec | None,
    allocator_lines: tuple[str, ...],
    history_keep: int,
    can_report: bool,
) -> str:
    """The user message: perception, capabilities, task and recipe, allocator section, history."""
    lines = [request.prompt, capability_line(request.card)]
    if request.own_reservations:
        held = ", ".join(request.own_reservations)
        lines.append(f"YOU HOLD WORKSPACES: {held} (unlock them when you no longer need them).")
    if task is None:
        lines.append("YOUR TASK: none.")
    else:
        lines.append(f"YOUR TASK: {task.task_id}: {task.description}")
        recipe = "; ".join(f"{step.name}({', '.join(step.args)})" for step in task.steps)
        ending = f"; then report({task.task_id})" if can_report else ""
        lines.append(f"RECIPE: {recipe}{ending}")
    lines += allocator_lines
    lines.append("RECENT ACTIONS (oldest first):")
    lines += history_lines(request.history, history_keep) or ["  none"]
    return "\n".join(lines)


def capability_line(card: CapabilityCard) -> str:
    """The agent's own capability card as one line."""
    scanner = "yes" if card.can_scan else "no"
    return (
        f"CAPABILITIES: hands {card.hands}, speed {card.speed} cells per tick, scanner {scanner}."
    )


def history_lines(history: tuple[tuple[int, ActionResult], ...], keep: int) -> list[str]:
    """The last `keep` (at least 1) actions verbatim, earlier ones as one summary line."""
    older, recent = history[:-keep], history[-keep:]
    lines = []
    if older:
        failed = sum(1 for _, result in older if not result.ok)
        reported = [
            result.call.args[0]
            for _, result in older
            if result.ok and result.call.name in COMPLETION_TOOLS
        ]
        summary = f"  (earlier: {len(older)} actions, {failed} failed"
        if reported:
            summary += f"; reported {', '.join(reported)}"
        lines.append(f"{summary})")
    for tick, result in recent:
        call = f"{result.call.name}({', '.join(result.call.args)})"
        # The outcome category is an analysis label; the agent sees ok or FAILED and the reason.
        lines.append(f"  t{tick} {call} -> {'ok' if result.ok else 'FAILED'}: {result.text}")
    return lines


def tool_call_schema(tool: str) -> dict[str, JsonValue]:
    """One tool's call: exactly its number of arguments, and whole ticks for `wait`.

    Left to a free `args` list, a weak model emitted `[` and then only whitespace (the grammar
    allows any amount) until it ran out of tokens, so arity belongs in the grammar.
    """
    argument: dict[str, JsonValue] = {"type": "string"}
    if tool == "wait":
        argument["pattern"] = "^[0-9]{1,3}$"
    arity = TOOL_ARITY[tool]
    return {
        "type": "object",
        "properties": {
            "tool": {"const": tool},
            "args": {"type": "array", "items": argument, "minItems": arity, "maxItems": arity},
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def executor_schema(tools: tuple[str, ...]) -> dict[str, JsonValue]:
    """The JSON schema an executor reply must match."""
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "maxLength": LLM_THOUGHT_CHARS},
            "calls": {
                "type": "array",
                "maxItems": LLM_MAX_CALLS_PER_REPLY,
                "items": {"anyOf": [tool_call_schema(tool) for tool in tools]},
            },
        },
        "required": ["thought", "calls"],
        "additionalProperties": False,
    }


def parse_calls(text: str, tools: tuple[str, ...]) -> tuple[list[ToolCall], str | None]:
    """Reads the tool calls from a reply.

    Returns:
        (calls, None) for a usable reply, or ([], the reason) for one that is not.
    """
    try:
        reply = json.loads(text)
    except json.JSONDecodeError as error:
        return [], f"reply was not valid JSON ({error.msg})"
    if not isinstance(reply, dict) or not isinstance(reply.get("calls"), list):
        return [], 'reply needs a "calls" list'
    calls = []
    for entry in reply["calls"][:LLM_MAX_CALLS_PER_REPLY]:
        if not isinstance(entry, dict) or entry.get("tool") not in tools:
            return [], f"unknown tool in {json.dumps(entry)}"
        args = entry.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            return [], f"args of {entry['tool']} must be a list of strings"
        wanted = TOOL_ARITY[entry["tool"]]
        if len(args) != wanted:
            # A model that sees only "takes 1 arguments" repeats the call; the usage line fixes it.
            return (
                [],
                f"{entry['tool']} takes {wanted} argument(s): {tool_descriptions()[entry['tool']]}",
            )
        calls.append(ToolCall(entry["tool"], tuple(args)))
    if not calls:
        return [], "reply had no tool calls"
    return calls, None
