"""C3's LLM utilities: the agent's own suitability score for each task it learns of.

When an agent knows tasks it has not scored, one LLM call scores all of them, 0-100, from the task
descriptions and the agent's capability card; distance is left to the bid's travel discount. A score
`s` becomes the suitability `s / 100` in CBBA's bid. Scores are cached per (agent, task) for the
episode, so each task is scored once per agent, independent of its bundle (hard rule 3 charges the
call's latency at the decision where it happens).
"""

import json
from pathlib import Path

from llm.prompts import PROMPTS_DIR, capability_line
from sim.agent import CapabilityCard
from sim.logger import JsonValue
from sim.scenario import TaskSpec

UTILITY_SYSTEM_PATH = PROMPTS_DIR / "utility_system.txt"
MAX_SCORE = 100
# A task the reply leaves out gets a neutral score, so it is not asked about again and again.
NEUTRAL_SCORE = 50


def utility_messages(agent_id: str, card: CapabilityCard, tasks: list[TaskSpec]) -> tuple[str, str]:
    """The system and user prompts for scoring tasks."""
    system = (
        Path(UTILITY_SYSTEM_PATH).read_text(encoding="utf-8").format(agent_id=agent_id).rstrip()
    )
    lines = [capability_line(card), "TASKS:"]
    lines += [f"  {task.task_id}: {task.description}" for task in tasks]
    return system, "\n".join(lines)


def utility_schema(task_ids: list[str]) -> dict[str, JsonValue]:
    """The JSON schema a scoring reply must match."""
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "enum": task_ids},
                        "score": {"type": "integer", "minimum": 0, "maximum": MAX_SCORE},
                    },
                    "required": ["task", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


def parse_scores(text: str, task_ids: list[str]) -> dict[str, float]:
    """Suitabilities in [0, 1] for every asked task; unparsable or missing ones are neutral."""
    scores = dict.fromkeys(task_ids, NEUTRAL_SCORE / MAX_SCORE)
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        return scores
    entries = reply.get("scores", []) if isinstance(reply, dict) else []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("task") not in scores:
            continue
        score = entry.get("score")
        if isinstance(score, int) and 0 <= score <= MAX_SCORE:
            scores[entry["task"]] = score / MAX_SCORE
    return scores
