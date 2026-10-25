# Simulator design

Design reference for the custom simulator. Decisions and their reasons are in `decisions.md`; this file says how the pieces fit together. The perception, relation, belief, channel, and latency snippets below were run against small unit checks (walls block sight, closed containers hide contents, negative evidence marks objects missing); treat them as reference implementations, not final code.

## 1. Overview

- 2D household/kitchen grid with rooms, walls, furniture, and objects.
- About 6 heterogeneous agents (e.g., two-hand arm, one-hand arm, drone, carrier, wipe-only robot), each with a capability card.
- A mission of 15-30 tasks; each task is a sequence of primitive steps with precedence constraints.
- A message-level communication channel with loss, range, wall attenuation, and optional latency.
- Agent churn: `k` agents crash at a random tick and rejoin after `delta` ticks.
- One model server shared by all agents. Each agent is a separate conversation.
- Single-threaded, deterministic Python sim loop; the model server is a separate process.

Precedent: PARTNR, LLaMAR, CoELA (skills + symbolic observations), MindAgent's CuisineWorld and LLM-Coordination's Overcooked-AI (kitchen/game worlds are accepted).

## 2. Two clocks

| | Sim clock | Wall clock |
|---|---|---|
| What | Integer tick counter owned by `World` | Real seconds on the machine |
| Advances by | The sim loop | Physics |
| Used for | Every reported result (makespan, recovery time, horizons) | Cost metrics and logs only |
| Depends on hardware? | No | Yes |

**Rule: no result in the paper depends on wall-clock time.** If makespan changed when moving from the laptop GPU to a cluster GPU, the experiment would measure the hardware.

## 3. Event-driven loop and the latency model

Agents query their LLM only at **decision points**: episode start, a skill finished or failed, a message arrived, or the task assignment changed. Skills such as `move_to(sink)` run scripted over many ticks with no LLM call.

As built in M1 (`sim/agent.py`, `sim/loop.py`):

- A policy returns a **list** of tool calls, which run in order. The next decision point comes when the list is used up, a call fails, or the agent rejoins after a crash.
- A message triggers a decision immediately only if the agent is idle or in an interruptible skill (`wait`). Otherwise it is read when the current skill finishes, and the new decision replaces whatever is left of the list.
- If any returned call fails the belief check, the whole response is rejected. The agent still pays `L`, and the failure is shown at its next decision.
- Every skill takes at least one tick.
- Order within a tick: churn, message delivery, perception, decisions, skill steps, ground-truth task progress, snapshot. Perception runs before decisions, so a prompt frozen at tick `t` shows what is in view at `t`.

At a decision point the sim:

1. Freezes the agent's observation (belief graph serialized to text) at tick `t`.
2. Sends the request and **blocks in wall-clock time** while the sim clock stays at `t`. If several agents decide on the same tick, send their requests together so the server can batch them.
3. Computes the latency `L` in ticks from token counts.
4. Schedules the returned tool calls to start at tick `t + L`. During those `L` ticks the agent idles; other agents keep executing.
5. At tick `t + L`, re-checks preconditions against ground truth. If they fail (the plate is gone, the gripper is full), the failure is returned to the LLM as text and triggers a new decision.

```python
import math


def latency_ticks(
    prompt_tokens: int,
    output_tokens: int,
    prefill_rate: float,
    decode_rate: float,
    seconds_per_tick: float,
) -> int:
    """Deliberation cost in simulated ticks, from token counts at fixed reference rates."""
    return math.ceil(
        (prompt_tokens / prefill_rate + output_tokens / decode_rate) / seconds_per_tick
    )
```

Proposed defaults (provisional): `prefill_rate = 2000` tok/s, `decode_rate = 40` tok/s, `seconds_per_tick = 1.0`. With these, a 500-token prompt and 150-token response cost `L = 4` ticks. Run sensitivity ablations with `L = 0` and with a fixed `L`.

Everything is charged in the same currency: a CBBA consensus hop costs 1 tick plus channel latency; an LLM negotiation message costs `L` to write plus channel latency to deliver. That puts CBBA and LLM-only negotiation on one comparable timeline.

```python
def run_episode(world, agents, llm_client, latency_model, max_ticks):
    for tick in range(max_ticks):
        deciding = [agent for agent in agents if agent.at_decision_point(tick)]
        if deciding:
            requests = [agent.build_request(world.observe(agent)) for agent in deciding]
            responses = llm_client.batch_complete(requests)  # wall-clock wait; sim clock frozen
            for agent, request, response in zip(deciding, requests, responses):
                delay = latency_model.ticks_for(response.usage)
                agent.schedule(response.tool_calls, start_tick=tick + delay)
                world.logger.llm_call(tick, agent.id, request, response, delay)  # this agent's request only
        world.network.deliver(tick)  # messages sent earlier arrive or are dropped
        for agent in agents:
            agent.step_skill(world, tick)  # move one cell, finish a pick, or idle while "thinking"
        world.logger.snapshot(tick, world, agents)
```

Later optimization (identical results, faster): keep ticking until `t + L - 1` before blocking, so the GPU and the sim overlap.

## 4. World ground truth

Objects live in a scene-graph tree (room, furniture, object). `on` and `in` are stored as a parent pointer and updated whenever something is placed, picked up, or dropped. The drop rule is "parent becomes the adjacent table if one exists, else the floor."

```python
from dataclasses import dataclass, field


@dataclass
class WorldObject:
    obj_id: str
    obj_type: str  # "plate", "pot", "dishwasher"
    cell: tuple[int, int]
    parent_id: str | None  # what it is on/in; None means floor
    parent_relation: str | None  # "on" | "in"
    visual_attrs: dict[str, str]  # perceivable: {"state": "open", "dirty": "yes"}
    hidden_attrs: dict[str, str]  # never perceived: {"fragile": "yes"}
    is_container: bool = False
    is_open: bool = True


@dataclass
class World:
    walls: set[tuple[int, int]]
    rooms: dict[tuple[int, int], str]  # cell -> "kitchen"
    objects: dict[str, WorldObject] = field(default_factory=dict)
    tick: int = 0
```

`hidden_attrs` carries the language-only constraints behind the `rho` knob. They never come from perception; they appear only in task text. Optional: give the drone a `scan` tool that reveals some hidden attributes.

Additions still needed: `hands_required` per object, preconditions and effects per primitive (for invalid-call detection), and `prob_failure` per primitive and agent.

## 5. Perception (deterministic)

An object is visible to an agent if it is within range, inside the view cone, has line of sight (no wall on the Bresenham line), and is not inside a closed container.

```python
import math


def has_line_of_sight(world: World, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """Bresenham walk from start to end; blocked if any intermediate cell is a wall."""
    (x0, y0), (x1, y1) = start, end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    step_x, step_y = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    error = dx + dy
    while (x0, y0) != (x1, y1):
        if (x0, y0) != start and (x0, y0) in world.walls:
            return False
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y
    return True


def visible_object_ids(
    world: World,
    cell: tuple[int, int],
    heading_deg: float,
    view_range: float,
    view_angle_deg: float,
) -> set[str]:
    visible = set()
    for obj in world.objects.values():
        dx, dy = obj.cell[0] - cell[0], obj.cell[1] - cell[1]
        if math.hypot(dx, dy) > view_range:
            continue
        bearing = math.degrees(math.atan2(dy, dx))
        if abs((bearing - heading_deg + 180) % 360 - 180) > view_angle_deg / 2:
            continue
        if not has_line_of_sight(world, cell, obj.cell):
            continue
        container = world.objects.get(obj.parent_id) if obj.parent_relation == "in" else None
        if container is not None and not container.is_open:
            continue
        visible.add(obj.obj_id)
    return visible
```

Walls used for line of sight are the same walls used for radio attenuation.

### Relations (fixed small vocabulary)

Among visible objects only: `on` / `in` (read from the parent pointer, always exact), `in_room`, `held_by`, and `next_to` (adjacent cells with no parent link). A held object's cell follows its holder, so it is reported as `held_by` and never with a stale cell. Held objects get no `next_to` relations. Do not emit all pairwise "near" relations; that is n^2 prompt noise. Output is sorted so identical states produce identical text (this keeps the response cache effective).

```python
def observed_relations(world: World, visible: set[str]) -> list[tuple[str, str, str]]:
    relations = []
    for obj_id in sorted(visible):
        obj = world.objects[obj_id]
        if obj.parent_id is not None:
            relations.append((obj_id, obj.parent_relation, obj.parent_id))
        else:
            relations.append((obj_id, "in_room", world.rooms[obj.cell]))
    ordered = sorted(visible)
    for index, first_id in enumerate(ordered):
        for second_id in ordered[index + 1 :]:
            first, second = world.objects[first_id], world.objects[second_id]
            adjacent = (
                max(abs(first.cell[0] - second.cell[0]), abs(first.cell[1] - second.cell[1])) == 1
            )
            linked = first.parent_id == second_id or second.parent_id == first_id
            if adjacent and not linked:
                relations.append((first_id, "next_to", second_id))
    return relations
```

### Distance

Distance for bidding is **shortest-path length in steps** (grid cells; BFS on the 4-connected grid, cached per target cell), not straight-line distance. Agents have per-agent speed, so travel time is `travel_ticks = max(1, ceil(steps / speed))`, derived from steps and never measured separately. The same function (`PathPlanner.steps` in `sim/pathing.py`) feeds the oracle, numeric CBBA utilities, and the text shown to LLMs ("plate_2: 6 steps away"), so no condition gets better distance information. Give LLMs step counts, not raw coordinates.

Furniture is impassable, so "distance to an object" means distance to the nearest free cell within one cell of it, diagonals included. Interaction primitives use the same adjacency (Chebyshev distance at most 1). Agents do not block each other in M1.

## 6. Belief graph (what the agent thinks is true)

Each agent owns its belief dictionary. Update rule, every tick:

1. **Visible objects:** overwrite with fresh relations and attributes; set `last_seen = tick`.
2. **Negative evidence:** if the agent can see the cell where it believes an object is, but the object is not visible, mark the entry `missing`. *Exception:* an object believed to be `in` a container that is now closed is not marked missing, because a closed container hides its contents. (The reference snippet below predates this fix; `sim/beliefs.py` has it.)
3. **Everything else:** leave it alone; it goes stale but keeps its `last_seen`.
4. **Gossip from teammates:** merge with `source = "told"` and the teammate's timestamp; the newer timestamp wins. Communication loss therefore degrades what agents *know*, not only what they have claimed.

```python
@dataclass
class Belief:
    obj_type: str
    relations: list[tuple[str, str, str]]
    visual_attrs: dict[str, str]
    believed_cell: tuple[int, int]
    last_seen: int
    source: str  # "seen" | "told"
    missing: bool = False


def update_beliefs(
    beliefs: dict[str, Belief],
    world: World,
    visible: set[str],
    visible_cells: set[tuple[int, int]],
    relations: list[tuple[str, str, str]],
) -> None:
    for obj_id in visible:
        obj = world.objects[obj_id]
        own_relations = [rel for rel in relations if rel[0] == obj_id]
        beliefs[obj_id] = Belief(
            obj.obj_type, own_relations, dict(obj.visual_attrs), obj.cell, world.tick, "seen"
        )
    for obj_id, belief in beliefs.items():
        if obj_id not in visible and belief.believed_cell in visible_cells:
            belief.missing = True
```

`visible_cells` is computed with the same range, cone, and line-of-sight test applied to cells. (In code this is `cell_visible`, evaluated only at believed cells.)

**Static furniture is known from the map at tick 0** (belief `source = "map"`). Movable objects must be perceived. Belief gossip (`merge_told`) is implemented and tested but is not wired into the loop yet. It arrives with ledger gossip in M2.

### Prompt serialization

```
TICK 140. You are arm_a at kitchen (3,4), facing east. Holding: nothing.
SEEN NOW:
  plate_2 on counter_1 (dirty) - 2 steps
  pot_1 next_to plate_2 - 3 steps
  dishwasher_1 in_room kitchen (closed) - 5 steps
REMEMBERED (not in view):
  mug_3 on table_1 - last seen 38 ticks ago - 11 steps
  plate_4 - MISSING (not at last known spot, last seen 12 ticks ago)
TOLD BY TEAMMATES:
  cup_7 on shelf_2 (from drone_b, 20 ticks ago)
```

Showing staleness is the default; it can be ablated.

## 7. Tools and validation

Tools: `move_to`, `pick`, `place`, `open`, `close`, `scan`, `handoff`, `report`, plus allocation tools (`submit_utilities`, `bid`, `claim`, `release`, `send_message`).

- **At call time (belief check):** the target must exist in the agent's beliefs. Calling `pick("plate_9")` for an object never seen is an invalid call, logged as a **hallucination**.
- **At execution (ground-truth check, tick `t + L`):** the object must still be there and the gripper must have free hands. If this fails, the failure is returned as text ("pick failed: plate not at sink") and logged as a **stale belief**.
- Skill outcomes use per-agent success probabilities; failures return text to the LLM.

Outcome categories, logged in `actions.jsonl`:

| Outcome | Meaning |
|---|---|
| `hallucination` | Call-time: named an object not in the agent's beliefs |
| `invalid_call` | Unknown tool, wrong arguments, or the agent's *own* state forbids it (not enough free hands, not holding the object, no scanner) |
| `stale_belief` | Apply-time: the world no longer matches the belief (object moved or held by someone else, container closed or already open, teammate not adjacent) |
| `skill_failure` | Preconditions held but the success roll failed (a failed `place` drops the object by the drop rule) |

Self-state failures count as `invalid_call`, not `stale_belief`, so the stale-belief rate measures only world drift. That drift is what communication loss should worsen.

## 8. Communication channel

```python
def delivery_prob(
    dist: float, walls_crossed: int, d0: float, alpha: float, wall_penalty: float
) -> float:
    return (1.0 / (1.0 + (dist / d0) ** alpha)) * (wall_penalty**walls_crossed)
```

Final delivery probability also includes the swept base loss `p`. Optional latency `d` in ticks. Range `r` is a hard cutoff that creates partitions. Sample drops with a seeded RNG and log every message with its reason for loss (`loss`, `range`, `offline`).

As built: delivery is `delivery_prob * (1 - p)`. Setting `d0: null` disables distance falloff, and perfect communication is `p = 0`, `r = null`, `d0 = null`, `wall_penalty = 1`. A message is delivered one hop tick plus `d` after it is sent. Range `r` is a single swept value for the whole team, not a capability-card field, so the `r` sweep is not confounded with per-agent radios. Distance for radio is Euclidean. BFS steps are for travel only.

## 9. Ledger and gossip

Per agent: `task_id -> {status, owner, bid, version, lease_expiry}`.

- Status is monotone: unclaimed, claimed, in progress, done. Done always wins on merge.
- Merge rule: higher status wins; ties go to the higher version/bid, then the lower agent ID.
- Gossip on contact (anti-entropy), broadcast-on-change.
- A lease expires if the owner has not renewed it; teammates may then reclaim the task. Track lease false positives (healthy owner, partitioned).
- Compound tasks derive their status from their children.
- The action-history narrative is compacted after `N` entries (summarized into the prompt). The ledger itself is never compacted.

## 10. Churn

`k` agents crash at a random tick and rejoin after `delta` ticks, in two variants: **memory retained** and **memory wiped** (see open question 7 in `decisions.md`). Open-team variant: a new agent joins mid-mission.

## 11. Conditions

All conditions use one agent interface; only the coordination logic differs. See the table in `CLAUDE.md`. C0 is a central solver (e.g., CP-SAT) with ground-truth feasibility. C1 is natural-language negotiation with the ledger. C2 is CBBA with travel-cost utilities. C3 and C4 use LLM utilities with CBBA and MURDOCH respectively.

## 12. Logging

Append-only JSONL per episode. Every row records both `tick` and `wall_time`; only `tick` appears in results.

| File | One row per | Key fields |
|---|---|---|
| `events.jsonl` | Task event | tick, agent, task, transition (claimed, started, done, failed, reclaimed; M1 logs done, reported, crashed, rejoined) |
| `actions.jsonl` | Tool-call outcome | tick, agent, call, outcome (see §7), text |
| `llm_calls.jsonl` | LLM call | tick issued, tick applied, `L`, prompt, raw response, parsed tool calls, tokens in/out, wall seconds |
| `messages.jsonl` | Message | tick sent, sender, receiver, type (bid, gossip, negotiation), delivered or dropped, reason |
| `ledgers.jsonl` | Agent every N ticks | each agent's ledger copy, for divergence vs ground truth |
| `beliefs.jsonl` | Belief change | added, updated, marked missing, merged from teammate; plus a full snapshot every N ticks |
| `state.jsonl` | Tick (or every N + deltas) | positions, object locations, held items |
| `run.json` | Episode | full config, seed, git commit, model file hash, latency parameters, server settings |

Metrics are computed from logs afterwards, never inside the sim. The pixel-art viewer is a replay tool that reads these logs, so rendering never slows experiments.

## 13. Model serving and reproducibility

- One server process (llama.cpp `llama-server` with parallel slots, or vLLM); each agent is its own prompt history and KV cache. Concurrent requests are batched.
- Temperature 0 or a fixed seed. vLLM batching can change outputs slightly even at temperature 0; llama.cpp with a single instance is more stable.
- Response cache keyed by a hash of the full prompt, so an episode re-runs identically and old runs can be re-scored without GPU time.
- Pin model file, quantization, server version, and prompt templates in the repo.
- Hardware: laptop RTX 4070 (8 GB) for development and pilots; UMBC cluster Slurm job arrays for sweeps.
- Report results with a frontier-model ceiling check on a small subset.

## 14. Code layout

Built in M1 (LLM-free sim core):

```
sim/            config.py, world.py, pathing.py, perception.py, beliefs.py, channel.py,
                agent.py (Policy seam), primitives.py, scenario.py, loop.py, logger.py
allocation/     scripted.py (test scaffolding, not a condition)
llm/            latency.py
configs/        m1_smoke.yaml, m1_smoke_p05.yaml, maps/*.txt, missions/*.yaml
experiments/    run_single.py
tests/          pathing, perception, beliefs, channel, primitives, loop
```

Still planned: `sim/ledger.py` (M2); `allocation/cbba.py` and `oracle.py` (M3); `llm/client.py`, `prompts/`, and `allocation/utility.py` (M4); `allocation/murdoch.py` (M5); `experiments/launch_sweep.sh` and `analyze.py` (M6).

Map format (`configs/maps/*.txt`): legend lines `<char>: <room>`, then `---`, then the grid, where `#` is a wall. The mission YAML holds furniture and objects (cells, parents, visual and hidden attributes, hands), agents (capability card and pose), and tasks (description, goal conditions, `after` precedence, `language_only`). Quote `"on"` in YAML, because bare `on` parses as a boolean. The loader rejects unknown goal relations.
