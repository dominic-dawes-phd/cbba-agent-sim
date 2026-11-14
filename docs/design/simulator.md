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

As built for the LLM conditions (v5.1.00, `sim/loop.py`):
- **Coordination step.** Before decisions, every policy with an `on_tick` hook (`TickingPolicy`) runs it, in code, with the agent's view: beliefs, known tasks, the code messages that reached it, its history, and its teammates' cells and online flags. It returns broadcasts, ledger rows to log, dialogue events, any LLM calls it made (C1's turns), and whether a decision is due. Messages of kind `ledger`, `cbba`, and `dialogue` go only to this step and never make a decision point by arriving. A change of assignment does.
- **Concurrent decisions.** All decisions due on a tick are sent at once from a thread pool, and the results are applied in agent order, so thread timing never changes an outcome.
- **Several calls per decision.** A decision may make several LLM calls (C3's utility query, then the action). Their latencies add, and each call is logged as its own row.
- **A reply that does not parse** is an `invalid_call` on `reply`, shown at the next decision. It costs `L` like any rejected response.
- **Holding still.** An LLM policy with no task and nothing to decide (C1 waiting for a plan) returns `wait` with no LLM call, so it costs no tokens and no `L`.

## 4. World ground truth

Objects live in a scene-graph tree (room, furniture, object). `on` and `in` are stored as a parent pointer and updated whenever something is placed, picked up, or dropped. Drops are a time penalty (§9): a non-fragile object stays in hand after a short pause, and plates and cups (`BREAKABLE_TYPES`) shatter on the agent's own tile and are cleaned up by reflex. Only a crash leaves a held object on the floor.

As built, furniture may span a rectangle of cells (`footprint`). Each cell is a slot holding at most `slot_capacity` items, and a movable object in or on furniture has its `cell` set to its slot. Trashcans use a whole-object `capacity` instead. Dishwashers have `washing_time` and a `cycle` state. Workspace reservations live in `World.reservations` (cell -> holder, furniture). Agents occupy cells and block each other; see `decisions.md` (Oct 25, fleet physics).

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

`hidden_attrs` carries facts that never come from perception; only the `scan` tool reveals them (generated missions have none since the `rho` knob was removed on Nov 10, 2026).

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

Distance for bidding is **shortest-path length in steps** (grid cells; BFS on the 4-connected grid, cached per target cell), not straight-line distance. Agents have per-agent speed, so travel time is `travel_ticks = max(1, ceil(steps / speed))`, derived from steps and never measured separately. The same function (`PathPlanner.steps_to_any` in `sim/pathing.py`, called without `avoid`) feeds the oracle, numeric CBBA utilities, and the text shown to LLMs ("plate_2: 6 steps away"), so no condition gets better distance information. Give LLMs step counts, not raw coordinates.

Furniture is impassable, so "distance to an object" means distance to the nearest free cell within one cell of any of its cells, diagonals included. Interaction primitives use the same adjacency (Chebyshev distance at most 1). Agents block each other.

### Movement (as built, `sim/motion.py`)

All agents in a `move_to` move together each tick, in rounds (one round per cell of speed).

- **Proposals.** In each round, every mover not yet next to its target proposes one cell: the next cell on its cached static shortest path. If a body or another holder's reservation blocks that cell, it runs a fresh BFS that avoids them and takes the first cell of the detour (local replanning). It does not predict others' moves; there is no space-time reservation.
- **No swaps.** Cells occupied at the start of a round are off-limits, so agents never follow into a vacated cell and never swap.
- **Conflicts.** When several movers propose the same cell, a seeded coin flip (`seed:motion`) picks the winner; the losers wait and count a `move_conflict`. There is no id priority.
- **Arrival.** A mover stops as soon as it stands next to its target (Chebyshev 1); unused movement is not carried over, and the next call starts on the following tick.
- **Patience.** A tick counts as stuck unless the agent got closer (fewer static steps than its best so far). While stuck:
  - An idle teammate on its path steps aside (`yields`).
  - Of two movers blocking each other head-on, a seeded coin flip picks one to step aside.
  - After 6 stuck ticks the move fails with `path ... blocked by X`.
- **Distances unchanged.** All of this changes how many ticks a move takes, never the static step counts used for bidding and prompts (hard rule 8).
- **Path shape.** Moves are 4-connected, and ties between equally short paths break in a fixed neighbour order (E, W, S, N).
- **Detours persist.** Once a mover leaves the static path, it follows the obstacle-aware path until its static distance improves. Otherwise a detour that first moves away steps straight back and oscillates forever.
- **Randomized backoff.** A mover with no progress for `BACKOFF_AFTER_STUCK_TICKS` (2) sits out each movement round with probability `BACKOFF_SIT_OUT_PROB` (0.5), from the seeded motion stream. This breaks the symmetry of two movers undoing each other's moves.

### Spawns

`validate_spawns` rejects a start cell that is a wall, off the map, or furniture, that is shared, or from which some furniture is unreachable. The episode config's `spawn.mode` picks where agents start:
- **`mission`:** the start cells the mission lists. Generated missions put each agent next to its own dock. The hand-written kitchens use this mode.
- **`uniform`:** a seeded uniform draw (`draw_spawns`, RNG `<seed>:spawn`) over free floor connected to the mission's first start cell. Each drawn cell is kept only if it is at least `min_separation` static steps from those already kept, and each agent gets a random heading from E/S/W/N. Every condition run at a seed starts from the same cells. Docks stay fixed furniture for parking and, later, charging. Generated houses use this mode (`min_separation: 4`).

### Task visibility, exploration, and coverage sharing (v2.0.00)

**Task visibility** (`task_visibility`):
- **`posted`:** every mission task is logged `posted` at tick 0 and known to the team.
- **`discovered`** (generated houses and paper runs): mission tasks are logged `exists` (ground truth, used only by metrics), and no agent knows any task at tick 0, only the floor plan.

**Revealing a task.** After perception, a hidden mission task becomes known to an agent that sees one of its **reveal objects** this tick. Seeing means own sight; map knowledge does not count.
- **Which objects:** a mission entry's `reveal_on` list, or by default its movable goal objects.
- **Explicit cases:** unpack and trash tasks set `reveal_on` to the dishwasher or trashcan, because their items are inside.
- **Logging:** the first finder in the episode logs `discovered`.
- **Announcing:** the finder broadcasts `task_found` and decides at once if it can be interrupted. A busy agent keeps working and only records the task, so tasks are found mid-task.

**No duplicate tasks.** Ids are mission ids. Reveal runs before the perception triggers, and triggers skip objects any mission task covers, so a seen dirty dish never becomes a second task. Two agents that find a task separately, even across a partition, hold the same id. The duplication then shows up as duplicated claims and work, which the ledger merge resolves.

**Coverage maps** (`sim/exploration.py`). Each agent keeps two maps of cell to tick in view:
- **Team map (`searched`):** own sightings plus teammates' coverage messages.
- **Own map (`own_searched`):** own sight only.

With `exploration.search_ttl: null`, a cell stays searched forever and the map records first sightings. A finite ttl expires cells, which is how regenerating tasks would be found again.

**Sharing coverage.**
- **When:** every `share_every` ticks, each online agent broadcasts the cells it newly recorded (`coverage` message, `x,y,tick` entries) over the lossy channel.
- **Merge:** receivers merge by latest sighting.
- **No decisions:** coverage messages are perception-layer map merging, the same for every condition, and **never create a decision point** (hard rule 2).
- **Separate accounting:** the analyzer counts them separately from coordination messages.

**`explore` and `search`** are S1 skills shared by every condition, with no arguments.
- **Where they go:** each walks to the nearest frontier cell by BFS over floor (the static metric), retargeting every tick before movement (`retarget_explorers`).
  - **`explore`** uses the team map and ends when nothing is unsearched. It is for finding new tasks.
  - **`search`** uses the agent's own map. It is for finding an object it needs, because coverage messages say where a teammate looked, not what it saw. A search started when everything has been seen begins a new round with a cleared own map, because objects move: dishes get carried off, and a crashed agent leaves what it held on the floor. A search that has swept its round ends with "searched everywhere you can reach" (v5.1.00), so the agent gets a decision point; before, it swept forever when the object was inside a closed container, and in C2/C3 no message interrupted it.
  - **A frontier** is a floor cell not yet seen, or one next to a furniture cell not yet seen (v4.0.00). A floor cell glimpsed at the edge of view range says nothing about the shelf just beyond it, and objects lie on furniture. Before this, an agent could count its map complete while a dish it needed sat unseen on a shelf, and never search again.
- **Hands:** both need empty hands.
- **Interruption:** both are interruptible like `wait`. Seeing a new object, seeing again an object believed missing, finding a task, or a message ends them at a decision.
- **Prompt:** it shows the steps to both frontiers (`UNSEARCHED: ...`).
- **Algorithm:** nearest-frontier exploration is the standard baseline (Yamauchi 1997; coordinated multi-robot variant, Burgard et al. 2005). **Neither is in `references.bib` yet; verify before citing.**

**Coded executor:**
- An idle agent explores while unsearched floor remains, then parks at its dock.
- A step that picks up an unseen object searches.
- It skips an item that already sits where the task wants it, rather than taking it back out.
- It does not leave a task reported done by a teammate while holding one of that task's items.

### Battery and scheduled disconnect (planned; config scaffold only)

`EpisodeConfig.battery` (`enabled`, `capacity`, `drain_idle`, `drain_move`, `drain_carry`, `drain_manipulate`, `charge_rate`, `low_threshold`) is parsed from every episode config. `enabled: true` is rejected until the model is built. The design:
- **Drain:** charge drains per tick by activity (idle, moving, carrying, manipulating). An agent charges only at its own dock (`dock_<id>`), at `charge_rate`. It can still talk while charging but cannot act.
- **Visibility:** an agent knows its own charge exactly (proprioception, not perception), and its capability card shows it. Teammates learn it only from messages.
- **Low battery:** dropping below `low_threshold x capacity` becomes a decision point. The agent can finish the current step, release its task, or go charge.
- **Allocation:** a task whose estimated energy exceeds the remaining charge is infeasible for that agent. Utilities and bids, including the oracle's, must respect this.
- **Scheduled disconnect:** an *announced* departure, such as a charging trip or a maintenance window. The agent releases its tasks gracefully, unlike crash churn, which gives no warning. Comparing the two separates the cost of missing information from the cost of a missing body.
- **Planned metrics:** charging ticks, tasks abandoned for battery, energy per completed task, and lease false positives during charging.

### Continuous task arrivals (planned)

New dirty dishes, groceries, and stray items appear at random valid surfaces at a Poisson rate during the episode, as in Liu et al. 2013 (whose bib entry still needs the user's OK). The mission then ends at `max_ticks`, and results become throughput and tasks outstanding over time. This pairs with a finite `search_ttl`, so searched areas expire and get re-searched.

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

**Static furniture is known from the map at tick 0** (belief `source = "map"`). Movable objects must be perceived, and in discovered mode so must tasks (§5). Belief gossip (`merge_told`) is implemented and tested but is not wired into the loop yet; only coverage (where an agent looked) is shared so far. Belief gossip arrives with ledger gossip in M2.

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

Tools: `move_to`, `explore`, `search`, `pick`, `place`, `open`, `close`, `scan`, `handoff`, `report`, `start`, `lock`, `unlock`, `tie_bag`, plus allocation tools. Each condition offers its own subset (`conditions.md`):
- **G:** `report` and `release`, which broadcast a task id and log an event; code issues the claim.
- **C2, C3:** the ledger tools `mark_done` and `give_up` (one tick each, instant). They change the agent's ledger copy, which gossips the change.
- **C1:** `talk_to_team`, which asks for a dialogue round.

The LLM never bids: C3's utilities come from a separate call, not a tool. The descriptions shown to the LLM are in `configs/prompts/tools.yaml`.

- **At call time (belief check):** the target must exist in the agent's beliefs. Calling `pick("plate_9")` for an object never seen is an invalid call, logged as a **hallucination**.
- **At execution (ground-truth check, tick `t + L`):** the object must still be there and the gripper must have free hands. If this fails, the failure is returned as text ("pick failed: plate not at sink") and logged as a **stale belief**.
- Skill outcomes use per-agent success probabilities; failures return text to the LLM.

Outcome categories, logged in the `actions` stream:

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
- Household tasks are templates (`configs/task_templates.yaml`: `clean_plate`, `unpack_dishwasher`, `pack_groceries`, `take_out_trash`, ...) that a mission instantiates by `kind`. A template unrolls at load time into a flat primitive sequence plus ground-truth goals, so a task is one orchestrated series of primitives, like a generalist policy told "unpack the dishwasher". Done is still a ground-truth goal check, so a task orphaned by a crashed owner can be resumed from whatever state the world is in.
- `lock(target)` reserves the *workspace*: the target's cells within reach plus the free floor around them. Other agents cannot work on reserved slots and route around reserved floor. Released on `unlock`, holder crash, or (for an appliance) when the cycle that took it over ends. A blocked primitive fails as `stale_belief`. This is physical exclusion, not a CBBA guarantee (hard rule 10).
- Dishwashers have `washing_time` (60 ticks): `start` runs the cycle in the background and hands the reservation to it, `open` is refused mid-cycle, and a finished cycle cleans the contents and frees the workspace. A running cycle's countdown (`cycle_left`) is perceivable, so an agent that saw it knows when the cycle ends without looking again. Trashcans have a `capacity`; every item counts 1, `fill` is a perceivable gauge, and `tie_bag` collapses the contents into one carriable bag.
- **Drops are a time penalty, not a coordination event** (`faults.drop_prob`, once per pick, place, and move while holding).
  - **Non-fragile:** the gripper drops the object on the agent's own tile and picks it up again after `DROP_RECOVERY_TICKS` (2). Ground truth keeps the object in that gripper's custody for those ticks, so no teammate can take it. The skill carries on, with no failure and no decision.
  - **Fragile (plates, cups):** the skill fails and a reflex runs in every condition, with no LLM call. It takes `SHARD_PICKUP_TICKS` (3), a walk to the nearest trashcan, and `SHARD_DISPOSE_TICKS` (1). The agent then walks back to the drop tile (`return_to_task`), and only there gets its decision point, so the next step proceeds from its task area. Shards count toward the can's `fill`. Neither `clean_up` nor `return_to_task` is an LLM tool, and neither gives up on a blocked path.
  - **The trashcan comes from ground truth** (`nearest_trashcan`), by static steps. This is a deliberate cheat, so a fragile drop can never turn into a search. The choice reaches no belief, so the agent learns nothing from it. With no reachable trashcan on the map, the shards stay on the floor and the agent gets its decision point at once.
  - **The task stays with that agent.** Only the broken item is lost: its goals are removed and the rest of the task continues, wash cycle included. A task with no movable item left is voided (finished for precedence and episode end, never logged `done`).
  - **A crash** leaves any held object on the floor. A fragile one shatters with no reflex: the shards stay as debris with no task.
  - **Stale beliefs:** triggers skip broken objects, since a task for a dish that no longer exists could never finish.
  - **Logged:** `dropped` and `broken` events.
- **Wash goals are per dish** (v4.0.00): `[dish, washed, yes]` holds once a cycle that held the dish has ended, in any dishwasher. A wash task's `{appliance}` is the nearest machine, a default; an agent may load another.
- **A cycle started mid-load splits the task** (`handle_cycle_starts`, every condition). When anyone starts a dishwasher that holds some, but not all, of a wash task's dishes:
  - **The task** keeps the loaded dishes, so it is done when the cycle ends. Its steps are emptied, since nothing is left to do.
  - **A new task** `<task>_r<k>` gets the remaining dishes and the full recipe. It is logged `split` and `exists`. Every agent holding one of its dishes, and the starter, finds it (`discovered`, `task_found`), so the remaining dishes become an ordinary task for the team's allocator.
  - **A dish in hand** goes back to the slot it was picked up from, by a no-LLM `return_item` reflex, before its holder decides anything.
- **A queued pick is cancelled once perception shows its item has moved** (every condition). When a decision queues a `pick`, the agent's belief of where the item was is recorded. If a later sighting shows it elsewhere, held by someone, or gone, the walk and the queue stop, with `stale_belief` "<item> is no longer on <place>". The pick itself also fails with that text if ground truth no longer matches (hard rule 4's re-check), for a move the agent never saw.
- **Washed dishes are unloaded as a follow-up task** (v5.0.00, every condition). An agent that sees clean dishes inside an open dishwasher (`sim/triggers.py`) finds `unload_<dishwasher>_<first dish>`, an `unpack_dishwasher` task into a cupboard in the dishwasher's room. It is announced like any found task (so CBBA, MURDOCH, or C1's dialogue allocates it), and the finder carries on with its own work. The unload template locks the dishwasher for its whole duration, so a loader that arrives during it waits until the dishes are out before it can load or start a cycle; this idling is intended. A dishwasher nobody reopens keeps its washed dishes, and no unload arises.
- **Proprioception:** an agent always knows what it holds and where it let go of something. A belief that it still holds an object it put out of sight (shards in a closed trashcan) is corrected from ground truth (`sync_hands`).
- New tasks are *discovered* (`sim/triggers.py`): after perception, each agent's beliefs are checked for dirty dishes outside the dishwasher, groceries on the entry table, clean dishes on the drying rack, and a trashcan more than half full. Ids derive from the triggering object. The finder logs `discovered` and broadcasts `task_found` on the lossy channel; allocation is the coordination layer's job.
- The action-history narrative is compacted after `N` entries (summarized into the prompt). The ledger itself is never compacted.

**Ledger as built** (v5.1.00, `sim/ledger.py`, used by C2 and C3 through `allocation/cbba.py`):
- **Merge order:** done is absorbing; otherwise higher status, then higher version, then higher bid, then the lower agent id wins. The merge is commutative, associative, and idempotent, so copies converge in any delivery order.
- **Versions:** every local change (claim, start, renew, done, release) bumps the row's version, and a release returns the row to unclaimed at a higher version.
- **Leases:** `lease_ticks` (60), renewed every `lease_ticks / 3`. A lapsed lease held by a teammate reopens the task, in the CBBA state too. Lease false positives are not yet measured.
- **Gossip:** changed rows go out as a `ledger` message on the tick they change, and the whole copy every `gossip_every` (10) ticks (anti-entropy). Rows merged from teammates are not re-broadcast, which would echo, but they are logged.
- **Compaction:** history compaction is deterministic (`llm/prompts.py` `history_lines`). The last `history_keep` (8) actions are shown verbatim; the rest become one line counting them, their failures, and the tasks reported.

## 10. Churn

`k` agents crash at a random tick and rejoin after `delta` ticks, in two variants: **memory retained** and **memory wiped** (see open question 7 in `decisions.md`). Open-team variant: a new agent joins mid-mission.

## 11. Conditions

All conditions use one agent interface; only the coordination logic differs. **`conditions.md` is the canonical definition** (layer model, allowed LLM tools, no-bleed rules). In brief:
- C0 is a central heuristic solver with ground-truth feasibility.
- C1 is RoCo-style lockstep dialogue per communication component, with no ledger.
- C2 is CBBA with travel-cost utilities.
- C3 and C4 use LLM utilities with CBBA and MURDOCH respectively.

The S2 executor (`coded | llm`) is a separate choice from the allocator.

C1's round protocol is configured by the `dialogue:` block: `M` = `turns_per_member` x members, `K` = `replans`, plus `retry_ticks` and `message_chars` (`conditions.md`, C1 as built). CBBA's settings are in the `cbba:` block (`bundle_size`, `discount`, `gossip_every`, `lease_ticks`, `give_up_ticks`).

**C0 as built** (`allocation/oracle.py`, `allocation/oracle_cpsat.py`, `condition: oracle`):
- **Planning model:** per (agent, task) durations from `PRIMITIVE_TICKS` and static BFS steps (hard rule 8), transitions between tasks, precedence, and exclusive furniture (lock spans, and each wash from its first loading trip to the end of its cycle, so no cycle can split it). A task completes when its last goal-making step ends. It has no bodies or doorway queues.
- **Unloads (v5.0.00):** between consecutive washes on a shared dishwasher, C0 plans an unload of the earlier batch (any agent; `after` its wash), and gates the next wash's first hold on that machine behind it (`PlanProblem.gates`, enforced in `schedule` and per presence literal in CP-SAT). Every order of the washes sharing a machine is solved (up to `ORACLE_MAX_MACHINE_ORDERS`); the best plan is kept, its bound is the minimum over the orders, and it is OPTIMAL only if every order is. The unload ids match the trigger's, so the replay finds the planned task when it sees the dishes.
- **Solvers:** a seeded heuristic, then CP-SAT with one circuit per agent, optional intervals, and a redundant capacity constraint per furniture. Limits are search counts and deterministic time, never a clock.
- **Replay:** `OraclePolicy` follows the plan with the coded executor at `L = 0`, with ground-truth beliefs (`omniscient_beliefs`) and a release tick per task. Replays drift, so a gated wash does not touch its dishwasher until the unload before it is reported, and a busy machine is waited for rather than avoided. It never explores but may search for an object a teammate moved out of sight.
- **Logged:** `oracle_plan.json` with assignment, planned, bound, method, status, and release ticks.
- **Config:** the `oracle:` block; `cpsat` needs the `[solver]` extra.

## 12. Logging

Append-only gzipped JSONL per episode (`<stream>.jsonl.gz`, read with `sim.logger.read_stream`). Every row records both `tick` and `wall_time`; only `tick` appears in results. A house run is under 1 MB (v4.0.00; 7-17 MB before, when full belief graphs were copied at every snapshot).

| File | One row per | Key fields |
|---|---|---|
| `events` | Task event | tick, agent, task, transition (planned: started, failed, reclaimed; logged now: posted, exists (a hidden mission task, discovered mode), discovered, claimed, released, reported, done, voided, dropped, broken, split (with child and items), crashed, rejoined) |
| `actions` | Tool-call outcome | tick (finished), start_tick, agent, task (from the policy response), call, outcome (see §7), text |
| `llm_calls` | LLM call | tick issued, tick applied, `L`, purpose (`act`, `utility`, `dialogue`), prompt, raw response, parsed tool calls, rejected, tokens in/out, cached, wall seconds |
| `messages` | Message | tick sent, sender, receiver, kind (task_found, claim, release, report, coverage, ledger, cbba, dialogue), delivered or dropped, reason |
| `ledgers` | Change to an agent's ledger copy | agent, and every row that changed in its copy that tick (its own changes and merged ones), for divergence vs ground truth |
| `dialogues` | Round event (C1) | round id and members at opening; per turn the speaker, text, PROCEED/EXECUTE, plan, and the plan check's problem. One transcript per round, so `k` partitions give `k` concurrent transcripts. Who adopted a plan shows in `actions` (task tags) |
| `beliefs` | Belief change | each agent's full graph at tick 0 (`initial`) and the end (`final`); in between every change (added, updated, missing) with the new belief. An unchanged sighting only refreshes `last_seen` and is not logged (each prompt shows it), so the changes rebuild any graph except `last_seen` |
| `poses` | Tick | every agent's cell, heading, held objects, online flag |
| `state` | Every `snapshot_every` ticks (50) and the last tick | object locations, held items, reservations, team `coverage` (share of floor searched), agent counters and `searched` counts |
| `oracle_plan.json` | Oracle episode | assignment, planned makespan, proven bound, method, solver status, release ticks |
| `run.json` | Episode | full config, seed, `sim_version` (the `docs/CHANGELOG.md` version), git commit and dirty flag, model file hash, latency parameters, server settings |

Metrics are computed from logs afterwards, never inside the sim. The pixel-art viewer is a replay tool that reads these logs, so rendering never slows experiments.

## 13. Model serving and reproducibility

- One server process (llama.cpp `llama-server` with parallel slots, or vLLM); each agent is its own prompt history and KV cache. Concurrent requests are batched.
- Temperature 0 or a fixed seed. vLLM batching can change outputs slightly even at temperature 0; llama.cpp with a single instance is more stable.
- Response cache keyed by a hash of the full prompt, so an episode re-runs identically and old runs can be re-scored without GPU time.
- Pin model file, quantization, server version, and prompt templates in the repo.
- As built: `llm_setup.md` pins llama.cpp b12211 and `Qwen3-8B-Q4_K_M`, the server flags, and the client settings. The response cache lives in `llm.cache_dir`, keyed by sha256 of the model hash and the request body.
- Hardware: laptop RTX 4070 (8 GB) for development and pilots; UMBC cluster Slurm job arrays for sweeps.
- Report results with a frontier-model ceiling check on a small subset.

## 14. Code layout

Built in M1 (LLM-free sim core):

```
sim/            config.py, world.py, pathing.py, perception.py, beliefs.py, channel.py,
                agent.py (Policy seam), primitives.py, scenario.py, loop.py, logger.py,
                motion.py, triggers.py, exploration.py (coverage maps, frontier, sharing)
allocation/     scripted.py (test scaffolding, not a condition), greedy.py (G),
                oracle.py and oracle_cpsat.py (C0), llm_policy.py (LLM executor + allocator),
                cbba.py (C2, C3), utility.py (C3's utility call), dialogue.py (C1)
sim/ledger.py   the replicated ledger (M2)
llm/            latency.py, client.py (OpenAI-compatible chat), cache.py, prompts.py
configs/prompts system prompts and tool descriptions shown to the LLM
configs/        m1_smoke.yaml, m1_smoke_p05.yaml, maps/*.txt, missions/*.yaml,
                dev/*.yaml (dev figure specs), sweeps/*.yaml (experiment specs),
                generator/, generated/
experiments/    run_single.py, sweep.py, analyze.py, plot.py, plot_episode.py, dev_refresh.py,
                generate_mission.py, headroom.py
runs/           dev/<env>/{logs,figures}/ (overwritten on each refresh),
                <experiment>/<env>/<cell>/<YYYYMMDD-HHMMSS>/seed_<n>/ (kept)
tests/          pathing, perception, beliefs, channel, primitives, loop
```

Still planned: `allocation/murdoch.py` (C4, M5), the L and R baselines, the C1+ledger and C3-noledger ablations, and `experiments/launch_sweep.sh` for the cluster (M6).

Map format (`configs/maps/*.txt`): legend lines `<char>: <room>`, then `---`, then the grid, where `#` is a wall. The mission YAML holds furniture and objects (cells, parents, visual and hidden attributes, hands), agents (capability card and pose), and tasks (description, goal conditions, `after` precedence, `language_only`). Quote `"on"` in YAML, because bare `on` parses as a boolean. The loader rejects unknown goal relations.
