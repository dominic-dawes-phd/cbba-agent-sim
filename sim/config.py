"""Episode configuration and fixed simulator parameters.

Settings that experiments sweep (latency rates, channel, churn) are read from one YAML file per
condition or sweep. Parameters that no experiment varies are module constants here.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Ticks a primitive takes once started. move_to takes travel_ticks(steps, speed) and wait takes
# its argument, so neither appears here.
PRIMITIVE_TICKS: dict[str, int] = {
    "pick": 2,
    "place": 2,
    "open": 1,
    "close": 1,
    "scan": 3,
    "handoff": 1,
    "report": 1,
    "start": 1,
    "lock": 1,
    "unlock": 1,
    "tie_bag": 2,
    "claim": 1,
    "release": 1,
    "mark_done": 1,
    "give_up": 1,
    "talk_to_team": 1,
}

# Task kinds (clean_plate, unpack_dishwasher, ...) that missions instantiate by name.
TASK_TEMPLATES_PATH = Path("configs/task_templates.yaml")

# Its [project] version is the docs/CHANGELOG.md version, recorded in every run.json.
PYPROJECT_PATH = Path("pyproject.toml")

# Wall-clock seconds between console progress lines during an episode (a display setting only;
# nothing simulated depends on it). The first and last tick always get a line.
PROGRESS_LOG_SECONDS = 60.0

# Ticks a moving agent may make no progress (blocked by bodies or reserved workspaces) before
# move_to fails and hands the decision back to the policy.
BLOCKED_PATIENCE_TICKS = 6

# Randomized backoff: after this many ticks without progress, a mover sits out each movement round
# with this probability, which breaks the symmetry of two movers undoing each other's moves.
BACKOFF_AFTER_STUCK_TICKS = 2
BACKOFF_SIT_OUT_PROB = 0.5

# Object types that shatter when dropped. Breakability is physical, not a hidden attribute.
BREAKABLE_TYPES = ("plate", "cup")

# Drops are a time penalty, not a coordination event. A non-fragile object lands on the agent's
# own tile and the same gripper picks it up again after a pause. A shattered one triggers a reflex:
# pick up the shards, walk to the nearest trashcan, dispose, walk back to the task area, then carry
# on with the same task.
DROP_RECOVERY_TICKS = 2
SHARD_PICKUP_TICKS = 3
SHARD_DISPOSE_TICKS = 1
CLEAN_UP_TOOL = "clean_up"
RETURN_TO_TASK_TOOL = "return_to_task"
# When a teammate starts a dishwasher mid-load, a dish in hand goes back to where it was picked up.
RETURN_ITEM_TOOL = "return_item"
# Skills the sim runs on its own, below the executor: no LLM call, not interruptible, and they
# never give up on a blocked path because there is nobody to report one to.
REFLEX_TOOLS = (CLEAN_UP_TOOL, RETURN_TO_TASK_TOOL, RETURN_ITEM_TOOL)
# Task kinds that load dishes into a dishwasher and run it. Only a multi-dish kind can split.
WASH_KINDS = ("clean_plate", "clean_dishes")
SPLITTABLE_KIND = "clean_dishes"

# Object types that go through the dishwasher and get stored in cupboards.
DISH_TYPES = ("plate", "cup", "bowl")

# Static distance maps the path planner keeps (one per target set), oldest dropped first.
PLANNER_CACHE_SIZE = 512

# Exploration skills: `explore` heads for floor no teammate is known to have searched (finding new
# tasks); `search` heads for floor this agent has not seen itself (finding an object it needs, since
# coverage messages say where teammates looked, not what they saw).
EXPLORING_TOOLS = ("explore", "search")

# Skills that move the body. The movement phase (`sim.motion`) runs all of them together.
MOVING_TOOLS = ("move_to", *EXPLORING_TOOLS, *REFLEX_TOOLS)

# "posted": every mission task is known to the team at tick 0. "discovered": no task is known at
# tick 0; a mission task becomes known to an agent when it sees one of the task's reveal objects.
TASK_VISIBILITY_MODES = ("posted", "discovered")

# "mission": the start cells the mission lists. "uniform": seeded uniform draw over valid floor.
SPAWN_MODES = ("mission", "uniform")

# How C0 plans. "heuristic": insertion plus local search, no dependency. "cpsat": the same plan
# improved by OR-Tools CP-SAT, which also returns a proven bound (needs the [solver] extra).
ORACLE_METHODS = ("heuristic", "cpsat")

# The solved plan, written next to an oracle run's logs (hard rule 9).
ORACLE_PLAN_FILE = "oracle_plan.json"
# C0 plans every order of the washes sharing a dishwasher, up to this many order combinations.
ORACLE_MAX_MACHINE_ORDERS = 24
# Message kinds that coordination code reads every tick (`TickingPolicy.on_tick`); like coverage
# messages, they never make a decision due by arriving.
CODE_MESSAGE_KINDS = ("ledger", "cbba", "dialogue")
# C1 (RoCo-style dialogue) gets no `task_found` announcements: they carry structured task ids,
# and C1 agents hear of a teammate's find only through its dialogue (conditions.md).
DIALOGUE_CONDITION = "llm_only"
# Who picks each primitive: the coded executor, or the agent's LLM.
EXECUTOR_KINDS = ("coded", "llm")
# An LLM reply: at most this many tool calls, and a short thought (logged, never acted on).
LLM_MAX_CALLS_PER_REPLY = 3
LLM_THOUGHT_CHARS = 200
# Tools by which an agent says a task is finished: G's report, the ledger conditions' mark_done.
COMPLETION_TOOLS = ("report", "mark_done")
# Tools whose argument is a task id; an LLM agent may name only its own current task.
TASK_TOOLS = (*COMPLETION_TOOLS, "release", "give_up")
# G with the LLM executor takes a task it released again after this many ticks, as the coded
# executor never drops a mission task for good (CBBA's equivalent is `cbba.give_up_ticks`).
RELEASE_RETAKE_TICKS = 120
# Log streams are gzipped JSONL: <run_dir>/<stream>.jsonl.gz.
LOG_SUFFIX = ".jsonl.gz"
LOG_COMPRESS_LEVEL = 6
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"  # folder name of one launch of a named experiment


@dataclass(frozen=True)
class FaultConfig:
    """Physical faults. `drop_prob` applies once per pick, place, and move while holding."""

    drop_prob: float


@dataclass(frozen=True)
class SpawnConfig:
    """Where agents start. `min_separation` is in static steps (uniform mode only)."""

    mode: str
    min_separation: int


@dataclass(frozen=True)
class ExplorationConfig:
    """Coverage map used by the `explore` skill.

    Attributes:
        search_ttl: Ticks a cell stays "searched" after it was last in view; None means forever
            (finite missions). A finite value lets regenerating tasks be found again.
        share_every: Ticks between coverage broadcasts to teammates; None disables sharing.
    """

    search_ttl: int | None
    share_every: int | None


@dataclass(frozen=True)
class OracleConfig:
    """How the oracle (C0) plans. Limits are search counts and deterministic time, never a clock.

    Attributes:
        method: One of `ORACLE_METHODS`.
        restarts: Seeded starts of the heuristic.
        max_passes: Local-search passes per start.
        deterministic_time: CP-SAT's deterministic time limit (its own unit, near a second).
        workers: CP-SAT workers; results must not depend on this (interleaved search).
    """

    method: str
    restarts: int
    max_passes: int
    deterministic_time: float
    workers: int


@dataclass(frozen=True)
class BatteryConfig:
    """Battery and charging (designed in simulator.md; not implemented, must stay disabled).

    Drains are charge units per tick of the activity; `charge_rate` is per tick at the agent's own
    dock; `low_threshold` is the share of capacity at which low battery becomes a decision point.
    """

    enabled: bool
    capacity: float
    drain_idle: float
    drain_move: float
    drain_carry: float
    drain_manipulate: float
    charge_rate: float
    low_threshold: float


# Token estimates for the scripted test policy, which has no tokenizer.
CHARS_PER_TOKEN = 4
SCRIPTED_TOKENS_PER_CALL = 30


@dataclass(frozen=True)
class LatencyConfig:
    """Reference throughput that converts token counts into simulated deliberation ticks."""

    prefill_rate: float
    decode_rate: float
    seconds_per_tick: float


@dataclass(frozen=True)
class ChannelConfig:
    """Message channel parameters.

    Attributes:
        base_loss: Swept packet-loss rate p, applied to every message.
        comm_range: Hard range cutoff r in cells; None means unlimited.
        d0: Distance at which the falloff term is 0.5; None disables distance falloff.
        alpha: Falloff exponent.
        wall_penalty: Delivery multiplier per wall cell crossed.
        latency_ticks: Extra delivery delay on top of the one-tick hop.
    """

    base_loss: float
    comm_range: float | None
    d0: float | None
    alpha: float
    wall_penalty: float
    latency_ticks: int


@dataclass(frozen=True)
class ChurnConfig:
    """k-agent dropout and rejoin. Crash ticks are drawn uniformly in [earliest, latest]."""

    num_agents: int
    rejoin_delay: int
    earliest_tick: int
    latest_tick: int


@dataclass(frozen=True)
class ScriptedConfig:
    """Parameters of the scripted test policy."""

    wait_ticks: int


@dataclass(frozen=True)
class CbbaConfig:
    """CBBA and the replicated ledger, for the auction conditions (C2, C3).

    Attributes:
        bundle_size: Most tasks one agent holds at once.
        discount: The travel discount lambda in a bid, s_ij * discount ** steps_ij.
        gossip_every: Ticks between full broadcasts of an agent's ledger and bids (anti-entropy);
            changes are also broadcast as they happen.
        lease_ticks: How long a claim lasts unless renewed; a crashed owner's tasks reopen after it.
        give_up_ticks: How long an agent stays off a task it gave up.
    """

    bundle_size: int
    discount: float
    gossip_every: int
    lease_ticks: int
    give_up_ticks: int


@dataclass(frozen=True)
class DialogueConfig:
    """C1's lockstep dialogue rounds (docs/design/conditions.md, C1).

    Attributes:
        turns_per_member: A round ends after this many turns per member (M = this x members).
        replans: Rejected EXECUTE plans a round tolerates (K) before it ends with no plan.
        retry_ticks: How long an agent with no task waits before asking for another round, after
            a round gave it nothing.
        message_chars: Longest message a turn may say.
    """

    turns_per_member: int
    replans: int
    retry_ticks: int
    message_chars: int


@dataclass(frozen=True)
class LLMConfig:
    """The model server every LLM agent shares (hard rule 6), and how it is queried.

    Attributes:
        endpoint: Base URL of an OpenAI-compatible server (llama.cpp `llama-server`, vLLM).
        model: The model name the server expects.
        model_path: The model file, hashed into run.json (hard rule 9); None when not local.
        temperature: Sampling temperature; 0 for reproducible runs.
        seed: Sampling seed sent with every request.
        max_tokens: Cap on each reply.
        timeout_seconds: Wall-clock limit per request (a cost, never a result).
        retries: Attempts after a failed request before the episode stops.
        cache_dir: Response cache folder; None disables caching.
        replay_only: Answer only from the cache and fail on a miss (re-scoring old runs).
        history_keep: Most recent actions shown verbatim; older ones are summarized in one line.
        pin_slots: Send each agent to its own server slot, so its prompt prefix stays cached.
        thinking: Let a reasoning-capable model think before it answers (`enable_thinking`). Its
            thinking tokens are output tokens, so they lengthen the decision's latency (hard
            rule 3); the model and this setting are frozen together for a paper run.
    """

    endpoint: str
    model: str
    model_path: str | None
    temperature: float
    seed: int
    max_tokens: int
    timeout_seconds: float
    retries: int
    cache_dir: str | None
    replay_only: bool
    history_keep: int
    pin_slots: bool
    thinking: bool = False


@dataclass(frozen=True)
class EpisodeConfig:
    """Everything needed to reproduce one episode.

    Attributes:
        env: The environment (map and mission family) the episode runs on; names its run folders.
        condition: Which coordination logic every agent runs: "scripted" (the mission's fixed
            assignment; test scaffolding), "greedy" (baseline G), or "oracle" (C0). The LLM and
            auction conditions join this list as they are built.
        executor: One of `EXECUTOR_KINDS`: the coded executor, or an LLM choosing each step.
        task_visibility: One of `TASK_VISIBILITY_MODES`.
    """

    name: str
    env: str
    condition: str
    executor: str
    seed: int
    max_ticks: int
    map_path: Path
    mission_path: Path
    output_dir: Path
    snapshot_every: int
    task_visibility: str
    spawn: SpawnConfig
    exploration: ExplorationConfig
    latency: LatencyConfig
    channel: ChannelConfig
    churn: ChurnConfig
    faults: FaultConfig
    battery: BatteryConfig
    scripted: ScriptedConfig
    oracle: OracleConfig
    cbba: CbbaConfig
    dialogue: DialogueConfig
    llm: LLMConfig


def load_config(path: Path) -> EpisodeConfig:
    """Reads an episode config from YAML. Relative paths resolve against the working directory.

    Raises:
        ValueError: An unknown executor, task visibility, spawn mode, or oracle method, or the
            battery model switched on.
    """
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw["executor"] not in EXECUTOR_KINDS:
        raise ValueError(f"{path}: executor must be one of {EXECUTOR_KINDS}")
    if raw["task_visibility"] not in TASK_VISIBILITY_MODES:
        raise ValueError(f"{path}: task_visibility must be one of {TASK_VISIBILITY_MODES}")
    if raw["spawn"]["mode"] not in SPAWN_MODES:
        raise ValueError(f"{path}: spawn.mode must be one of {SPAWN_MODES}")
    if raw["oracle"]["method"] not in ORACLE_METHODS:
        raise ValueError(f"{path}: oracle.method must be one of {ORACLE_METHODS}")
    if raw["battery"]["enabled"]:
        raise ValueError(f"{path}: the battery model is not implemented yet; set enabled: false")
    return EpisodeConfig(
        name=raw["name"],
        env=raw["env"],
        condition=raw["condition"],
        executor=raw["executor"],
        seed=int(raw["seed"]),
        max_ticks=int(raw["max_ticks"]),
        map_path=Path(raw["map"]),
        mission_path=Path(raw["mission"]),
        output_dir=Path(raw["output_dir"]),
        snapshot_every=int(raw["snapshot_every"]),
        task_visibility=raw["task_visibility"],
        spawn=SpawnConfig(**raw["spawn"]),
        exploration=ExplorationConfig(**raw["exploration"]),
        latency=LatencyConfig(**raw["latency"]),
        channel=ChannelConfig(**raw["channel"]),
        churn=ChurnConfig(**raw["churn"]),
        faults=FaultConfig(**raw["faults"]),
        battery=BatteryConfig(**raw["battery"]),
        scripted=ScriptedConfig(**raw["scripted"]),
        oracle=OracleConfig(**raw["oracle"]),
        cbba=CbbaConfig(**raw["cbba"]),
        dialogue=DialogueConfig(**raw["dialogue"]),
        llm=LLMConfig(**raw["llm"]),
    )
