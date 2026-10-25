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
}

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
class EpisodeConfig:
    """Everything needed to reproduce one episode."""

    name: str
    seed: int
    max_ticks: int
    map_path: Path
    mission_path: Path
    output_dir: Path
    snapshot_every: int
    latency: LatencyConfig
    channel: ChannelConfig
    churn: ChurnConfig
    scripted: ScriptedConfig


def load_config(path: Path) -> EpisodeConfig:
    """Reads an episode config from YAML. Relative paths resolve against the working directory."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return EpisodeConfig(
        name=raw["name"],
        seed=int(raw["seed"]),
        max_ticks=int(raw["max_ticks"]),
        map_path=Path(raw["map"]),
        mission_path=Path(raw["mission"]),
        output_dir=Path(raw["output_dir"]),
        snapshot_every=int(raw["snapshot_every"]),
        latency=LatencyConfig(**raw["latency"]),
        channel=ChannelConfig(**raw["channel"]),
        churn=ChurnConfig(**raw["churn"]),
        scripted=ScriptedConfig(**raw["scripted"]),
    )
