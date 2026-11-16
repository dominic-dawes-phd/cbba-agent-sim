"""Append-only gzipped JSONL episode logs (hard rule 9).

Every row records `tick` (used in results) and `wall_time` (cost only). Metrics are computed from
these files afterwards, never inside the sim. Streams:
- events: task transitions, crashes, drops, breaks, splits.
- actions: every finished tool call with its outcome.
- llm_calls: every decision with its full prompt and response.
- messages: every message to each receiver, dropped ones included.
- beliefs: each agent's full belief graph at tick 0 (`initial`) and at the end (`final`), and every
  change in between with the new belief (`added`, `updated`, `missing`). Seeing an object again
  unchanged only refreshes its `last_seen`, which is not logged; every prompt shows it.
- poses: every agent's cell, heading, held objects, and online flag, every tick.
- ledgers: the rows each agent changed in its own ledger copy (auction conditions), as they change.
- dialogues: C1's canonical transcripts: each turn's speaker, round, members, text, and plan.
- state: full world snapshots every `snapshot_every` ticks and at the end.
"""

import dataclasses
import gzip
import json
import logging
import subprocess
import time
import tomllib
from pathlib import Path
from types import TracebackType
from typing import IO, TypeAlias

from sim.config import LOG_COMPRESS_LEVEL, LOG_SUFFIX, PYPROJECT_PATH, EpisodeConfig

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

# One file per stream, written to <run_dir>/<stream>.jsonl.gz. ledgers arrives with the ledger.
LOG_STREAMS = (
    "events",
    "actions",
    "llm_calls",
    "messages",
    "beliefs",
    "poses",
    "state",
    "ledgers",
    "dialogues",
)

log = logging.getLogger(__name__)


class EpisodeLogger:
    """Opens the episode's JSONL files on entry and closes them on exit."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._files: dict[str, IO[str]] = {}
        # Running totals for the console's progress lines; metrics come from the files afterwards.
        self.llm_calls = 0
        self.cached_calls = 0
        self.tokens = 0

    def __enter__(self) -> "EpisodeLogger":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for stream in LOG_STREAMS:
            self._files[stream] = gzip.open(
                self.run_dir / f"{stream}{LOG_SUFFIX}",
                "wt",
                encoding="utf-8",
                newline="\n",
                compresslevel=LOG_COMPRESS_LEVEL,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._files.values():
            handle.close()

    def write(self, stream: str, tick: int, row: dict[str, JsonValue]) -> None:
        """Appends one row to a stream."""
        record = {"tick": tick, "wall_time": time.time(), **row}
        self._files[stream].write(json.dumps(record, sort_keys=True) + "\n")
        if stream == "llm_calls":
            self.llm_calls += 1
            self.cached_calls += bool(row["cached"])
            self.tokens += row["prompt_tokens"] + row["output_tokens"]

    def write_run(
        self,
        config: EpisodeConfig,
        experiment: str | None = None,
        stamp: str | None = None,
        model_hash: str | None = None,
    ) -> None:
        """Writes run.json: config, seed, code version, and the model file's hash (hard rule 9).

        Args:
            config: The episode config.
            experiment: The named experiment this run belongs to, if any.
            stamp: The launch this run belongs to, if any.
            model_hash: The sha256 of the LLM's model file, or None (coded runs, remote models).
        """
        commit, dirty = git_state()
        record = {
            "config": dataclasses.asdict(config),
            "seed": config.seed,
            "experiment": experiment,
            "stamp": stamp,
            "sim_version": sim_version(),
            "git_commit": commit,
            "git_dirty": dirty,
            "model_file_hash": model_hash,
            "started_wall_time": time.time(),
        }
        with (self.run_dir / "run.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")


def read_stream(run_dir: Path, stream: str) -> list[dict[str, JsonValue]]:
    """Rows of one log stream."""
    with gzip.open(run_dir / f"{stream}{LOG_SUFFIX}", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def format_duration(seconds: float) -> str:
    """Wall-clock seconds as "1h02m", "4m05s", or "12s" for console lines (a cost, not a result)."""
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def git_state() -> tuple[str | None, bool | None]:
    """The checkout's commit and whether it has uncommitted changes; (None, None) outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        log.warning("Could not read the git state: %s", error)
        return None, None
    return commit, bool(status.strip())


def sim_version() -> str | None:
    """Reads the CHANGELOG version from pyproject.toml (installed metadata lags until reinstall)."""
    try:
        with PYPROJECT_PATH.open("rb") as handle:
            return tomllib.load(handle)["project"]["version"]
    except (OSError, tomllib.TOMLDecodeError, KeyError) as error:
        log.warning("Could not read the sim version: %s", error)
        return None
