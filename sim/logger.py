"""Append-only JSONL episode logs (hard rule 9).

Every row records `tick` (used in results) and `wall_time` (cost only). Metrics are computed from
these files afterwards, never inside the sim.
"""

import dataclasses
import json
import logging
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import IO, TypeAlias

from sim.config import EpisodeConfig

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

# One file per stream, written to <run_dir>/<stream>.jsonl. ledgers.jsonl arrives with the ledger.
LOG_STREAMS = ("events", "actions", "llm_calls", "messages", "beliefs", "state")

log = logging.getLogger(__name__)


class EpisodeLogger:
    """Opens the episode's JSONL files on entry and closes them on exit."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._files: dict[str, IO[str]] = {}

    def __enter__(self) -> "EpisodeLogger":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for stream in LOG_STREAMS:
            path = self.run_dir / f"{stream}.jsonl"
            self._files[stream] = path.open("w", encoding="utf-8", newline="\n")
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

    def write_run(self, config: EpisodeConfig) -> None:
        """Writes run.json: config, seed, code version. The model hash is null until LLMs run."""
        commit, dirty = _git_state()
        record = {
            "config": dataclasses.asdict(config),
            "seed": config.seed,
            "git_commit": commit,
            "git_dirty": dirty,
            "model_file_hash": None,
            "started_wall_time": time.time(),
        }
        with (self.run_dir / "run.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        log.warning("Could not read git state for run.json: %s", error)
        return None, None
    return commit, bool(status.strip())
