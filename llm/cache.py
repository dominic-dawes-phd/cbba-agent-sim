"""Response cache keyed by everything that determines a reply (simulator.md section 13).

A rerun of an episode sends the same prompts, so it replays the same replies without a GPU, and an
old run can be re-scored offline (`replay_only`). Entries are one JSON file each, written
atomically, so the parallel requests of one tick never corrupt each other.
"""

import hashlib
import json
import os
from pathlib import Path

from sim.logger import JsonValue


class CacheMissError(LookupError):
    """A replay-only cache was asked for a reply it does not hold."""


class ResponseCache:
    """Replies stored under the sha256 of (model, parameters, messages, schema)."""

    def __init__(self, folder: Path) -> None:
        self._folder = folder

    @staticmethod
    def key(fields: dict[str, JsonValue]) -> str:
        """The cache key for a request's determining fields."""
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, JsonValue] | None:
        """The stored reply for a key, or None."""
        path = self._path(key)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def put(self, key: str, reply: dict[str, JsonValue]) -> None:
        """Stores a reply. A concurrent writer of the same key writes the same content."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.{id(reply)}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(reply, handle, sort_keys=True)
        os.replace(temporary, path)

    def _path(self, key: str) -> Path:
        return self._folder / key[:2] / f"{key}.json"
