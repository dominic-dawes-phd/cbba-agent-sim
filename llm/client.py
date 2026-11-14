"""The LLM client: one shared OpenAI-compatible server, many agents (hard rule 6).

It speaks `/v1/chat/completions` (llama.cpp `llama-server`, vLLM), asks for JSON that matches a
schema, and returns the reply text with the server's token counts, which the latency model turns
into simulated ticks (hard rule 3). Wall-clock time is recorded as a cost only (hard rule 1).
Standard library only, so the sim gains no dependency.
"""

import functools
import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm.cache import CacheMissError, ResponseCache
from sim.agent import DecisionRequest
from sim.config import LLMConfig
from sim.logger import JsonValue
from sim.scenario import TaskSpec

HASH_CHUNK_BYTES = 1 << 24
RETRY_BACKOFF_SECONDS = 2.0

log = logging.getLogger(__name__)


class LLMServerError(RuntimeError):
    """The server kept failing after every retry; the message carries its last answer.

    A plain string error, so a sweep's worker process can hand it back (an HTTPError holds an
    open response, which cannot be pickled).
    """


@dataclass(frozen=True)
class ChatRequest:
    """One model call.

    Attributes:
        messages: Chat messages, each {"role": ..., "content": ...}.
        schema: The JSON schema the reply must match.
        purpose: "act", "utility", or "dialogue", for the logs.
        slot: The server slot to use (one per agent), or None.
        decision: The decision being made, for test doubles only; never sent to a server.
        task: The agent's current task, for test doubles only; never sent to a server.
    """

    messages: tuple[dict[str, str], ...]
    schema: dict[str, JsonValue]
    purpose: str
    slot: int | None = None
    decision: DecisionRequest | None = None
    task: TaskSpec | None = None


@dataclass(frozen=True)
class ChatReply:
    """A model reply with what it cost."""

    text: str
    prompt_tokens: int
    output_tokens: int
    wall_seconds: float
    cached: bool


class LLMClient(Protocol):
    """Anything that answers chat requests: the server client, or a test double."""

    def complete(self, request: ChatRequest) -> ChatReply:
        """Returns the reply to one request."""
        ...


class OpenAIChatClient:
    """Calls an OpenAI-compatible server, through the response cache when one is configured."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._cache = ResponseCache(Path(config.cache_dir)) if config.cache_dir else None
        self._model_hash = model_file_hash(config.model_path) if config.model_path else None

    def complete(self, request: ChatRequest) -> ChatReply:
        """Returns the reply, from the cache if it holds one.

        Raises:
            CacheMissError: In replay-only mode, when the cache lacks the reply.
            LLMServerError: The server failed or stayed unreachable after the configured retries.
        """
        body = self._body(request)
        key = None
        if self._cache is not None:
            # The slot only routes the request, so it is not part of what determines the reply.
            fields = {
                "model_hash": self._model_hash,
                **{k: v for k, v in body.items() if k != "id_slot"},
            }
            key = ResponseCache.key(fields)
            stored = self._cache.get(key)
            if stored is not None:
                return ChatReply(
                    stored["text"], stored["prompt_tokens"], stored["output_tokens"], 0.0, True
                )
        if self._config.replay_only:
            raise CacheMissError(f"no cached reply for a {request.purpose} request")
        began = time.perf_counter()
        reply = self._post(body)
        wall_seconds = time.perf_counter() - began
        text = reply["choices"][0]["message"]["content"]
        usage = reply["usage"]
        result = ChatReply(
            text, usage["prompt_tokens"], usage["completion_tokens"], wall_seconds, False
        )
        if self._cache is not None:
            self._cache.put(
                key,
                {
                    "text": text,
                    "prompt_tokens": result.prompt_tokens,
                    "output_tokens": result.output_tokens,
                },
            )
        return result

    def _body(self, request: ChatRequest) -> dict[str, JsonValue]:
        body: dict[str, JsonValue] = {
            "model": self._config.model,
            "messages": list(request.messages),
            "temperature": self._config.temperature,
            "seed": self._config.seed,
            "max_tokens": self._config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": request.schema, "strict": True},
            },
            # Qwen3 would otherwise think at length before every reply, and those tokens would
            # dominate each decision's latency, so thinking is opt-in (`llm.thinking`).
            "chat_template_kwargs": {"enable_thinking": self._config.thinking},
        }
        if self._config.pin_slots and request.slot is not None:
            body["id_slot"] = request.slot
        return body

    def _post(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Sends a request, retrying a failed one; raises LLMServerError after the last attempt."""
        data = json.dumps(body).encode("utf-8")
        for attempt in range(self._config.retries + 1):
            try:
                return self._send(data)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                error.close()
                if attempt == self._config.retries:
                    raise LLMServerError(f"server answered {error.code}: {detail}") from None
                log.warning("LLM request failed (%s: %s); retrying", error, detail)
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self._config.retries:
                    raise LLMServerError(f"server unreachable: {error}") from None
                log.warning("LLM request failed (%s); retrying", error)
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    def _send(self, data: bytes) -> dict[str, JsonValue]:
        url = f"{self._config.endpoint.rstrip('/')}/v1/chat/completions"
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as handle:
            return json.load(handle)


@functools.lru_cache(maxsize=4)
def model_file_hash(path: str) -> str:
    """The sha256 of a model file, computed once per process (a multi-gigabyte read)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(functools.partial(handle.read, HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
