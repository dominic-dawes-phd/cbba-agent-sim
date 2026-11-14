"""Message-level communication channel: loss, range, wall attenuation, latency, offline agents.

Every message is logged once, when its fate is known: dropped at send (reason `range`, `loss`, or
`offline`), or delivered or dropped as `offline` when it comes due.
"""

import math
import random
from dataclasses import dataclass

from sim.config import ChannelConfig
from sim.logger import EpisodeLogger
from sim.perception import walls_between
from sim.world import Cell


@dataclass
class Message:
    """One copy of a message addressed to one receiver."""

    sender: str
    receiver: str
    kind: str
    payload: dict[str, str]
    tick_sent: int
    tick_due: int


def delivery_prob(distance: float, walls_crossed: int, config: ChannelConfig) -> float:
    """Bounded falloff with distance and walls, times the swept base delivery rate 1 - p."""
    falloff = 1.0 if config.d0 is None else 1.0 / (1.0 + (distance / config.d0) ** config.alpha)
    return falloff * config.wall_penalty**walls_crossed * (1.0 - config.base_loss)


class Network:
    """Seeded lossy broadcast channel. Delivery takes one hop tick plus the configured latency."""

    def __init__(
        self,
        config: ChannelConfig,
        walls: set[Cell],
        seed: int,
        logger: EpisodeLogger,
        silenced: tuple[str, ...] = (),
    ) -> None:
        """Sets up the channel.

        Args:
            config: Loss, range, falloff, and latency.
            walls: Wall cells, for the wall penalty.
            seed: The episode seed (the channel has its own random stream).
            logger: Where every message, delivered or dropped, is logged.
            silenced: Message kinds this condition never sends (C1 sends no `task_found`).
        """
        self._silenced = silenced
        self._config = config
        self._walls = walls
        self._rng = random.Random(f"{seed}:channel")
        self._logger = logger
        self._in_flight: list[Message] = []

    def broadcast(
        self,
        sender: str,
        kind: str,
        payload: dict[str, str],
        tick: int,
        positions: dict[str, Cell],
        online: dict[str, bool],
    ) -> None:
        """Sends one copy to every other agent, deciding drops now with the seeded RNG."""
        if kind in self._silenced:
            return
        for receiver in sorted(positions):
            if receiver == sender:
                continue
            message = Message(
                sender, receiver, kind, payload, tick, tick + 1 + self._config.latency_ticks
            )
            start, end = positions[sender], positions[receiver]
            distance = math.dist(start, end)
            if not online[receiver]:
                self._log(message, tick, "dropped", "offline")
            elif self._config.comm_range is not None and distance > self._config.comm_range:
                self._log(message, tick, "dropped", "range")
            elif self._rng.random() >= delivery_prob(
                distance, walls_between(self._walls, start, end), self._config
            ):
                self._log(message, tick, "dropped", "loss")
            else:
                self._in_flight.append(message)

    def deliver(self, tick: int, online: dict[str, bool]) -> list[Message]:
        """Returns the messages due this tick whose receiver is online."""
        due = [message for message in self._in_flight if message.tick_due <= tick]
        self._in_flight = [message for message in self._in_flight if message.tick_due > tick]
        delivered = []
        for message in due:
            if online[message.receiver]:
                self._log(message, tick, "delivered", None)
                delivered.append(message)
            else:
                self._log(message, tick, "dropped", "offline")
        return delivered

    def _log(self, message: Message, tick: int, outcome: str, reason: str | None) -> None:
        self._logger.write(
            "messages",
            tick,
            {
                "tick_sent": message.tick_sent,
                "sender": message.sender,
                "receiver": message.receiver,
                "kind": message.kind,
                "payload": dict(message.payload),
                "outcome": outcome,
                "reason": reason,
            },
        )
