import dataclasses
import json

from sim.channel import Network, delivery_prob
from sim.config import ChannelConfig
from sim.logger import EpisodeLogger

PERFECT = ChannelConfig(
    base_loss=0.0, comm_range=None, d0=None, alpha=4.0, wall_penalty=1.0, latency_ticks=0
)


def channel(**overrides):
    return dataclasses.replace(PERFECT, **overrides)


def message_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "messages.jsonl").read_text().splitlines()]


def send_many(config, walls, seed, run_dir, positions, online, count=50):
    with EpisodeLogger(run_dir) as logger:
        network = Network(config, walls, seed, logger)
        for tick in range(count):
            network.broadcast("a", "report", {"task_id": "t1"}, tick, positions, online)
        delivered = []
        for tick in range(count + 5):
            delivered.extend(network.deliver(tick, online))
    return delivered


def test_delivery_prob_falloff_and_walls():
    assert delivery_prob(10.0, 0, PERFECT) == 1.0
    config = channel(d0=10.0, alpha=2.0, wall_penalty=0.5, base_loss=0.2)
    assert abs(delivery_prob(10.0, 2, config) - 0.5 * 0.25 * 0.8) < 1e-12


def test_perfect_channel_delivers_everything_one_tick_later(tmp_path):
    positions = {"a": (0, 0), "b": (30, 0)}
    online = {"a": True, "b": True}
    delivered = send_many(PERFECT, {(5, 0)}, 0, tmp_path, positions, online, count=10)
    assert len(delivered) == 10
    assert all(message.tick_due == message.tick_sent + 1 for message in delivered)
    assert {row["outcome"] for row in message_rows(tmp_path)} == {"delivered"}


def test_latency_delays_delivery(tmp_path):
    positions = {"a": (0, 0), "b": (1, 0)}
    online = {"a": True, "b": True}
    with EpisodeLogger(tmp_path) as logger:
        network = Network(channel(latency_ticks=3), set(), 0, logger)
        network.broadcast("a", "report", {"task_id": "t1"}, 2, positions, online)
        assert network.deliver(5, online) == []
        assert len(network.deliver(6, online)) == 1


def test_range_cutoff(tmp_path):
    positions = {"a": (0, 0), "b": (10, 0)}
    online = {"a": True, "b": True}
    assert send_many(channel(comm_range=5.0), set(), 0, tmp_path, positions, online, 3) == []
    assert {row["reason"] for row in message_rows(tmp_path)} == {"range"}


def test_wall_penalty_attenuates_across_walls(tmp_path):
    positions = {"a": (0, 0), "b": (4, 0)}
    online = {"a": True, "b": True}
    delivered = send_many(channel(wall_penalty=0.0), {(2, 0)}, 0, tmp_path, positions, online, 5)
    assert delivered == []
    assert {row["reason"] for row in message_rows(tmp_path)} == {"loss"}


def test_offline_receiver_drops_at_send_and_at_delivery(tmp_path):
    positions = {"a": (0, 0), "b": (1, 0)}
    with EpisodeLogger(tmp_path) as logger:
        network = Network(PERFECT, set(), 0, logger)
        network.broadcast("a", "report", {"task_id": "t1"}, 0, positions, {"a": True, "b": False})
        network.broadcast("a", "report", {"task_id": "t2"}, 1, positions, {"a": True, "b": True})
        assert network.deliver(2, {"a": True, "b": False}) == []
    rows = message_rows(tmp_path)
    assert [(row["outcome"], row["reason"]) for row in rows] == [
        ("dropped", "offline"),
        ("dropped", "offline"),
    ]


def test_seeded_drops_are_reproducible(tmp_path):
    positions = {"a": (0, 0), "b": (3, 0), "c": (6, 0)}
    online = {"a": True, "b": True, "c": True}
    lossy = channel(base_loss=0.5)
    send_many(lossy, set(), 7, tmp_path / "first", positions, online)
    send_many(lossy, set(), 7, tmp_path / "second", positions, online)
    send_many(lossy, set(), 8, tmp_path / "other_seed", positions, online)
    outcomes = {}
    for name in ("first", "second", "other_seed"):
        outcomes[name] = [row["outcome"] for row in message_rows(tmp_path / name)]
    assert outcomes["first"] == outcomes["second"]
    assert outcomes["first"] != outcomes["other_seed"]
    assert 0 < outcomes["first"].count("delivered") < len(outcomes["first"])
