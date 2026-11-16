import dataclasses
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from experiments.console import BANNER, check_server, print_banner, run_checks
from experiments.run_single import load_episode_scenario, run_config
from sim.config import FaultConfig, load_config
from sim.logger import format_duration
from tests.fake_llm import CodedBrainClient

SMOKE = Path("configs/m1_smoke.yaml")
HOME_GREEDY = Path("configs/m2_home_greedy.yaml")


class _LlamaServer(BaseHTTPRequestHandler):
    """Answers llama-server's /health and /props routes."""

    def do_GET(self) -> None:  # noqa: N802  (the http.server API)
        if self.path == "/health":
            reply = {"status": "ok"}
        else:
            reply = {
                "total_slots": 6,
                "default_generation_settings": {"n_ctx": 32768},
                "model_path": "/models/Qwen3-8B-Q4_K_M.gguf",
            }
        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


def test_the_banner_names_the_sim_and_its_version(capsys):
    print_banner("single episode")
    out = capsys.readouterr().out
    assert BANNER.strip() in out and "HiveSim v" in out and "single episode" in out


def test_durations_read_as_hours_minutes_or_seconds():
    assert [format_duration(s) for s in (12.7, 245, 3720)] == ["12s", "4m05s", "1h02m"]


def test_coded_runs_need_no_server_and_pass_the_checks(tmp_path, capsys):
    config = load_config(SMOKE)
    assert run_checks([config], tmp_path)
    out = capsys.readouterr().out
    assert "[ OK ] python" in out and "no model server needed" in out


def test_a_server_that_is_down_fails_its_check(capsys):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # free once the probe closes, so nothing answers
    assert not check_server(f"http://127.0.0.1:{port}")
    assert "start it: bash scripts/start_server.sh" in capsys.readouterr().out


def test_a_healthy_server_reports_its_slots_and_context(capsys):
    server = HTTPServer(("127.0.0.1", 0), _LlamaServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert check_server(f"http://127.0.0.1:{server.server_port}")
    server.shutdown()
    out = capsys.readouterr().out
    assert "6 slots, context 32768, Qwen3-8B-Q4_K_M.gguf" in out


def test_an_llm_episode_logs_progress_at_its_first_and_last_tick(tmp_path, caplog):
    config = dataclasses.replace(
        load_config(HOME_GREEDY), executor="llm", output_dir=tmp_path, faults=FaultConfig(0.0)
    )
    client = CodedBrainClient(config.scripted, load_episode_scenario(config).tasks)
    with caplog.at_level(logging.INFO, logger="sim.loop"):
        _, final_tick = run_config(config, client=client)
    lines = [record.getMessage() for record in caplog.records if record.name == "sim.loop"]
    assert lines[0].startswith(f"m2_home/{config.name}/seed_0 | tick 0/{config.max_ticks}")
    assert f"| tick {final_tick}/" in lines[-1] and "LLM " in lines[-1] and "calls" in lines[-1]
