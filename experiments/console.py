"""HiveSim's console: banner, status checks, run headers, and summaries for the command line.

Everything here is display. Results come from the log files afterwards (hard rule 9), and the
wall-clock times shown are costs (hard rule 1). All lines go to stdout, log lines with a timestamp,
so a tmux pane and a Slurm log read the same. Colors appear only on a terminal, and never when the
NO_COLOR environment variable is set.
"""

import ctypes
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from sim.config import EpisodeConfig, LLMConfig
from sim.logger import JsonValue, format_duration, git_state, sim_version
from sim.scenario import Scenario

BANNER = r"""
    __  ___           _____ _
   / / / (_)   _____ / ___/(_)___ ___
  / /_/ / / | / / _ \\__ \/ / __ `__ \
 / __  / /| |/ /  __/__/ / / / / / / /
/_/ /_/_/ |___/\___/____/_/_/ /_/ /_/
"""

# How each condition is named in the paper (CLAUDE.md, "The five conditions").
CONDITION_LABELS = {
    "oracle": "C0 oracle",
    "llm_only": "C1 LLM-only",
    "numeric_cbba": "C2 numeric CBBA",
    "llm_cbba": "C3 LLM+CBBA",
    "greedy": "G greedy",
    "scripted": "dev only",
}

TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"
# A status check waits this long for the model server or nvidia-smi.
CHECK_TIMEOUT_SECONDS = 5.0
# Free disk below this under the output folder is a warning (a house episode logs about 1.6 MB).
MIN_FREE_DISK_GB = 5.0

YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"
STATUS_COLORS = {"OK": GREEN, "WARN": YELLOW, "FAIL": RED}
LEVEL_COLORS = {logging.WARNING: MAGENTA, logging.ERROR: RED, logging.CRITICAL: RED}

# Windows console mode flag that makes the console interpret the color codes above.
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
STD_OUTPUT_HANDLE = -11


class ConsoleFormatter(logging.Formatter):
    """Formats a log line as "<timestamp>| <message>", colored by level on a terminal."""

    def format(self, record: logging.LogRecord) -> str:
        """The record as one timestamped line (plus its traceback, if any)."""
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        if record.levelno in LEVEL_COLORS:
            message = paint(message, LEVEL_COLORS[record.levelno])
        return f"{paint(self.formatTime(record, TIMESTAMP_FORMAT), YELLOW)}| {message}"


def configure_console() -> None:
    """Sends INFO log lines to stdout through `ConsoleFormatter`. Call once per process."""
    if sys.platform == "win32" and sys.stdout.isatty():
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ConsoleFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def paint(text: str, color: str) -> str:
    """The text in a color on a terminal, plain otherwise."""
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        return f"{color}{text}{RESET}"
    return text


def print_banner(subtitle: str) -> None:
    """Prints the HiveSim banner with the version, commit, and what is starting."""
    commit, dirty = git_state()
    revision = "git unknown" if commit is None else f"git {commit[:7]}"
    if dirty:
        revision += " (uncommitted changes)"
    print(paint(BANNER, YELLOW + BOLD))
    print(f"  HiveSim v{sim_version()} | {revision} | {platform.node()}")
    print(f"  {subtitle}", flush=True)


def section(title: str) -> None:
    """Prints a section heading."""
    print(paint(f"\n== {title}", CYAN + BOLD), flush=True)


def field(name: str, value: object) -> None:
    """Prints one aligned "name  value" line."""
    print(f"  {name:<10} {value}", flush=True)


def report_check(status: str, name: str, detail: str) -> bool:
    """Prints one status line; returns False when the status is FAIL."""
    print(f"  [{paint(f'{status:^4}', STATUS_COLORS[status])}] {name:<8} {detail}", flush=True)
    return status != "FAIL"


def run_checks(configs: list[EpisodeConfig], output_dir: Path) -> bool:
    """Checks what a run needs before it starts and prints one line per check.

    Args:
        configs: Every episode about to run.
        output_dir: Where their logs go.

    Returns:
        False when a check failed, so the run should not start.
    """
    section("status checks")
    python_ok = sys.version_info >= (3, 11)
    passed = report_check("OK" if python_ok else "FAIL", "python", platform.python_version())
    commit, dirty = git_state()
    if commit is None:
        report_check("WARN", "git", "not a git checkout: run.json records no commit")
    elif dirty:
        report_check("WARN", "git", f"{commit[:7]} with uncommitted changes (fine for testing)")
    else:
        report_check("OK", "git", f"{commit[:7]}, clean")
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / 1e9
    disk_status = "OK" if free_gb >= MIN_FREE_DISK_GB else "WARN"
    report_check(disk_status, "disk", f"{free_gb:.0f} GB free under {output_dir}")
    llm_blocks = {config.llm for config in configs if config.executor == "llm"}
    if not llm_blocks:
        report_check("OK", "llm", "coded executor only: no model server needed")
        return passed
    check_gpu()
    for llm in sorted(llm_blocks, key=str):
        passed = check_llm(llm) and passed
    return passed


def check_gpu() -> None:
    """Reports the GPU name, memory in use, and driver from nvidia-smi (a warning if absent)."""
    query = "--query-gpu=name,memory.used,memory.total,driver_version"
    try:
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        report_check("WARN", "gpu", f"nvidia-smi unavailable ({error})")
        return
    for line in result.stdout.strip().splitlines():
        name, used, total, driver = (part.strip() for part in line.split(","))
        report_check("OK", "gpu", f"{name}, {used} of {total} in use, driver {driver}")


def check_llm(llm: LLMConfig) -> bool:
    """Checks one `llm:` block: the model file, the response cache, and the server.

    Returns:
        False when the model file is missing or the server does not answer.
    """
    passed = True
    if llm.model_path is not None:
        model = Path(llm.model_path)
        if model.is_file():
            report_check("OK", "model", f"{model.name}, {model.stat().st_size / 1e9:.1f} GB")
        else:
            fix = "fetch it: bash scripts/fetch_model.sh"
            passed = report_check("FAIL", "model", f"{model} not found; {fix}")
    cache = llm.cache_dir or "off"
    if llm.replay_only:
        report_check("OK", "cache", f"{cache}, replay only: answers come from the cache")
        return passed
    report_check("OK", "cache", cache)
    return check_server(llm.endpoint) and passed


def check_server(endpoint: str) -> bool:
    """Asks a llama.cpp server for its health, then for its slots and context size.

    Returns:
        False when the server is not up and ready.
    """
    try:
        with urllib.request.urlopen(f"{endpoint}/health", timeout=CHECK_TIMEOUT_SECONDS):
            pass
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        fix = "start it: bash scripts/start_server.sh (cluster: run_shard.sh starts it)"
        return report_check("FAIL", "server", f"{endpoint} not ready ({error}); {fix}")
    try:
        with urllib.request.urlopen(f"{endpoint}/props", timeout=CHECK_TIMEOUT_SECONDS) as reply:
            props = json.load(reply)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        return report_check("OK", "server", f"{endpoint} healthy (no /props: {error})")
    context = props.get("default_generation_settings", {}).get("n_ctx", "?")
    model = Path(props.get("model_path", "?")).name
    detail = f"{endpoint} healthy, {props.get('total_slots', '?')} slots, context {context}"
    return report_check("OK", "server", f"{detail}, {model}")


def print_episode_header(config: EpisodeConfig, scenario: Scenario, run_dir: Path) -> None:
    """Prints what one episode is about to run."""
    section("episode")
    label = CONDITION_LABELS.get(config.condition, config.condition)
    field("condition", f"{label} ({config.condition}), executor {config.executor}")
    field(
        "mission",
        f"{config.env}: {len(scenario.tasks)} tasks ({config.task_visibility}),"
        f" {len(scenario.agents)} agents, seed {config.seed}",
    )
    field("budget", f"{config.max_ticks} ticks")
    comm_range = config.channel.comm_range or "unlimited"
    field("channel", f"packet loss {config.channel.base_loss}, range {comm_range}")
    if config.churn.num_agents:
        churn = config.churn
        field("churn", f"{churn.num_agents} agents crash, rejoin after {churn.rejoin_delay} ticks")
    else:
        field("churn", "none")
    if config.executor == "llm":
        llm = config.llm
        field("model", f"{llm.model} at {llm.endpoint}, temperature {llm.temperature}")
    field("logs", run_dir)
    print(flush=True)


def print_episode_summary(
    config: EpisodeConfig,
    metrics: dict[str, JsonValue],
    final_tick: int,
    wall_seconds: float,
    run_dir: Path,
) -> None:
    """Prints how one episode went, from its metrics.json fields."""
    section("result")
    ending = "every task done or voided" if metrics["finished"] else "tick budget reached"
    field("ended", f"tick {final_tick} of {metrics['max_ticks']}: {ending}")
    done = f"{metrics['done']}/{metrics['tasks']} done, {metrics['voided']} voided"
    field("tasks", f"{done} ({metrics['partial_success']:.0%})")
    if metrics["makespan_vs_oracle"] is not None:
        field("vs oracle", f"{metrics['makespan_vs_oracle']:.2f}x the oracle's makespan")
    if config.executor == "llm":
        server_time = format_duration(metrics["llm_wall_seconds"])
        tokens = f"{metrics['tokens'] / 1e6:.2f}M tokens"
        field("llm", f"{metrics['llm_calls']} calls, {tokens}, {server_time} waiting on the server")
    field("messages", f"{metrics['messages_sent']} sent, {metrics['messages_dropped']} dropped")
    field("wall", format_duration(wall_seconds))
    field("logs", run_dir)


def print_sweep_header(
    spec: dict, episodes: list[EpisodeConfig], total: int, shard: int, shards: int, stamp: str
) -> None:
    """Prints what a sweep (or one shard of it) is about to run."""
    section("sweep")
    field("experiment", f"{spec['experiment']}, stamp {stamp}")
    share = f"{len(episodes)} of {total} (shard {shard} of {shards})" if shards > 1 else total
    field("episodes", f"{share}, {spec['workers']} at a time")
    cells = {}
    for config in episodes:
        cells[config.name] = cells.get(config.name, 0) + 1
    field("cells", ", ".join(f"{name} x{count}" for name, count in sorted(cells.items())))
    llm_endpoints = sorted({config.llm.endpoint for config in episodes if config.executor == "llm"})
    if llm_endpoints:
        field("server", ", ".join(llm_endpoints))
    field("logs", f"runs/{spec['experiment']}/<env>/<cell>/{stamp}/seed_<n>/")
    print(flush=True)


def print_sweep_footer(finished: int, failed: list[str], wall_seconds: float) -> None:
    """Prints the sweep's outcome before the metrics summary."""
    section("sweep finished")
    status = "OK" if not failed else "WARN"
    report_check(status, "episodes", f"{finished} finished, {len(failed)} failed")
    for label in failed:
        print(f"             {paint(label, RED)}")
    field("wall", format_duration(wall_seconds))
    print(flush=True)
