"""The LLM pipeline: client and cache, prompts and replies, and G with the LLM executor."""

import dataclasses
import json
import pickle
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from allocation.greedy import GreedyAllocator
from allocation.llm_policy import Assignment, LLMPolicy
from experiments.analyze import run_metrics
from experiments.run_single import load_episode_scenario, run_config
from llm.cache import CacheMissError
from llm.client import ChatRequest, LLMServerError, OpenAIChatClient
from llm.prompts import (
    PRIMITIVE_TOOLS,
    executor_schema,
    executor_system,
    history_lines,
    parse_calls,
)
from sim.agent import ActionResult, DecisionRequest, TickReport, ToolCall
from sim.config import FaultConfig, load_config
from sim.logger import read_stream
from tests.fake_llm import CodedBrainClient, FixedReplyClient

HOME_GREEDY = Path("configs/m2_home_greedy.yaml")
TOOLS = (*PRIMITIVE_TOOLS, "report", "release")


def test_a_reply_is_parsed_into_calls_or_rejected_with_a_reason():
    good = json.dumps({"thought": "go", "calls": [{"tool": "pick", "args": ["plate_1"]}]})
    assert parse_calls(good, TOOLS) == ([ToolCall("pick", ("plate_1",))], None)
    assert parse_calls("not json", TOOLS)[1].startswith("reply was not valid JSON")
    claim = json.dumps({"thought": "", "calls": [{"tool": "claim", "args": ["t_1"]}]})
    assert parse_calls(claim, TOOLS)[1].startswith("unknown tool")
    assert (
        parse_calls(json.dumps({"thought": "", "calls": []}), TOOLS)[1] == "reply had no tool calls"
    )
    numeric = json.dumps({"thought": "", "calls": [{"tool": "wait", "args": [5]}]})
    assert parse_calls(numeric, TOOLS)[1] == "args of wait must be a list of strings"


def test_a_call_with_the_wrong_number_of_arguments_is_refused_with_the_usage_line():
    empty_wait = json.dumps({"thought": "", "calls": [{"tool": "wait", "args": []}]})
    error = parse_calls(empty_wait, TOOLS)[1]
    assert error.startswith("wait takes 1 argument(s): wait(ticks)") and "wait(10)" in error


def test_the_reply_schema_fixes_each_tools_number_of_arguments():
    variants = executor_schema(("wait", "pick", "explore"))["properties"]["calls"]["items"]["anyOf"]
    by_tool = {v["properties"]["tool"]["const"]: v["properties"]["args"] for v in variants}
    assert (by_tool["wait"]["minItems"], by_tool["wait"]["maxItems"]) == (1, 1)
    assert by_tool["wait"]["items"]["pattern"] == "^[0-9]{1,3}$"
    assert by_tool["explore"]["maxItems"] == 0 and by_tool["pick"]["maxItems"] == 1


def test_the_system_prompt_offers_only_the_conditions_tools():
    system = executor_system("arm_a", TOOLS, ("Report when done.",))
    assert "report(task)" in system and "release(task)" in system
    assert "claim(" not in system  # G claims in code; the LLM never sees the tool
    without = executor_system("arm_a", PRIMITIVE_TOOLS, ())
    assert "report(" not in without and "release(" not in without


def test_older_history_is_summarized_and_recent_actions_kept_verbatim():
    report = ActionResult(ToolCall("report", ("t_1",)), "ok", "announced report t_1")
    failed = ActionResult(ToolCall("pick", ("plate_1",)), "stale_belief", "plate_1 is not here")
    moves = [
        (tick, ActionResult(ToolCall("move_to", ("sink",)), "ok", "arrived")) for tick in (5, 6)
    ]
    lines = history_lines(((1, report), (2, failed), *moves), 2)
    assert lines[0] == "  (earlier: 2 actions, 1 failed; reported t_1)"
    assert lines[1:] == ["  t5 move_to(sink) -> ok: arrived", "  t6 move_to(sink) -> ok: arrived"]


class _FakeServer(BaseHTTPRequestHandler):
    """A one-route OpenAI-compatible server that counts its requests."""

    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802  (the http.server API)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeServer.requests.append(body)
        reply = {
            "choices": [{"message": {"content": '{"thought": "", "calls": []}'}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 7},
        }
        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


def test_the_client_asks_for_schema_json_and_caches_the_reply(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakeServer.requests = []
    llm = dataclasses.replace(
        load_config(HOME_GREEDY).llm,
        endpoint=f"http://127.0.0.1:{server.server_port}",
        cache_dir=str(tmp_path / "cache"),
    )
    request = ChatRequest(({"role": "user", "content": "hi"},), {"type": "object"}, "act", slot=2)
    first = OpenAIChatClient(llm).complete(request)
    second = OpenAIChatClient(llm).complete(dataclasses.replace(request, slot=4))
    server.shutdown()
    assert (first.prompt_tokens, first.output_tokens, first.cached) == (40, 7, False)
    assert second.cached and second.text == first.text  # the slot only routes the request
    sent = _FakeServer.requests
    assert len(sent) == 1 and sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["id_slot"] == 2 and sent[0]["temperature"] == 0.0
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}  # the default
    replay = dataclasses.replace(llm, replay_only=True)
    assert OpenAIChatClient(replay).complete(request).cached
    other = dataclasses.replace(request, messages=({"role": "user", "content": "new"},))
    with pytest.raises(CacheMissError):
        OpenAIChatClient(replay).complete(other)


class _FullServer(BaseHTTPRequestHandler):
    """Answers every request the way llama-server does when its context is full."""

    def do_POST(self) -> None:  # noqa: N802  (the http.server API)
        self.rfile.read(int(self.headers["Content-Length"]))
        data = b'{"error": {"message": "Context size has been exceeded."}}'
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


def test_a_server_that_keeps_failing_raises_a_picklable_error_with_its_message(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _FullServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    llm = dataclasses.replace(
        load_config(HOME_GREEDY).llm,
        endpoint=f"http://127.0.0.1:{server.server_port}",
        cache_dir=None,
        retries=0,
    )
    request = ChatRequest(({"role": "user", "content": "hi"},), {"type": "object"}, "act")
    with pytest.raises(LLMServerError, match="Context size has been exceeded") as raised:
        OpenAIChatClient(llm).complete(request)
    server.shutdown()
    # A sweep's worker process pickles the error to report it.
    assert "500" in str(pickle.loads(pickle.dumps(raised.value)))


def test_thinking_is_opt_in_and_changes_the_cache_key(tmp_path):
    base = load_config(HOME_GREEDY).llm
    thinking = dataclasses.replace(base, thinking=True, cache_dir=str(tmp_path))
    quiet = dataclasses.replace(base, cache_dir=str(tmp_path))
    request = ChatRequest(({"role": "user", "content": "hi"},), {"type": "object"}, "act")
    on_body = OpenAIChatClient(thinking)._body(request)
    off_body = OpenAIChatClient(quiet)._body(request)
    assert on_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert off_body["chat_template_kwargs"] == {"enable_thinking": False}


def llm_greedy(tmp_path, **changes):
    config = load_config(HOME_GREEDY)
    return dataclasses.replace(
        config,
        executor="llm",
        output_dir=tmp_path,
        faults=FaultConfig(0.0),
        **changes,
    )


def test_greedy_with_the_llm_executor_finishes_and_logs_each_model_call(tmp_path):
    config = llm_greedy(tmp_path)
    scenario = load_episode_scenario(config)
    client = CodedBrainClient(config.scripted, scenario.tasks)
    run_dir, final_tick = run_config(config, client=client)
    assert final_tick < config.max_ticks - 1
    metrics = run_metrics(run_dir)
    assert metrics["finished"] and metrics["hallucination_rate"] == 0.0
    calls = read_stream(run_dir, "llm_calls")
    assert {call["purpose"] for call in calls} == {"act"}
    assert all(call["prompt_tokens"] > 0 and "YOUR TASK" in call["prompt"] for call in calls)
    assert any(call["tool_calls"] and call["tool_calls"][0][0] == "claim" for call in calls)


def test_an_unusable_reply_costs_a_decision_and_is_shown_to_the_agent(tmp_path):
    config = llm_greedy(tmp_path, max_ticks=40)
    run_dir, _ = run_config(config, client=FixedReplyClient("I will pick up the plate."))
    actions = read_stream(run_dir, "actions")
    rejected = [row for row in actions if row["call"] == ["reply"]]
    assert rejected and rejected[0]["outcome"] == "invalid_call"
    assert rejected[0]["text"].startswith("reply was not valid JSON")
    calls = read_stream(run_dir, "llm_calls")
    assert all(call["rejected"] and call["latency_ticks"] > 0 for call in calls)


def test_the_greedy_allocator_claims_its_nearest_task_once():
    config = load_config(HOME_GREEDY)
    scenario = load_episode_scenario(config)
    allocator = GreedyAllocator(scenario.tasks, posted=True)
    assert "claim" not in allocator.tools and "report" in allocator.tools


class SetTaskAllocator:
    """An allocator that hands out whatever task the test sets."""

    tools = PRIMITIVE_TOOLS
    rules = ()

    def __init__(self, task):
        self.task = task

    def assign(self, request):
        return Assignment(self.task)

    def on_tick(self, view):
        return TickReport()

    def unscored(self, request):
        return []

    def set_scores(self, scores):
        return None

    def wants_observation(self, tick):
        return False


def test_code_unlocks_any_workspace_the_current_task_does_not_lock():
    scenario = load_episode_scenario(load_config(HOME_GREEDY))
    tasks = {task.task_id: task for task in scenario.tasks}
    allocator = SetTaskAllocator(tasks["t_unpack"])  # its recipe locks dishwasher_1
    reply = json.dumps({"thought": "", "calls": [{"tool": "wait", "args": ["5"]}]})
    policy = LLMPolicy(0, allocator, FixedReplyClient(reply), load_config(HOME_GREEDY).llm)
    card = scenario.agents[0].card
    free = DecisionRequest("arm_a", 0, "", [], None, {}, {}, (), (), None, None, card, ())
    holding = dataclasses.replace(free, own_reservations=("dishwasher_1",))
    assert policy.decide(holding).tool_calls == [ToolCall("wait", ("5",))]
    allocator.task = tasks["t_groceries"]
    unlock_first = [ToolCall("unlock", ("dishwasher_1",)), ToolCall("wait", ("5",))]
    assert policy.decide(holding).tool_calls == unlock_first
    # A rejected reply drops the unlock with it, so the next decision issues it again.
    assert policy.decide(holding).tool_calls == unlock_first


def test_a_task_tool_may_name_only_the_agents_own_task():
    scenario = load_episode_scenario(load_config(HOME_GREEDY))
    tasks = {task.task_id: task for task in scenario.tasks}
    allocator = SetTaskAllocator(tasks["t_unpack"])
    reply = json.dumps({"thought": "", "calls": [{"tool": "report", "args": ["none"]}]})
    allocator.tools = (*PRIMITIVE_TOOLS, "report")
    policy = LLMPolicy(0, allocator, FixedReplyClient(reply), load_config(HOME_GREEDY).llm)
    card = scenario.agents[0].card
    request = DecisionRequest("arm_a", 0, "", [], None, {}, {}, (), (), None, None, card, ())
    assert policy.decide(request).error == "report takes your own task (t_unpack)"
