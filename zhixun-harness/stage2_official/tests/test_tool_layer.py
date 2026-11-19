import json

import pytest

from stage2_official import tool_layer
from stage2_official.tool_layer import ToolBox, ToolRefused, calls_of, latest_receipt, run_tool_rounds


def test_latest_receipt_walks_back_past_keep_rounds(tmp_path):
    task = tmp_path / "B"
    for index in (1, 2, 3):
        receipt = task / f"round-{index:02d}" / "candidate"
        receipt.mkdir(parents=True)
        if index == 2:
            (receipt / "training.json").write_text("{}", encoding="utf-8")
    current = task / "round-04" / "workspace" / "B-04-xyz"
    current.mkdir(parents=True)
    assert latest_receipt(current, 4) == task / "round-02" / "candidate"
    assert latest_receipt(current, 1) is None


def test_latest_receipt_gives_up_on_an_unknown_layout(tmp_path):
    assert latest_receipt(tmp_path / "nowhere", 5) is None


def test_read_metrics_reports_absence(tmp_path):
    assert ToolBox(tmp_path).read_metrics({}) == {"available": False,
                                                 "note": "no metrics recorded for this round"}


def test_read_metrics_merges_round_files(tmp_path):
    (tmp_path / "decision.json").write_text('{"decision": "adopted"}', encoding="utf-8")
    state = ToolBox(tmp_path).read_metrics({})["state"]
    assert state["decision"] == {"decision": "adopted"}


def test_read_log_tail_limits_lines(tmp_path):
    (tmp_path / "history.jsonl").write_text("\n".join(str(i) for i in range(30)), encoding="utf-8")
    tail = ToolBox(tmp_path).read_log_tail({"name": "history.jsonl", "lines": 5})
    assert tail["lines"] == 5
    assert tail["text"].splitlines()[-1] == "29"


@pytest.mark.parametrize("name", ["secret.txt", "../history.jsonl", "/etc/passwd"])
def test_read_log_tail_refuses_outside_allowlist(tmp_path, name):
    with pytest.raises(ToolRefused):
        ToolBox(tmp_path).read_log_tail({"name": name})


def test_missing_file_error_names_what_is_readable(tmp_path):
    (tmp_path / "training.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ToolRefused) as failure:
        ToolBox(tmp_path).read_log_tail({"name": "history.jsonl"})
    assert "training.json" in str(failure.value)
    assert "outside" not in str(failure.value)


def test_write_config_is_single_shot(tmp_path):
    box = ToolBox(tmp_path)
    assert box.write_config({"rate": 0.001})["written"] is True
    assert json.loads((tmp_path / "config-patch.json").read_text(encoding="utf-8")) == {"rate": 0.001}
    with pytest.raises(ToolRefused):
        box.write_config({"batch": 64})


@pytest.mark.parametrize("patch", [{"rate": 0.5}, {"batch": 96}, {"epochs": 9},
                                   {"recipe": "made-up"}, {"unknown": 1}, {}])
def test_write_config_refuses_out_of_range(tmp_path, patch):
    with pytest.raises(ToolRefused):
        ToolBox(tmp_path).write_config(patch)


def test_write_config_lands_in_the_current_round_not_the_evidence(tmp_path):
    evidence = tmp_path / "earlier"
    current = tmp_path / "current"
    evidence.mkdir()
    current.mkdir()
    box = ToolBox(evidence, write_root=current)
    assert box.write_config({"rate": 0.001})["written"] is True
    assert (current / "config-patch.json").is_file()
    assert not (evidence / "config-patch.json").exists()


def test_dispatch_refuses_unknown_tool(tmp_path):
    with pytest.raises(ToolRefused):
        ToolBox(tmp_path).dispatch("shell", {})


def test_tool_round_feeds_result_back(tmp_path):
    (tmp_path / "history.jsonl").write_text("loss 1.2\nloss 1.1", encoding="utf-8")
    seen = []

    def invoke(messages):
        seen.append([item["role"] for item in messages])
        if "tool" not in seen[-1]:
            return {"text": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_log_tail",
                              "arguments": json.dumps({"name": "history.jsonl", "lines": 2})}}]}
        return {"text": '{"action": "keep"}', "tool_calls": []}

    text, trace = run_tool_rounds(invoke, [{"role": "user", "content": "{}"}], ToolBox(tmp_path))
    assert text == '{"action": "keep"}'
    assert trace[0]["ok"] is True and trace[0]["name"] == "read_log_tail"
    assert seen[-1][-2:] == ["assistant", "tool"]


def test_tool_round_turns_refusal_into_result(tmp_path):
    def invoke(messages):
        if "tool" not in [item["role"] for item in messages]:
            return {"text": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "write_config",
                                                              "arguments": '{"rate": 99}'}}]}
        return {"text": "", "tool_calls": []}

    _, trace = run_tool_rounds(invoke, [{"role": "user", "content": "{}"}], ToolBox(tmp_path))
    assert trace[0]["ok"] is False
    assert "rate" in trace[0]["result"]["error"]


def test_tool_round_stops_at_budget(tmp_path):
    def invoke(messages):
        return {"text": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_metrics", "arguments": "{}"}}]}

    calls = {"n": 0}

    def limited(messages):
        calls["n"] += 1
        return invoke(messages)

    _, trace = run_tool_rounds(limited, [{"role": "user", "content": "{}"}], ToolBox(tmp_path),
                               max_rounds=2, max_calls=1)
    assert calls["n"] == 2
    assert len(trace) == 1


def test_finish_forces_an_answer_when_the_budget_runs_out(tmp_path):
    def invoke(messages):
        return {"text": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_metrics", "arguments": "{}"}}]}

    def finish(messages):
        return {"text": '{"action": "keep", "reason": "forced"}', "tool_calls": []}

    text, trace = run_tool_rounds(invoke, [{"role": "user", "content": "{}"}], ToolBox(tmp_path),
                                  max_rounds=2, finish=finish)
    assert text == '{"action": "keep", "reason": "forced"}'
    assert len(trace) == 2


def test_without_finish_an_exhausted_budget_returns_the_last_text(tmp_path):
    def invoke(messages):
        return {"text": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_metrics", "arguments": "{}"}}]}

    text, _ = run_tool_rounds(invoke, [{"role": "user", "content": "{}"}], ToolBox(tmp_path), max_rounds=1)
    assert text == ""


def test_calls_of_reads_receipt_response():
    class Receipt:
        response = {"choices": [{"message": {"tool_calls": [{"id": "c1"}]}}]}

    assert calls_of(Receipt())[0]["id"] == "c1"
    assert calls_of(object()) == []


def test_specs_are_openai_function_schema():
    for spec in tool_layer.TOOL_SPECS:
        assert spec["type"] == "function"
        assert spec["function"]["name"]
        assert spec["function"]["parameters"]["type"] == "object"
