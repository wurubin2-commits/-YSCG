"""Tool-call parsing for the live rehearsal adapter (no model access required)."""

from stage2_official.live_advisor import LiveAdvisor, split_tool_calls


def test_adapter_class_exposes_the_rehearsal_interface():
    # Import-only: instantiating would need the paid sampling service.
    assert callable(LiveAdvisor.ask)
    assert callable(LiveAdvisor.report)


def test_plain_answer_has_no_calls():
    content, calls = split_tool_calls('{"action":"keep","reason":"converged"}')
    assert content == '{"action":"keep","reason":"converged"}'
    assert calls == []


def test_single_tool_call_is_structured():
    content, calls = split_tool_calls(
        '<tool_call>\n{"name":"read_log_tail","arguments":{"name":"train.stderr.log","lines":10}}\n</tool_call>')
    assert content == ""
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["id"] == "call-1"
    assert call["function"]["name"] == "read_log_tail"
    assert call["function"]["arguments"] == {"name": "train.stderr.log", "lines": 10}


def test_multiple_calls_get_unique_ids():
    _, calls = split_tool_calls(
        '<tool_call>{"name":"read_metrics","arguments":{}}</tool_call>'
        '<tool_call>{"name":"read_log_tail","arguments":{"name":"summary.json"}}</tool_call>')
    assert [call["id"] for call in calls] == ["call-1", "call-2"]
    assert [call["function"]["name"] for call in calls] == ["read_metrics", "read_log_tail"]


def test_prose_around_a_call_is_preserved_as_content():
    content, calls = split_tool_calls(
        'I need the log first.\n<tool_call>{"name":"read_metrics"}</tool_call>')
    assert content == "I need the log first."
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == {}


def test_broken_payload_is_dropped_not_raised():
    content, calls = split_tool_calls('<tool_call>{"name": broken}</tool_call>')
    assert calls == []
    assert content == ""


def test_empty_and_none_input():
    assert split_tool_calls("") == ("", [])
    assert split_tool_calls(None) == ("", [])


def xml_call(tool, **params):
    """Build the XML spelling without embedding literal closing tags in this source."""
    parts = [f"<tool_call><function={tool}>"]
    for key, value in params.items():
        parts.append(f"<parameter={key}>{value}" + "<" + "/parameter>")
    parts.append("<" + "/function>" + "<" + "/tool_call>")
    return "".join(parts)


def test_xml_spelling_is_parsed_with_typed_arguments():
    content, calls = split_tool_calls(xml_call("read_log_tail", name="train.stderr.log", lines=10))
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_log_tail"
    assert calls[0]["function"]["arguments"] == {"name": "train.stderr.log", "lines": 10}
    assert type(calls[0]["function"]["arguments"]["lines"]) is int


def test_end_token_and_prose_are_cleaned():
    content, calls = split_tool_calls(xml_call("read_metrics") + "<|im_end|>")
    assert content == ""
    assert len(calls) == 1
    text, none = split_tool_calls('{"action":"keep","reason":"done"}<|im_end|>')
    assert text == '{"action":"keep","reason":"done"}'
    assert none == []


def test_json_and_xml_calls_share_the_id_sequence():
    _, calls = split_tool_calls(
        '<tool_call>{"name":"read_metrics","arguments":{}}</tool_call>'
        + xml_call("read_log_tail", name="summary.json"))
    assert [call["id"] for call in calls] == ["call-1", "call-2"]
    assert calls[1]["function"]["arguments"] == {"name": "summary.json"}
