"""Local rehearsal adapter that puts a real PyTRiO model behind the mocked decision source.

The graded path never imports this module: it receives model access through the host gateway.
This adapter exists so a rehearsal can measure what the policy layer actually decides, and how
many tokens that costs, without touching the official entry.
"""

import json
import re

# Qwen-style tool calling travels in the sampled text: the chat template renders the tool
# specifications into the prompt, and the model answers with <tool_call> blocks. Two spellings
# occur in practice: a JSON payload, and an XML <function=...><parameter=...> body. Parsing
# both here lets the mocked host populate the same tool_calls field the real gateway returns.
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
XML_CALL_PATTERN = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)\s*</tool_call>", re.DOTALL)
PARAM_PATTERN = re.compile("<parameter=([^>]+)>([^<]*)" + "<" + "/parameter>", re.DOTALL)


def _coerce(value):
    """XML-spelled parameters arrive as text; recover JSON scalars where possible."""
    text = value.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def split_tool_calls(text):
    """Split one sampled answer into visible content and structured tool requests."""
    raw = text or ""
    calls = []

    def add(name, arguments):
        if isinstance(name, str) and name.strip():
            calls.append({"id": f"call-{len(calls) + 1}", "type": "function",
                          "function": {"name": name.strip(), "arguments": arguments}})

    for match in TOOL_CALL_PATTERN.finditer(raw):
        try:
            payload = json.loads(match.group(1))
        except ValueError:
            continue
        add(payload.get("name"), payload.get("arguments") or {})
    for match in XML_CALL_PATTERN.finditer(raw):
        arguments = {key: _coerce(value) for key, value in PARAM_PATTERN.findall(match.group(2))}
        add(match.group(1), arguments)
    content = XML_CALL_PATTERN.sub("", TOOL_CALL_PATTERN.sub("", raw))
    return content.replace("<|im_end|>", "").strip(), calls


class LiveAdvisor:
    def __init__(self, base_model, first_chunk_timeout=45.0, total_timeout=120):
        # The SDK defaults are a 60-minute first-chunk wait and a 4-hour total wait,
        # so a stalled sample looks like a hang. Bound it, otherwise a bad run costs
        # an hour before it reports anything.
        import pytrio
        pytrio.configure(
            sampling_first_chunk_timeout_seconds=first_chunk_timeout,
            sampling_chunk_idle_timeout_seconds=first_chunk_timeout,
            sampling_timeout_seconds=total_timeout,
            actor_capacity_wait_timeout_seconds=60,
        )
        from pytrio import ModelInput, SamplingParams, ServiceClient
        self._input, self._params = ModelInput, SamplingParams
        self.client = ServiceClient().create_sampling_client(base_model=base_model)
        self.tokenizer = self.client.get_tokenizer()
        self.model = base_model
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def ask(self, messages, max_tokens, tools=None):
        """One sampling step. Reasoning stays off, otherwise the budget goes into a draft.

        Returns the visible text plus any tool requests, so the caller can answer them and
        sample again instead of forcing the whole decision into a single completion.
        """
        options = {"tools": tools} if tools else {}
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False, **options)
        try:
            ids = [int(value) for value in rendered["input_ids"]]
        except (TypeError, KeyError, IndexError):
            ids = [int(value) for value in rendered]
        response = self.client.sample(
            self._input.from_ints(ids), 1,
            self._params(max_tokens=max_tokens, temperature=0)).result()
        sequence = response.sequences[0]
        used = len(sequence.tokens or [])
        self.calls += 1
        self.prompt_tokens += len(ids)
        self.completion_tokens += used
        content, calls = split_tool_calls(sequence.text or "")
        return {"text": content, "tool_calls": calls}

    def report(self):
        return {"model": self.model, "calls": self.calls,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}