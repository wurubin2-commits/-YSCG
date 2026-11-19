"""Bounded tool access for the model-driven strategy layer.

The model may inspect metrics and the tail of its own training log, and may
write one candidate configuration patch. Everything else is refused: the toolbox
resolves every path against a single round directory and rejects escapes, unknown
files, unknown keys and out-of-range values before any input/output happens.
"""

import json
from pathlib import Path

MAX_TOOL_ROUNDS = 3
MAX_TOOL_CALLS = 6
MAX_LOG_LINES = 40
MAX_LOG_BYTES = 4096
MAX_WRITE_BYTES = 2048
MAX_CONFIG_WRITES = 1

ALLOWED_FILES = ("training.json", "history.jsonl", "train.stderr.log")
BATCH_VALUES = (32, 64, 128, 256)
RATE_RANGE = (1e-5, 0.01)
EPOCH_RANGE = (1, 3)
RECIPES = ("standard", "l2sp", "sensor_mask_v1", "b_residual_v1", "c_group_mse_v1")

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_metrics",
            "description": "Read the numeric state of the current round (parent score, elapsed time, epochs used).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log_tail",
            "description": "Read the last lines of one allowed file in the current round directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(ALLOWED_FILES)},
                    "lines": {"type": "integer", "minimum": 1, "maximum": MAX_LOG_LINES},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_config",
            "description": "Write one candidate hyper-parameter patch for the next training attempt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rate": {"type": "number"},
                    "batch": {"type": "integer"},
                    "epochs": {"type": "integer"},
                    "recipe": {"type": "string", "enum": list(RECIPES)},
                },
                "required": [],
            },
        },
    },
]


class ToolRefused(Exception):
    """Raised when a tool request is outside the allowed surface."""


def _inside(root, candidate):
    try:
        return Path(candidate).resolve().is_relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False


class ToolBox:
    """The whole tool surface available to the model, scoped to one round directory.

    ``root`` is where evidence is read from (often an earlier round's candidate), while
    ``write_root`` is the current round: a patch must never modify the historical receipt
    it was derived from.
    """

    def __init__(self, root, write_root=None):
        self.root = Path(root)
        self.write_root = Path(write_root) if write_root is not None else self.root
        self.writes = 0

    def read_metrics(self, arguments=None):
        state = {}
        for name in ("metrics.json", "decision.json"):
            path = self.root / name
            if path.is_file() and _inside(self.root, path):
                try:
                    state[name[:-5]] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raise ToolRefused(f"{name} is unreadable")
        if not state:
            return {"available": False, "note": "no metrics recorded for this round"}
        return {"available": True, "state": state}

    def read_log_tail(self, arguments=None):
        arguments = arguments or {}
        name = arguments.get("name")
        if name not in ALLOWED_FILES:
            raise ToolRefused("file is not on the tool allowlist")
        lines = arguments.get("lines", MAX_LOG_LINES)
        if type(lines) is not int or not 1 <= lines <= MAX_LOG_LINES:
            raise ToolRefused(f"lines must be an integer in [1,{MAX_LOG_LINES}]")
        path = self.root / name
        if not _inside(self.root, path):
            raise ToolRefused("file is outside the round directory")
        if not path.is_file():
            # Naming what actually exists is cheaper than letting the model guess again.
            present = [item.name for item in self.root.iterdir()
                       if item.is_file() and item.name in ALLOWED_FILES]
            raise ToolRefused(f"{name} is not present; readable here: {', '.join(present) or 'none'}")
        text = path.read_text(encoding="utf-8", errors="replace")[-MAX_LOG_BYTES:]
        tail = text.splitlines()[-lines:]
        return {"name": name, "lines": len(tail), "truncated": len(text) >= MAX_LOG_BYTES,
                "text": "\n".join(tail)}

    def write_config(self, arguments=None):
        arguments = arguments or {}
        if not arguments:
            raise ToolRefused("empty patch")
        if self.writes >= MAX_CONFIG_WRITES:
            raise ToolRefused(f"config writes are limited to {MAX_CONFIG_WRITES} per round")
        allowed = {"rate", "batch", "epochs", "recipe"}
        unknown = set(arguments) - allowed
        if unknown:
            raise ToolRefused("unknown keys: " + ",".join(sorted(unknown)))
        patch = {}
        if "rate" in arguments:
            rate = arguments["rate"]
            if type(rate) not in (int, float) or not RATE_RANGE[0] <= rate <= RATE_RANGE[1]:
                raise ToolRefused(f"rate must be within [{RATE_RANGE[0]},{RATE_RANGE[1]}]")
            patch["rate"] = rate
        if "batch" in arguments:
            if arguments["batch"] not in BATCH_VALUES:
                raise ToolRefused("batch must be one of " + ",".join(map(str, BATCH_VALUES)))
            patch["batch"] = arguments["batch"]
        if "epochs" in arguments:
            epochs = arguments["epochs"]
            if type(epochs) is not int or not EPOCH_RANGE[0] <= epochs <= EPOCH_RANGE[1]:
                raise ToolRefused(f"epochs must be within {EPOCH_RANGE}")
            patch["epochs"] = epochs
        if "recipe" in arguments:
            if arguments["recipe"] not in RECIPES:
                raise ToolRefused("recipe is not available for this task")
            patch["recipe"] = arguments["recipe"]
        if not patch:
            raise ToolRefused("patch carries no allowed value")
        path = self.write_root / "config-patch.json"
        if not _inside(self.write_root, path):
            raise ToolRefused("write target escapes the round directory")
        payload = json.dumps(patch, ensure_ascii=False, sort_keys=True)
        if len(payload.encode("utf-8")) > MAX_WRITE_BYTES:
            raise ToolRefused("patch exceeds the size limit")
        path.write_text(payload, encoding="utf-8")
        self.writes += 1
        return {"written": True, "path": path.name, "patch": patch}

    def dispatch(self, name, arguments):
        handler = {"read_metrics": self.read_metrics,
                   "read_log_tail": self.read_log_tail,
                   "write_config": self.write_config}.get(name)
        if handler is None:
            raise ToolRefused(f"unknown tool: {name}")
        return handler(arguments or {})


def latest_receipt(directory, round_index):
    """Locate the most recent candidate receipt this round is allowed to inspect.

    The current round has not trained yet, and keep rounds train nothing, so the useful
    evidence can sit several rounds back. Workspace layouts differ between the local
    rehearsal and the platform, so the round root is discovered by walking upwards; when
    no receipt exists the caller simply offers no tools.
    """
    root = Path(directory)
    for _ in range(4):
        if (root / f"round-{round_index:02d}").is_dir():
            break
        root = root.parent
    else:
        return None
    for index in range(round_index - 1, 0, -1):
        candidate = root / f"round-{index:02d}" / "candidate"
        if (candidate / "training.json").is_file():
            return candidate
    return None


def _arguments(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except ValueError:
            raise ToolRefused("arguments are not valid JSON")
        return value if isinstance(value, dict) else {}
    return {}


def run_tool_rounds(invoke, messages, toolbox, *, max_rounds=MAX_TOOL_ROUNDS, max_calls=MAX_TOOL_CALLS,
                    finish=None):
    """Answer one decision under a bounded tool loop.

    ``invoke(messages)`` returns ``{"text": str, "tool_calls": list}``. Tool
    requests are answered in place; the loop stops at the first answer without
    tool calls, or when the round or call budget runs out. A model that spends
    the whole budget asking for tools would otherwise leave an empty answer, so
    ``finish`` (one last tool-free invocation) can force a decision.
    """
    trace = []
    text = ""
    for _ in range(max_rounds):
        reply = invoke(messages)
        text = reply.get("text") or ""
        calls = reply.get("tool_calls") or []
        if not calls:
            return text, trace
        messages.append({"role": "assistant", "content": text, "tool_calls": calls})
        for call in calls:
            name = (call.get("function") or {}).get("name")
            identifier = call.get("id") or f"call-{len(trace) + 1}"
            if len(trace) >= max_calls:
                result = {"error": "tool call budget exhausted"}
            else:
                try:
                    result = toolbox.dispatch(name, _arguments((call.get("function") or {}).get("arguments")))
                except ToolRefused as error:
                    result = {"error": str(error)}
                trace.append({"name": name, "call_id": identifier,
                              "ok": "error" not in result, "result": result})
            messages.append({"role": "tool", "tool_call_id": identifier,
                             "content": json.dumps(result, ensure_ascii=False)})
    if finish is not None:
        reply = finish(messages)
        return reply.get("text") or "", trace
    return text, trace


def calls_of(reply):
    """Tool calls carried by a model receipt, or an empty list."""
    response = getattr(reply, "response", None)
    if not isinstance(response, dict):
        return []
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return []
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    return calls if isinstance(calls, list) else []
