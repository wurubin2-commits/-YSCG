"""Use Pydantic AI to validate one proposed continuation or keep action."""

import json
import os
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, ModelResponse, PromptedOutput, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits

# The official continuation example spends five epochs per round; keep two as the default so
# an ordinary run behaves exactly as before.
EPOCH_MAX = int(os.environ.get("ZHIXUN_EPOCH_MAX", "2"))


class Proposal(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    action: Literal["train", "keep"]
    reason: Annotated[str, Field(min_length=1, max_length=250)]
    rate: Annotated[float, Field(ge=0.00005, le=0.005, allow_inf_nan=False)] | None = None
    batch: Literal[32, 64, 128, 256] | None = None
    epochs: Annotated[int, Field(ge=1, le=EPOCH_MAX)] | None = None

    @model_validator(mode="after")
    def check_parameters(self):
        values = (self.rate, self.batch, self.epochs)
        valid = all(v is not None for v in values) if self.action == "train" else all(v is None for v in values)
        if not valid or not self.reason.strip():
            raise ValueError("Train needs every parameter, keep needs none, and both need a reason")
        return self


def create_agent(model):
    return Agent(model, output_type=PromptedOutput(Proposal), retries=0, instructions=(
        "You advise continuation of an existing industrial checkpoint. Use the current batch summary "
        "and completed experiment history. Propose one bounded train or keep action with a reason. "
        "The evaluator alone chooses a checkpoint. Use only the supplied information."
    ))


def suggest(agent, context):
    return agent.run_sync(json.dumps(context, allow_nan=False), usage_limits=UsageLimits(
        request_limit=1, tool_calls_limit=0
    )).output


def simulated():
    def respond(messages, _info):
        message = next(p for message in reversed(messages) for p in message.parts if p.part_kind == "user-prompt")
        context = json.loads(message.content)
        history, batch = context["history"], context["batch_size"]
        attempted = [item for item in history if item["plan"] and item["plan"]["action"] == "train"]
        if len(attempted) < 3:
            rate, batch = [(0.001, batch), (0.0005, batch), (0.00075, batch // 2)][len(attempted)]
            reason = f"Compare bounded continuation recipe {len(attempted) + 1}."
        else:
            last = attempted[-1]
            rate, batch = last["plan"]["rate"], last["plan"]["batch"]
            if last["status"] != "accepted" or context["shift"] > 0.35:
                rate = max(0.00005, rate * 0.5)
                reason = "Reduce the update after rejection or a large change in the batch summary."
            else:
                reason = "Repeat the last accepted settings on the next available batch."
        output = {"action": "train", "reason": reason, "rate": rate, "batch": batch, "epochs": 1}
        return ModelResponse(parts=[TextPart(json.dumps(output))])

    return create_agent(FunctionModel(respond))


def pytrio_connector(base_url, model_id, api_key, *, transport=None):
    """Construct the connector for later use. The local CLI selects a simulated policy."""
    from urllib.parse import urlsplit

    import httpx2
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    endpoint = urlsplit(base_url)
    if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username is not None
            or endpoint.password is not None or endpoint.query or endpoint.fragment
            or not model_id.strip() or not api_key):
        raise ValueError("Provide an approved HTTPS endpoint, model ID and credential")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=15.0,
                         http_client=httpx2.AsyncClient(transport=transport, trust_env=False,
                                                       follow_redirects=False))
    model = OpenAIChatModel(model_id, provider=OpenAIProvider(openai_client=client),
                            profile=OpenAIModelProfile(supports_json_object_output=False, supports_tools=False,
                                                       openai_chat_supports_max_completion_tokens=False),
                            settings={"max_tokens": 512, "timeout": 15.0})
    return create_agent(model), client
