"""LLM 顾问：可插拔接口。

- MockAdvisor：本地开发/校选赛阶段（无 PyTRiO 额度）的默认实现，不做任何建议。
- OpenAIAdvisor：OpenAI 兼容接口（企业赛接 PyTRiO Qwen），读取漂移/归因/预算
  上下文，输出策略参数调整建议（JSON），由 guardrails 白名单校验后生效。

注意：LLM 只建议策略参数（策略层自改进），不直接决定训练动作——
动作决策始终由规则引擎兜底，保证受控。
"""
import json
from abc import ABC, abstractmethod

import requests


class LLMAdvisor(ABC):
    @abstractmethod
    def advise(self, context: dict) -> dict | None:
        """返回 {"strategy_patch": {...}} 或 None（无建议/调用失败）。"""


class MockAdvisor(LLMAdvisor):
    def advise(self, context: dict) -> dict | None:
        return None


class OpenAIAdvisor(LLMAdvisor):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def advise(self, context: dict) -> dict | None:
        prompt = self._build_prompt(context)
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是工业模型自进化系统的策略优化顾问。基于漂移监测与"
                                "实验记录上下文，输出 JSON：{\"strategy_patch\": {参数名: 数值}}。"
                                "只允许调整：psi_mild, psi_severe, conf_drop_frac, "
                                "promote_min_gain, hpo_trials, annotate_per_drift, "
                                "retrain_annotate, light_boost_rounds, full_boost_rounds。"
                                "只输出 JSON，不要解释。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            return {"strategy_patch": data.get("strategy_patch", {})}
        except Exception:
            return None

    @staticmethod
    def _build_prompt(context: dict) -> str:
        return (
            "当前系统上下文（JSON）：\n"
            + json.dumps(context, ensure_ascii=False, default=str)
            + "\n请评估是否需要调整策略参数，并输出建议 JSON。"
        )


def create_advisor(advisor_type: str = "mock", **kwargs) -> LLMAdvisor:
    if advisor_type == "mock":
        return MockAdvisor()
    if advisor_type == "openai":
        return OpenAIAdvisor(**kwargs)
    raise ValueError(f"未知顾问类型: {advisor_type}")
