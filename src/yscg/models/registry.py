"""模型注册表：champion/challenger 版本管理、promote/rollback、状态持久化。"""
import json
from datetime import datetime
from pathlib import Path

from .factory import MODEL_EXT, create_model


class ModelRegistry:
    def __init__(self, artifacts_dir: Path, model_type: str):
        self.model_type = model_type
        self.ext = MODEL_EXT[model_type]
        self.models_dir = artifacts_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = artifacts_dir / "registry.json"
        self._state = {"champion": None, "history": []}
        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def register(self, model, metrics: dict, tag: str, admission: str | None = None) -> str:
        version_id = f"v{len(self._state['history']) + 1}"
        filename = f"{version_id}-{tag}{self.ext}"
        model.save(str(self.models_dir / filename))
        record = {
            "id": version_id,
            "tag": tag,
            "file": filename,
            "metrics": metrics,
            "admission": admission,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._state["history"].append(record)
        self._save()
        return version_id

    def promote(self, version_id: str):
        if version_id not in {r["id"] for r in self._state["history"]}:
            raise ValueError(f"未知版本: {version_id}")
        self._state["champion"] = version_id
        self._save()

    def champion_record(self) -> dict | None:
        cid = self._state["champion"]
        if cid is None:
            return None
        return next((r for r in self._state["history"] if r["id"] == cid), None)

    def champion_metrics(self) -> dict:
        rec = self.champion_record()
        return rec["metrics"] if rec else {}

    def load_champion(self):
        rec = self.champion_record()
        if rec is None:
            return None
        return create_model(self.model_type).load(str(self.models_dir / rec["file"]))

    def champion_file(self) -> Path | None:
        """冠军模型的持久化文件路径（回滚护栏用它校验完整保留）。"""
        rec = self.champion_record()
        if rec is None:
            return None
        return self.models_dir / rec["file"]

    def qualified_versions(self) -> list:
        """获得准入资格（admitted）的版本，即「被保留的准入版本」。"""
        return [r for r in self._state["history"] if r.get("admission") == "pass"]

    def history(self) -> list:
        return list(self._state["history"])

    def _save(self):
        self.state_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
