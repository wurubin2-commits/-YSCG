"""诊断归因：对漂移报告逐特征排序，给出贡献占比与可读结论。"""
from .monitor import DriftReport


class Attributor:
    def attribute(self, report: DriftReport, top_k: int = 5) -> dict:
        ranked = sorted(report.psi_per_feature.items(), key=lambda kv: -kv[1])
        total = sum(v for _, v in ranked) or 1.0
        top = [
            {"feature": f"f{feat}", "psi": round(psi, 4), "share": round(psi / total, 3)}
            for feat, psi in ranked[:top_k]
        ]
        conclusion = (
            f"检测到 {len(ranked)} 个特征参与漂移，主要贡献："
            + ", ".join(f"f{feat}({psi:.3f})" for feat, psi in ranked[:3])
            if ranked
            else "未检测到显著特征漂移"
        )
        return {"top_features": top, "conclusion": conclusion}
