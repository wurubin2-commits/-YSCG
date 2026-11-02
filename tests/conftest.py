import sys
from pathlib import Path

# 让测试可直接 import yscg（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
