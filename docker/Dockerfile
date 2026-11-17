# YSCG 工业模型自进化系统 · Harness 镜像
# 构建（仓库根目录）：
#   docker build --platform linux/amd64 -f docker/Dockerfile -t yscg-harness:0.1.0 .
# 导出：
#   docker save -o yscg-harness.tar yscg-harness:0.1.0
FROM python:3.12-slim-bookworm

# SDK（无第三方依赖）
COPY vendor/contest-harness-sdk-1.0.2/pyproject.toml /opt/contest-harness/pyproject.toml
COPY vendor/contest-harness-sdk-1.0.2/README.md /opt/contest-harness/README.md
COPY vendor/contest-harness-sdk-1.0.2/contest_harness /opt/contest-harness/contest_harness
RUN pip install --no-cache-dir /opt/contest-harness

# 数值依赖：工业模型是纯 NumPy MLP，仅需 numpy
RUN pip install --no-cache-dir numpy==2.2.6

# 选手代码（只读 /app）
COPY submission/harness.py /app/harness.py
COPY submission/contest_submission.py /app/contest_submission.py
COPY submission/diagnostics.py /app/diagnostics.py
COPY submission/policy.py /app/policy.py
COPY submission/baselines /app/baselines

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    BASELINE_CPU_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]