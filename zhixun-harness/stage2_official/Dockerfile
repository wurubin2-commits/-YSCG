ARG PYTHON_BASE=python:3.12-slim-bookworm
FROM ${PYTHON_BASE}
ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp \
    PYDANTIC_AI_NO_BANNER=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 BASELINE_CPU_THREADS=1

WORKDIR /app
COPY runtime.txt /tmp/runtime.txt
RUN --mount=type=bind,source=wheelhouse,target=/tmp/wheelhouse \
    if [ -n "$(ls -A /tmp/wheelhouse)" ]; then \
        python3 -m pip install --no-cache-dir --no-index --find-links=/tmp/wheelhouse -r /tmp/runtime.txt; \
    else \
        python3 -m pip install --no-cache-dir -r /tmp/runtime.txt; \
    fi \
    && rm /tmp/runtime.txt
COPY sdk /tmp/contest-sdk
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /tmp/contest-sdk \
    && python3 -m pip check \
    && rm -r /tmp/contest-sdk
COPY app /app
USER 10001:10001
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
