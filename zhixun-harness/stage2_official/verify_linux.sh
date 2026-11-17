#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || { echo "Usage: $0 <new-run-name> <new-image-tag>" >&2; exit 2; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo "Linux/amd64 required" >&2; exit 2; }
name=$1
image=$2
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$ ]]
command -v docker >/dev/null
endpoint=${DOCKER_HOST:-}
if [[ -z $endpoint || -n ${DOCKER_CONTEXT:-} ]]; then
  endpoint=$(docker context inspect --format '{{.Endpoints.docker.Host}}' "$(docker context show)")
fi
[[ $endpoint == unix:///* ]] || { echo "Select a Docker daemon local to this Linux host" >&2; exit 2; }
[[ $(docker info --format '{{.OSType}}/{{.Architecture}}') =~ ^linux/(x86_64|amd64)$ ]]
if docker image inspect "$image" >/dev/null 2>&1; then
  echo "Choose a new image tag: $image" >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python=${ZHIXUN_PYTHON:-python3}
build="stage2_official/build/$name"
mkdir -p stage2_official/build
mkdir "$build"
context_options=()
base=${ZHIXUN_BASE_IMAGE:-python:3.12-slim-bookworm}
build_options=(--platform linux/amd64 --build-arg "PYTHON_BASE=$base")
if [[ -n ${ZHIXUN_WHEELHOUSE:-} ]]; then
  context_options+=(--wheelhouse "$ZHIXUN_WHEELHOUSE")
  build_options+=(--network none)
fi
printf '%s\n' "$base" > "$build/base-image.txt"
"$python" -B -m stage2_official.build_context "$build/context" "${context_options[@]}"
docker build "${build_options[@]}" -t "$image" "$build/context" 2>&1 | tee "$build/build.log"
image_id=$(docker image inspect --format '{{.Id}}' "$image")
[[ $(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image_id") == linux/amd64 ]]
docker run --rm --read-only --network none --cap-drop ALL --security-opt no-new-privileges \
  --cpus 4 --memory 8g --memory-swap 8g --entrypoint python3 "$image_id" -c '
import json, os
from pathlib import Path

cgroup = Path("/sys/fs/cgroup")
quota, period = (cgroup / "cpu.max").read_text().split()
resources = {"cpu": int(quota) / int(period), "memory": int((cgroup / "memory.max").read_text()),
             "swap": int((cgroup / "memory.swap.max").read_text()), "uid": os.getuid(),
             "interfaces": sorted(p.name for p in Path("/sys/class/net").iterdir()),
             "gpu": any(Path("/dev").glob("nvidia*")) or Path("/dev/dri").exists(),
             "read_only": bool(os.statvfs("/app").f_flag & os.ST_RDONLY)}
print(json.dumps(resources))
assert (resources["cpu"], resources["memory"], resources["swap"]) == (4, 8 * 1024**3, 0)
assert resources["interfaces"] == ["lo"] and not resources["gpu"]
assert resources["uid"] != 0 and resources["read_only"]
' | tee "$build/resources.json"
"$python" -B -m stage2_official.rehearse --name "$name" --image "$image_id" --rounds 10 | tee "$build/rehearsal.log"
"$python" -B - "stage2_official/runs/$name/summary.json" <<'PY'
import json
from pathlib import Path
import sys

tasks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["tasks"]
if set(tasks) != {"A", "B", "C"}:
    raise SystemExit("Expected all three tasks in the Linux rehearsal")
for name, task in tasks.items():
    if any(task[key] != 10 for key in ("rounds", "trials", "model_calls")) or len(task["predictions"]) != 2:
        raise SystemExit(f"{name}: incomplete training, measurement, model receipts or predictions")
print("Verified all 30 rounds, measurements and model receipts, and six predictions")
PY
[[ $(docker image inspect --format '{{.Id}}' "$image") == "$image_id" ]]
archive="$build/zhixun-harness-linux-amd64.tar"
docker save -o "$archive.partial" "$image"
bytes=$(wc -c < "$archive.partial")
[[ $bytes -le 2000000000 ]] || { echo "Archive exceeds the 2 GB rehearsal limit: $bytes" >&2; exit 2; }
mv "$archive.partial" "$archive"
sha256sum "$archive" > "$build/SHA256SUMS"
printf '%s\n' "$image_id" > "$build/image-id.txt"
printf '%s\n' "$bytes" > "$build/archive-bytes.txt"
echo "$archive"
