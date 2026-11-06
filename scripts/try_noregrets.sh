#!/usr/bin/env bash
# Functional probe only: the tiny generated model is NOT a strong poker AI.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="$repo_root/.cache/noregrets"
commit=757f7692738069522195d2b486eec60a8b010c0c
mkdir -p "$repo_root/.cache" "$repo_root/artifacts/noregrets-smoke"
if [[ ! -e "$checkout" ]]; then
    git clone --no-checkout https://github.com/conorarmstrong/noregrets.git "$checkout"
    git -C "$checkout" checkout --detach "$commit"
fi
if [[ "$(git -C "$checkout" rev-parse HEAD)" != "$commit" || -n "$(git -C "$checkout" status --porcelain)" ]]; then
    echo 'Reviewed checkout has changed; preserving it for review.' >&2
    exit 1
fi
export CARGO_HOME="${CARGO_HOME:-/workspace/.tools/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/workspace/.tools/rustup}"
export CARGO_BUILD_JOBS=4
export RAYON_NUM_THREADS=4
cargo_bin="${CARGO_HOME}/bin/cargo"
if [[ ! -x "$cargo_bin" ]]; then
    echo 'Install Rust 1.90.0 first; see docs/robopoker-assessment.md.' >&2
    exit 1
fi
out="$(mktemp -d "$repo_root/artifacts/noregrets-smoke/run-XXXXXXXX")"
cd "$checkout"
run() {
    local name="$1" limit="$2"
    shift 2
    echo "Running $name; log: $out/$name.log"
    if timeout "$limit" "$@" > "$out/$name.log" 2>&1; then
        echo "$name: exit 0"
    else
        local status=$?
        tail -60 "$out/$name.log" >&2
        return "$status"
    fi
}
run build 300 "$cargo_bin" +1.90.0 build --release --locked
run engine-tests 300 "$cargo_bin" +1.90.0 test --release --locked engine::tests:: -- --test-threads=1
run checkpoint-test 120 "$cargo_bin" +1.90.0 test --release --locked cfr::tests::blueprint_and_checkpoint_roundtrip -- --exact
bin="$checkout/target/release/pluribus"
run train 90 "$bin" train --players 6 --stack 10000 --iters 200 --threads 4 \
    --raw-buckets --buckets 4 --rollouts 8 --runouts 2 \
    --out "$out/smoke.bin" --checkpoint "$out/checkpoint.bin"
run resume 90 "$bin" train --players 6 --stack 10000 --iters 20 --threads 4 \
    --resume "$out/checkpoint.bin" --out "$out/resumed.bin"
run inspect 30 "$bin" inspect --blueprint "$out/resumed.bin"
run eval 90 "$bin" eval --players 6 --blueprint "$out/resumed.bin" \
    --hands 200 --baseline caller --seed 20261105
python - "$out" "$commit" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys
out = Path(sys.argv[1])
summary = {'upstream_commit': sys.argv[2], 'purpose': 'functional-only, not strength evaluation'}
for name in ('engine-tests', 'checkpoint-test'):
    log = (out / (name + '.log')).read_text()
    m = re.search(r'test result: ok\. (\d+) passed; 0 failed;', log)
    if not m or int(m[1]) == 0:
        raise SystemExit('Required tests did not execute: ' + name)
    summary[name + '-passed'] = int(m[1])
inspected = (out / 'inspect.log').read_text()
if not re.search(r'players:\s+6\b', inspected) or not re.search(r'trained iterations:\s+220\b', inspected):
    raise SystemExit('Resumed blueprint is not a 6-player, 220-iteration artifact')
summary['players'] = 6
summary['iterations'] = 220
summary['files'] = {}
for name in ('smoke.bin', 'checkpoint.bin', 'resumed.bin'):
    p = out / name
    if not p.stat().st_size:
        raise SystemExit('Empty artifact: ' + name)
    summary['files'][name] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
(out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
PY
echo "Functional probe completed: $out/results.json"
