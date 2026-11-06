#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="$repo_root/.cache/robopoker"
commit=b6bf9e2788d43a129c0938f24b278ac2f910a5ed
mkdir -p "$repo_root/.cache"
if [[ ! -e "$checkout" ]]; then
    git clone --no-checkout https://github.com/krukah/robopoker.git "$checkout"
    git -C "$checkout" checkout --detach "$commit"
fi
actual="$(git -C "$checkout" rev-parse HEAD)"
if [[ "$actual" != "$commit" ]]; then
    echo "Checkout differs from the reviewed commit; preserving it: $actual" >&2
    exit 1
fi
if [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
    echo 'Upstream checkout has local changes; preserving it for review.' >&2
    exit 1
fi
echo "Verified robopoker $actual"
