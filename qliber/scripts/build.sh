#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURES="${1:-}"

pushd "${ROOT_DIR}" >/dev/null
if [[ -z "${FEATURES}" ]]; then
  cargo build --release
else
  cargo build --release --features "${FEATURES}"
fi
popd >/dev/null
