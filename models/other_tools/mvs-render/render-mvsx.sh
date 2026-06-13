#!/usr/bin/env bash
# Render .mvsj / .mvsx to PNG using Mol* mvs-render.
# Requires: conda env .conda-molstar-render, npm deps, xvfb-run, and a small
# patch to molstar (see node_modules/.../headless-screenshot.js — domCanvas).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV="$ROOT/.conda-molstar-render"
if [[ ! -d "$ENV" ]]; then
  echo "Missing conda env: $ENV" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
exec xvfb-run -a --server-args="-screen 0 1024x768x24" \
  node -r "$ROOT/document-polyfill.cjs" \
  "$ROOT/node_modules/molstar/lib/commonjs/cli/mvs/mvs-render.js" "$@"
