#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 AGGREGATE_REPO_ID INPUT_REPO_ID [INPUT_REPO_ID ...]" >&2
    exit 2
fi

AGGREGATE_REPO_ID="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
python src/lerobot/scripts/lerobot_aggregate.py \
    --repo-ids "$@" \
    --aggr-repo-id "${AGGREGATE_REPO_ID}"
