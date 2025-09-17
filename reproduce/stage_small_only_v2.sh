#!/usr/bin/env bash
set -euo pipefail
MAX=2048k  # 2 MB

# Prune big dirs so we never even consider their files
# Add more patterns if you need to.
prune_dirs=(
  "./.git"
  "./.venv"
  "./.venv-llava15"
  "./datasets"
  "./checkpoints"
  "./runs"
  "./logs"
  "./data"
  "./external/LLaVA/checkpoints"
  "./external/LLaVA/datasets"
  "./external/LLaVA/runs"
  "./external/LLaVA/logs"
  "./external/vlmevalkit/checkpoints"
  "./external/vlmevalkit/datasets"
  "./external/vlmevalkit/runs"
  "./external/vlmevalkit/logs"
)
# Build the prune expression for find
prunes=()
for d in "${prune_dirs[@]}"; do
  prunes+=( -path "$d" -prune -o )
done

# Stage: files <= 2MB, regular files only, excluding pruned dirs
# Batch with xargs to avoid "argument list too long"
find . \( "${prunes[@]}" -false \) -o -type f -size -$MAX -print0 \
| xargs -0 -n 200 git add --

# Log: files > 2MB that were skipped (for your review)
find . \( "${prunes[@]}" -false \) -o -type f -size +$MAX -print \
| sed 's|^\./||' > logs/large_skipped.txt || true

echo "Staged up to 2MB-sized files in batches."
echo "Logged >2MB files to logs/large_skipped.txt"
