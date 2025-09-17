#!/usr/bin/env bash
set -euo pipefail
MAX=$((2*1024*1024))   # 2 MB
# List modified & untracked, respecting .gitignore
mapfile -t files < <(git ls-files -o -m --exclude-standard)

small=()
large=()
for f in "${files[@]}"; do
  [[ -f "$f" ]] || continue
  size=$(stat -c%s "$f")
  if (( size <= MAX )); then
    small+=("$f")
  else
    large+=("$f")
  fi
done
# Stage small files
if ((${#small[@]})); then
  git add -- "${small[@]}"
fi

# Log skipped large files
mkdir -p logs
printf "%s\n" "${large[@]}" > logs/large_skipped.txt
echo "Staged ${#small[@]} files ≤2MB"
echo "Skipped ${#large[@]} files >2MB (logged to logs/large_skipped.txt)"
