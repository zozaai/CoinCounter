#!/bin/bash
# Convert every HEIC in SRC to a 480x480 JPEG in OUT (uses macOS `sips`, no deps).
set -euo pipefail

SRC="${1:-/Users/alicheraghian/Documents/Eng skills/dataset}"
OUT="${2:-/Users/alicheraghian/Documents/Eng skills/CoinCounter/dataset/images}"
SIZE=480

mkdir -p "$OUT"
n=0
shopt -s nullglob nocaseglob
for f in "$SRC"/*.heic; do
  base="$(basename "$f")"
  name="${base%.*}"
  sips -s format jpeg -s formatOptions 90 \
       --resampleHeightWidth "$SIZE" "$SIZE" \
       "$f" --out "$OUT/$name.jpg" >/dev/null
  n=$((n+1))
done
echo "Converted $n images -> $OUT"
