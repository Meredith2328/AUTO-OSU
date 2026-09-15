#!/bin/bash
# Generate comparison maps for one song with the two open-source baselines.
# Usage: scripts/run_baselines.sh "path/to/song.mp3" [out_dir]
set -e
SONG="$1"; OUT="${2:-out/baselines}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
SONG_WIN="$(cygpath -w "$SONG")"; OUT_WIN="$(cygpath -w "$ROOT/$OUT")"
# torchcodec needs FFmpeg shared DLLs (external/ffmpeg-shared, BtbN 8.1 build)
export PATH="$ROOT/external/ffmpeg-shared/bin:$PATH"

echo "== Mapperatorinator (V32, std, 3 difficulties)"
cd "$ROOT/external/Mapperatorinator"
for SR in 2.5 4.0 5.5; do
  .venv/Scripts/python.exe inference.py audio_path="'$SONG_WIN'" output_path="'$OUT_WIN'" \
    gamemode=0 difficulty=$SR year=2023 export_osz=true hitsounded=true version="'Mapperatorinator ${SR}'" seed=1
done

echo "== osu-dreamer (v8, 3 difficulties)"
cd "$ROOT/external/osu-dreamer"
.venv/Scripts/python.exe -m osu_dreamer predict --model-path inference.pt --audio-file "$SONG_WIN" \
  --diff 2.5 5 4 3.5 4 --diff 4.0 8 6 4 5 --diff 5.5 9.3 8 4 6
mv -f _*.osz "$ROOT/$OUT/" 2>/dev/null || true
echo "== done: $ROOT/$OUT"
