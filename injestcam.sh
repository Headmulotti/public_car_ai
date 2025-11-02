#!/bin/bash
# ingestcam.sh — Continuous HTTPS HLS recorder with auto chunking (no AI blocking)

STREAM_URL="https://6855e4345af72.streamlock.net:1935/lexington-live/lex-cam-035.stream/playlist.m3u8?tokenstarttime=1762051694&tokenendtime=1762062494&tokenhash=zWhr_kAy4UmP0x4YughNpJc3_4QX2rE3jNwc4slg1kA="
CAM_NAME="cam-080"
OUTPUT_DIR="./recordings"
LOG_DIR="./logs"
DURATION=100
FPS=10

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

while true; do
    CHUNK_INDEX=$((CHUNK_INDEX+1))
    DATE_TIME=$(date '+%Y%m%d_%H%M%S')
    OUTPUT_FILE="${OUTPUT_DIR}/${DATE_TIME}_${CAM_NAME}_${CHUNK_INDEX}.mp4"
    META_FILE="${OUTPUT_FILE%.mp4}.json"
    START_TIME_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "[INFO] Recording chunk #${CHUNK_INDEX} → ${OUTPUT_FILE}"

    ffmpeg -hide_banner -loglevel error -http_seekable 0 -t "$DURATION" -i "$STREAM_URL" \
        -c:v copy -c:a aac -b:a 128k -movflags +faststart -f mp4 "$OUTPUT_FILE"

    if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
        SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "[OK] Saved ($SIZE)"
        cat <<EOF > "$META_FILE"
{
  "camera": "$CAM_NAME",
  "stream_url": "$STREAM_URL",
  "start_time_utc": "$START_TIME_UTC",
  "duration_sec": $DURATION,
  "approx_fps": $FPS,
  "chunk_index": $CHUNK_INDEX,
  "filename": "$(basename "$OUTPUT_FILE")"
}
EOF
    else
        echo "[WARN] Failed chunk #$CHUNK_INDEX"
        sleep 5
    fi
    sleep 2
done
