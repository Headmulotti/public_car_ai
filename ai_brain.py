#!/usr/bin/env python3
# ai_brain.py — Single-worker queue for video chunks
# Robust YOLOv8 tracking + CLIP labeling, persistent unlabeled recheck & cleanup,
# detailed logging, and fallbacks when no detections are produced.

import os
import time
import glob
import shutil
import cv2
import json
import sys
from datetime import datetime, timedelta
from ultralytics import YOLO
from real_ai import what_is_this_car

# =========================
# CONFIG
# =========================
WATCH_DIR       = "./recordings"
PROCESSED_DIR   = "./processed"
RESULTS_DIR     = "./processed"          # keep results with processed video
UNCERTAIN_DIR   = "./unlabeled_clips"
MODEL_PATH      = "yolov8n.pt"
TRACKER_CFG     = "bytetrack.yaml"

# Object classes (COCO): car=2, motorcycle=3, bus=5, truck=7
# If you truly only want cars, set CLASSES=[2].
CLASSES         = [2, 7]

# YOLO knobs
YOLO_CONF       = 0.2
YOLO_IOU        = 0.45
YOLO_IMGSZ      = 960
YOLO_VIDSTRIDE  = 2    # increase to 2 to skip frames if CPU bound

# Labeling gates
CAR_CLASS_ID    = 2     # used only for your own logic; detections are filtered by CLASSES
PAD             = 20
MIN_FRAMES      = 5
UNCERTAIN_THRESHOLD = 0.60

# Logging / debug
HEARTBEAT_EVERY = 150
MAX_EMPTY_FRAMES_WARN = 500   # warn if tracker runs this many frames with 0 boxes
RECHECK_AFTER_EACH_VIDEO = True

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UNCERTAIN_DIR, exist_ok=True)

# =========================
# UTIL: colored logging
# =========================
C = {
    "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
    "blue": "\033[94m", "cyan": "\033[96m", "reset": "\033[0m"
}
def log(msg, col=None):
    if col and col in C:
        print(f"{C[col]}{msg}{C['reset']}")
    else:
        print(msg)

# =========================
# OPTIONAL: single-file mode
# =========================
single_file_mode = None
if len(sys.argv) >= 2:
    # If user passed a specific video, process once and exit
    single_file_mode = sys.argv[1]
    if not os.path.exists(single_file_mode):
        log(f"❌ Video not found: {single_file_mode}", "red")
        sys.exit(1)

# =========================
# Load YOLO model once
# =========================
log("Loading YOLO model...", "blue")
model = YOLO(MODEL_PATH)
log("YOLO ready.", "green")

# =========================
# Recheck & cleanup unlabeled images
# =========================
def recheck_uncertain_and_cleanup():
    unlabeled = sorted(glob.glob(os.path.join(UNCERTAIN_DIR, "*.jpg")))
    if not unlabeled:
        log("[RECHECK] No unlabeled images found.", "green")
        return []

    log(f"[RECHECK] Found {len(unlabeled)} unlabeled images.", "cyan")
    reviewed = []
    for path in unlabeled:
        try:
            img = cv2.imread(path)
            if img is None:
                log(f"[RECHECK] Skipping unreadable: {path}", "yellow")
                continue
            label = what_is_this_car(img)
            full_label = f"{label.get('color','?')} {label.get('make','?')} {label.get('model','?')}"
            log(f"→ Re-ID {os.path.basename(path)} → {full_label}", "cyan")

            reviewed.append({
                "file": os.path.basename(path),
                "make":  label.get("make"),
                "model": label.get("model"),
                "color": label.get("color"),
                "make_conf": label.get("make_conf"),
                "model_conf": label.get("model_conf")
            })
        except Exception as e:
            log(f"[RECHECK ERROR] {path}: {e}", "red")
        finally:
            # Always delete after checking
            try:
                os.remove(path)
            except Exception as e:
                log(f"[WARN] Could not delete {path}: {e}", "yellow")

    log("[RECHECK] Cleanup complete — unlabeled_clips emptied.", "green")
    return reviewed

# =========================
# Sanity probes for a video file
# =========================
def probe_video_quick(video_path, max_probe_frames=30):
    """Open the file with cv2 directly to ensure frames can be read."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"[PROBE] Cannot open video: {video_path}", "red")
        return {"ok": False}

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= cap.get(cv2.CAP_PROP_FPS)
    frames_ok = 0
    for _ in range(max_probe_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames_ok += 1
    cap.release()
    return {"ok": frames_ok > 0, "width": w, "height": h, "fps": fps, "frames_ok": frames_ok}

# =========================
# Core processing for one file
# =========================
def process_video(video_path):
    base = os.path.basename(video_path)
    meta_path = video_path.replace(".mp4", ".json")
    # Default timestamps from meta (if present)
    start_time_utc = None
    approx_fps = 30.0

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            # Parse camera meta
            st = meta.get("start_time_utc")
            if st:
                # support Z suffix
                if st.endswith("Z"):
                    st = st.replace("Z", "+00:00")
                start_time_utc = datetime.fromisoformat(st)
            approx_fps = float(meta.get("approx_fps") or approx_fps)
        except Exception as e:
            log(f"[META] Failed to read meta: {e}", "yellow")

    if start_time_utc is None:
        # Fallback to file mtime as base
        start_time_utc = datetime.utcfromtimestamp(os.path.getmtime(video_path))

    log("------------------------------------------------------------")
    log(f"[PROCESSING] {video_path}", "cyan")
    log(f"[META] start_time_utc={start_time_utc.isoformat()} fps≈{approx_fps}", "yellow")

    # Quick probe to ensure frames are readable
    probe = probe_video_quick(video_path)
    if not probe.get("ok"):
        log("[WARN] Probe failed or no frames read; will try YOLO tracker anyway.", "yellow")
    else:
        log(f"[PROBE] ok frames={probe['frames_ok']} res={probe['width']}x{probe['height']} fps≈{probe['fps']:.2f}", "green")

    start = time.time()
    seen_cars = set()
    frame_counts = {}
    results = []
    frame_index = 0
    frames_with_boxes = 0
    empty_frames = 0

    # Run YOLO tracking
    tracker = model.track(
        source=video_path,
        show=False,
        tracker=TRACKER_CFG,
        classes=CLASSES,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        vid_stride=YOLO_VIDSTRIDE,
        stream=True,
        persist=True
    )

    for result in tracker:
        if result is None or getattr(result, "orig_img", None) is None:
            empty_frames += 1
            frame_index += 1
            continue

        frame = result.orig_img
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            empty_frames += 1
        else:
            frames_with_boxes += 1

        # Heartbeat
        if frame_index % HEARTBEAT_EVERY == 0:
            log(f"[HEARTBEAT] frames={frame_index} boxes_frames={frames_with_boxes} empty={empty_frames}", "blue")

        # Safety: warn if *really* empty
        if empty_frames >= MAX_EMPTY_FRAMES_WARN and frames_with_boxes == 0:
            log("[WARN] Many frames with no detections — check classes/conf or input file.", "yellow")

        if boxes is None or len(boxes) == 0:
            frame_index += 1
            continue

        # Iterate detections
        for box in boxes:
            if box.id is None:
                continue

            tid = int(box.id)
            frame_counts[tid] = frame_counts.get(tid, 0) + 1

            # Only classify once per track after stability
            if frame_counts[tid] < MIN_FRAMES or tid in seen_cars:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
            x2, y2 = min(w, x2 + PAD), min(h, y2 + PAD)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Timestamp estimate from frame_index and approx_fps
            det_time = start_time_utc + timedelta(seconds=(frame_index / max(approx_fps, 1.0)))

            # CLIP classification
            label = what_is_this_car(crop)
            make        = label.get("make", "Unknown")
            model_name  = label.get("model", "Unknown")
            color_name  = label.get("color", "Unknown")
            make_conf   = float(label.get("make_conf", 0.0))
            model_conf  = float(label.get("model_conf", 0.0))
            mean_conf   = (make_conf + model_conf) / 2.0

            is_uncertain = (
                make == "Unknown"
                or model_name == "Unknown"
                or color_name == "Unknown"
                or mean_conf < UNCERTAIN_THRESHOLD
            )

            log(f"[CAR] id={tid} t={det_time.isoformat()}  {color_name} {make} {model_name}  (mean={mean_conf:.2f})", "cyan")

            if is_uncertain:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(UNCERTAIN_DIR, f"car_{tid}_{ts}.jpg")
                try:
                    cv2.imwrite(save_path, crop)
                    log(f"[UNCERTAIN] snapshot → {save_path}", "yellow")
                except Exception as e:
                    log(f"[UNCERTAIN] failed to save snapshot: {e}", "red")

            results.append({
                "id": tid,
                "utc_time": det_time.isoformat(),
                "make": make,
                "model": model_name,
                "color": color_name,
                "make_conf": make_conf,
                "model_conf": model_conf,
                "mean_conf": mean_conf,
                "uncertain": is_uncertain
            })
            seen_cars.add(tid)

        frame_index += 1

    # If NOTHING had boxes, attempt a tiny fallback sanity run on sampled frames
    if frames_with_boxes == 0:
        log("[FALLBACK] No boxes during track(); sampling a few frames with predict()...", "yellow")
        # sample every Nth frame
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(total // 20, 1)  # ~20 samples
            idx = 0
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                preds = model.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, verbose=False)
                if preds and len(preds) > 0 and len(preds[0].boxes) > 0:
                    log(f"[FALLBACK] predict() found {len(preds[0].boxes)} boxes at frame {idx}", "green")
                    break
                idx += step
            cap.release()
        else:
            log("[FALLBACK] Could not open video for sampling.", "red")

    # Save results JSON
    base_noext = os.path.splitext(os.path.basename(video_path))[0]
    out_json = os.path.join(RESULTS_DIR, f"{base_noext}_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "video": os.path.basename(video_path),
            "start_time_utc": start_time_utc.isoformat(),
            "cars": results,
            "metrics": {
                "frames_total_est": frame_index,
                "frames_with_boxes": frames_with_boxes,
                "empty_frames": empty_frames,
                "runtime_sec": round(time.time() - start, 2)
            }
        }, f, indent=2)
    log(f"[DONE] Results saved → {out_json}", "green")

    # Move video + sidecar meta to processed/
    try:
        shutil.move(video_path, os.path.join(PROCESSED_DIR, os.path.basename(video_path)))
        if os.path.exists(meta_path):
            shutil.move(meta_path, os.path.join(PROCESSED_DIR, os.path.basename(meta_path)))
    except Exception as e:
        log(f"[WARN] Move to processed failed: {e}", "yellow")

    # Recheck unlabeled after each video if requested
    if RECHECK_AFTER_EACH_VIDEO:
        _ = recheck_uncertain_and_cleanup()


# =========================
# Main single-file / worker loop
# =========================
if single_file_mode:
    # Process just one file passed in argv and exit
    try:
        process_video(single_file_mode)
    except Exception as e:
        log(f"[ERROR] Failed on {single_file_mode}: {e}", "red")
    sys.exit(0)

# Queue worker loop
log("=========================================", "cyan")
log("[AI WORKER] Watching ./recordings for new chunks...", "cyan")
log("=========================================", "cyan")

while True:
    mp4_files = sorted(glob.glob(os.path.join(WATCH_DIR, "*.mp4")))
    if not mp4_files:
        log("[WAIT] No new files. Sleeping 5s...", "blue")
        time.sleep(5)
        continue

    for video_file in mp4_files:
        # Skip if already in processed directory
        if os.path.exists(os.path.join(PROCESSED_DIR, os.path.basename(video_file))):
            continue
        try:
            process_video(video_file)
        except Exception as e:
            log(f"[ERROR] Failed on {video_file}: {e}", "red")
            # do not delete the file; leave it for manual inspection
    log("[LOOP] Waiting for next file...", "blue")
    time.sleep(5)
