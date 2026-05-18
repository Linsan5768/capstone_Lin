# demo.py — Cattle hip-height with YOLO11n-seg + AprilTag + manual ground + CSV logging
# Enhanced version:
# - DepthAI v3 (OAK) direct capture: --source oak --oak_w/--oak_h/--fps
# - Manual ground calibration coordinate correction: window coordinates -> image coordinates (cv2.getWindowImageRect)
# - Adjustable display window: --win_w/--win_h/--fullscreen + Hotkeys [ ] + - r f
# - SORT tracking id (uses your src.external.sort.Sort)

import argparse, time, os, csv
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# ---- Your SORT path (keep) --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.external.sort import Sort

# ---- DepthAI (OAK) Optional Import (v3 API) ----------------------------------------
try:
    import depthai as dai
    HAVE_DAI = True
except Exception:
    dai = None
    HAVE_DAI = False

# ---------------- OAK capture (DepthAI v3) ------------------------------------
class OakCapture:
    """
    DepthAI 2.x capturer (compatible with cv2.VideoCapture interface)
    - Uses ColorCamera.preview BGR output
    - Non-blocking queue tryGet(), allows window to appear before the first frame
    """
    def __init__(self, width=1280, height=720, fps=30):
        if not HAVE_DAI:
            raise RuntimeError("depthai not installed. Please pip install depthai first")

        self.width, self.height, self.fps = int(width), int(height), int(fps)

        # --- Pipeline construction ---
        self.pipeline = dai.Pipeline()

        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(self.width, self.height)          # Get preview directly, BGR
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(self.fps)
        cam.setPreviewKeepAspectRatio(False)                  # Disable black bars, output at set resolution

        xout = self.pipeline.createXLinkOut()
        xout.setStreamName("preview")
        cam.preview.link(xout.input)

        # --- Device and output queue ---
        self.device = dai.Device(self.pipeline)               # 2.x: Constructing Device starts the pipeline
        # Reserving a multi-frame pool can reduce frame drop probability
        self.q = self.device.getOutputQueue(name="preview", maxSize=4, blocking=False)

        # Record the actual output dimensions (some firmware aligns to even numbers)
        probe = self.q.tryGet()
        if probe is not None:
            frm = probe.getCvFrame()
            self.out_h, self.out_w = frm.shape[:2]
        else:
            self.out_w, self.out_h = self.width, self.height

        print(f"[INFO] Using OAK via DepthAI 2.x preview {self.out_w}x{self.out_h} @ {self.fps} FPS")

    def isOpened(self):
        return True

    def read(self):
        pkt = self.q.tryGet()
        if pkt is None:
            return False, None
        frame = pkt.getCvFrame()  # BGR np.ndarray
        return True, frame

    def set(self, prop, value):
        # To be compatible with cv2.VideoCapture interface, just return False (modification not supported at runtime)
        return False

    def release(self):
        try:
            self.device.close()
        except Exception:
            pass
   

# ---------------- Utils --------------------------------------------------------
def open_cam(src):
    for backend in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
        cap = cv2.VideoCapture(src, backend)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(src)

def now_s(): return time.monotonic()

def put_text(img, txt, xy, color=(255,255,255), scale=0.7, th=2):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), th+2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)

# ---------------- AprilTag (aruco) ---------------------------------------------
def init_apriltag_36h11():
    tag_ok, aruco, dic, params = False, None, None, None
    if not hasattr(cv2, "aruco"):
        return tag_ok, aruco, dic, params
    aruco = cv2.aruco
    if not hasattr(aruco, "DICT_APRILTAG_36h11"):
        return tag_ok, None, None, None
    dic = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

    # handle both APIs
    try:
        params = aruco.DetectorParameters_create()  # old API
        detector = None
    except AttributeError:
        params = aruco.DetectorParameters()  # new API
        detector = aruco.ArucoDetector(dic, params)
    tag_ok = True
    return tag_ok, aruco, dic, (params, detector)


def detect_two_tags(frame, aruco, dic, param_pack, prefer_ids=None):
    params, detector = param_pack
    vis = frame.copy()
    centers, used_ids = [], []
    tag_count = 0
    all_ids = []

    try:
        # Detect markers using appropriate API
        if detector is not None:
            corners, ids, _ = detector.detectMarkers(frame)
        else:
            corners, ids, _ = aruco.detectMarkers(frame, dic, parameters=params)

        if ids is not None and len(ids) > 0:
            tag_count = len(ids)
            for i, c in enumerate(corners):
                pts = c[0]
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                centers.append((cx, cy))
                used_ids.append(int(ids[i][0]))
                cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 255, 255), 2)
                put_text(vis, f"id={used_ids[-1]}", (int(cx)+6, int(cy)-6), (0,255,255), 0.6, 2)

            all_ids = used_ids[:]

            # Prefer specific IDs
            if prefer_ids and len(prefer_ids) == 2:
                pair = [centers[used_ids.index(pid)] for pid in prefer_ids if pid in used_ids]
                if len(pair) == 2:
                    (c1x, c1y), (c2x, c2y) = pair
                    cv2.circle(vis, (int(c1x), int(c1y)), 5, (0,255,255), -1)
                    cv2.circle(vis, (int(c2x), int(c2y)), 5, (0,255,255), -1)
                    cv2.line(vis, (int(c1x), int(c1y)), (int(c2x), int(c2y)), (0,255,255), 2)
                    return True, pair, prefer_ids, vis, tag_count, all_ids

            # Automatic vertical pairing
            best = None
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    (x1, y1), (x2, y2) = centers[i], centers[j]
                    vy, vx = abs(y2 - y1), abs(x2 - x1)
                    score = vy - 0.4 * vx
                    if best is None or score > best[0]:
                        best = (score, (i, j))
            if best:
                i, j = best[1]
                id_pair = (used_ids[i], used_ids[j])
                c_pair = (centers[i], centers[j])
                (c1x, c1y), (c2x, c2y) = c_pair
                cv2.circle(vis, (int(c1x), int(c1y)), 5, (0,255,255), -1)
                cv2.circle(vis, (int(c2x), int(c2y)), 5, (0,255,255), -1)
                cv2.line(vis, (int(c1x), int(c1y)), (int(c2x), int(c2y)), (0,255,255), 2)
                return True, c_pair, id_pair, vis, tag_count, all_ids
    except Exception as e:
        print(f"[WARN] AprilTag detection failed: {e}")
    return False, None, None, vis, tag_count, all_ids

def px_per_cm_from_two_vertical_centers(c1, c2, known_cm=100.0):
    vy = abs(c2[1]-c1[1])
    if vy <= 1e-6: return None
    return vy / float(known_cm)

# ---------------- Ground line --------------------------------------------------
def estimate_ground_y_auto(frame, roi_ratio=0.45):
    H, W = frame.shape[:2]
    y0 = int(H*(1.0 - roi_ratio))
    roi = frame[y0:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    m = float(np.mean(blur))
    lo = max(20, m*0.4); hi = min(200, m*1.6)
    edges = cv2.Canny(blur, lo, hi)
    klen = max(15, W//30)
    kernel = np.ones((1, klen), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=W//3, maxLineGap=W//10)
    if lines is None:
        return H-1, 0.0
    cands = []
    for ln in lines:
        x1,y1,x2,y2 = ln[0]
        y1 += y0; y2 += y0
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang < 5 or ang > 175:
            length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            ymid = 0.5*(y1+y2)
            length_score = length / W
            pos_score = 1.0 - ((ymid - y0) / max(1, H-y0))
            score = 0.4*length_score + 0.6*pos_score
            cands.append((ymid, score, length))
    if not cands:
        return H-1, 0.0
    cands.sort(key=lambda x: x[1], reverse=True)
    topk = cands[:max(3, int(len(cands)*0.3))]
    totw = sum(s*L for _,s,L in topk)
    y = int(sum(y*s*L for y,s,L in topk) / max(1e-6, totw))
    conf = float(np.clip((totw / (W*roi_ratio*H))*2.0, 0.0, 1.0))
    y = int(np.clip(y, y0, H-1))
    return y, conf

# ---------------- Cow Hip-Height helpers (LEFT/RIGHT ROI) ---------------------
def hip_from_mask_side(mask, box, side):
    """
    Take the left/right 1/3 ROI within the bbox, find the highest point (minimum y) of the mask in that ROI.
    Returns: hip_y(int or None), roi_box(tuple), density(int: total foreground pixels in ROI)
    """
    x1,y1,x2,y2 = box
    w = max(1, x2 - x1)
    if side == "right":
        rx1, rx2 = x1 + 2*w//3, x2
    else:
        rx1, rx2 = x1, x1 + w//3
    roi_mask = mask[y1:y2, rx1:rx2]
    if roi_mask.size == 0:
        return None, (rx1,y1,rx2,y2), 0

    ys = []
    # Sample multiple columns, find the first foreground pixel (highest point) in each column
    cols = np.linspace(0, roi_mask.shape[1]-1, 40, dtype=int)
    for c in cols:
        col = np.where(roi_mask[:, c] > 0)[0]
        if len(col):
            ys.append(col[0])

    if not ys:
        return None, (rx1,y1,rx2,y2), int(roi_mask.sum())

    hip_y = y1 + int(np.median(ys))
    density = int(roi_mask.sum())
    return hip_y, (rx1,y1,rx2,y2), density

def _choose_butt_side(hip_r, dens_r, hip_l, dens_l):
    """
    Side selection:
      1) The side with more foreground pixels (higher density) in the ROI is considered the hip;
      2) If densities are equal and both sides have points, take the higher side (smaller y);
      3) Otherwise, use whichever side has a point; default to right if neither does.
    """
    if dens_r > dens_l: return "right"
    if dens_l > dens_r: return "left"
    if hip_r is not None and hip_l is not None:
        return "right" if hip_r < hip_l else "left"
    if hip_r is not None: return "right"
    if hip_l is not None: return "left"
    return "right"

# ====== Independent Hip CSV (Keeps your original heights_*.csv unchanged) =========================
class HipCsvLogger:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.path = os.path.join("logs", f"hip_heights_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "timestamp_iso", "hip_height_cm", "px_per_cm", "ground_y",
            "dens_right", "dens_left",
            "tag_ids_pair", "tag_count"
        ])
        self._f.flush()

    def log(self, hip_cm, px_per_cm, ground_y, dens_right, dens_left, ids_pair, tag_count):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._w.writerow([
            ts,
            f"{hip_cm:.2f}" if hip_cm is not None else "",
            f"{px_per_cm:.6f}" if px_per_cm else "",
            int(ground_y) if ground_y is not None else "",
            int(dens_right), int(dens_left),
            f"{list(ids_pair)}" if ids_pair else "",
            int(tag_count) if tag_count is not None else 0
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

# ---------------- YOLO helpers -------------------------------------------------
def load_yolo(weights):
    try:
        return YOLO(weights)
    except Exception as e:
        print(f"[WARN] load {weights} failed -> {e}")
        for fb in ("yolo11n.pt", "yolov8n-seg.pt", "yolov8n.pt"):
            try:
                print("[INFO] trying fallback:", fb)
                return YOLO(fb)
            except Exception as e2:
                print("[WARN]", fb, "->", e2)
        raise SystemExit("No valid weights could be loaded.")

def get_cow_class_ids(model):
    names = model.model.names if hasattr(model, "model") else model.names
    if isinstance(names, dict):
        return [i for i, n in names.items() if str(n).lower() == "cow"]
    if isinstance(names, (list, tuple)):
        return [i for i, n in enumerate(names) if str(n).lower() == "cow"]
    return [19]  # cow=19 in common datasets; adjust according to your weights if it doesn't match

def top_of_head_y_from_mask(mask, box):
    x1,y1,x2,y2 = box
    if mask is None: return None
    m = mask[y1:y2, x1:x2]
    if m.size == 0: return None
    ys = np.where(m>0)[0]
    if ys.size == 0: return None
    return y1 + int(np.min(ys))

def process_frame(frame, model, cow_ids, conf_thr, tracker):
    H, W = frame.shape[:2]
    r = model(frame, classes=cow_ids, conf=conf_thr, iou=0.5,
              max_det=1, verbose=False)[0]
    out = {"ok": False, "id": None, "box": None, "mask": None, "conf": 0.0, "head_y": None}
    if r.boxes is None or len(r.boxes)==0:
        return out
    # Select the box with the highest confidence and largest area
    best_i, best_c, best_area = 0, 0.0, 0
    for i, b in enumerate(r.boxes.xyxy):
        c = float(r.boxes.conf[i])
        x1,y1,x2,y2 = map(int, b[:4])
        area = max(0, x2-x1)*max(0, y2-y1)
        if c > best_c and area > best_area:
            best_c, best_i, best_area = c, i, area
    x1,y1,x2,y2 = map(int, r.boxes.xyxy[best_i][:4])
    conf = float(r.boxes.conf[best_i])
    dets = np.array([[x1, y1, x2, y2, conf]])
    tracks = tracker.update(dets)
    track_id = int(tracks[0][-1]) if len(tracks) > 0 else None

    mask = None
    if getattr(r, "masks", None) is not None and len(r.masks):
        m = r.masks.data[best_i].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    head_y = top_of_head_y_from_mask(mask, (x1,y1,x2,y2)) if mask is not None else None
    out.update(ok=True, id=track_id, box=(x1,y1,x2,y2), mask=mask, conf=best_c, head_y=head_y)
    return out

# ---------------- Main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="oak", help="Camera index/video path, or 'oak' to use DepthAI OAK (v3)")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--weights", default="yolo11n-seg.pt", help="Prioritize YOLO11n-seg; auto-fallback on failure")
    ap.add_argument("--known_cm", type=float, default=100.0, help="Preset distance between the centers of two AprilTags (cm)")
    ap.add_argument("--tag_ids", type=int, nargs=2, default=None, help="Preferred tag id pair, e.g., --tag_ids 0 1")
    ap.add_argument("--keep_scale_s", type=float, default=2.0, help="Calibration hold time in seconds when only one tag is visible")
    ap.add_argument("--fps", type=int, default=30, help="Target processing FPS")
    ap.add_argument("--pre_post_frames", type=int, default=15, help="Accumulated frames before/after trigger line (±N)")
    # OAK-only
    ap.add_argument("--oak_w", type=int, default=1280, help="OAK preview width")
    ap.add_argument("--oak_h", type=int, default=720,  help="OAK preview height")
    # Display related
    ap.add_argument("--win_w", type=int, default=0, help="Window display width (0=follow source)")
    ap.add_argument("--win_h", type=int, default=0, help="Window display height (0=follow source)")
    ap.add_argument("--fullscreen", action="store_true", help="Start window in fullscreen")
    args = ap.parse_args()

    # Open video source
    if str(args.source).lower() == "oak":
        if not HAVE_DAI:
            raise SystemExit("DepthAI not installed: pip install depthai")
        cap = OakCapture(args.oak_w, args.oak_h, args.fps)
        opened = cap.isOpened()
        src_w, src_h = args.oak_w, args.oak_h
    else:
        source = int(args.source) if str(args.source).isdigit() else args.source
        cap = open_cam(source)
        opened = cap.isOpened()
        if not opened:
            raise SystemExit(f"Cannot open source {args.source}")
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        # Try to get source resolution
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # SORT tracker
    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

    model = load_yolo(args.weights)
    cow_ids = get_cow_class_ids(model)

    tag_ok, aruco, dic, params = init_apriltag_36h11()

    # === CSV logging ===
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", f"heights_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    csv_file, csv_writer = None, None

    def ensure_csv():
        nonlocal csv_file, csv_writer
        if csv_writer is None:
            csv_file = open(csv_path, "a", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "timestamp_iso", "height_cm_median", "frames_used",
                "px_per_cm", "scale_state", "tag_ids_pair", "tag_count",
                "ground_mode", "ground_y", "trigger_x"
            ])
            csv_file.flush()

    def csv_log(height_cm_median, frames_used, px_per_cm, scale_from_two_tags,
                ids_pair, tag_count, ground_mode, ground_y, trigger_x):
        ensure_csv()
        scale_state = "LIVE" if scale_from_two_tags else ("CACHE" if px_per_cm else "NONE")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([
            ts,
            f"{height_cm_median:.1f}",
            frames_used,
            f"{px_per_cm:.6f}" if px_per_cm else "",
            scale_state,
            f"{list(ids_pair)}" if ids_pair else "",
            tag_count,
            ground_mode,
            ground_y,
            trigger_x if trigger_x is not None else ""
        ])
        csv_file.flush()

    # Independent hip CSV
    hip_logger = HipCsvLogger()

    # State
    scale_px_per_cm = None
    last_scale_t = -1e9
    scale_from_two_tags = False
    trigger_x = None

    # Ground line: auto/manual
    ground_mode = "auto"     # "auto" or "manual"
    ground_y_manual = None
    clicks = []

    # Pass-through detection (constant speed pass) buffer
    pre_buf = deque(maxlen=args.pre_post_frames)
    post_buf, active, last_side = [], False, None

    # ---- Window & Display Size Control -----------------------------------------------
    win = "Cow Hip Height (YOLO + AprilTag + OAK v3)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    init_w = args.win_w if args.win_w > 0 else (src_w if src_w else 1280)
    init_h = args.win_h if args.win_h > 0 else (src_h if src_h else 720)
    cur_win_w, cur_win_h = int(init_w), int(init_h)
    cv2.resizeWindow(win, cur_win_w, cur_win_h)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Record the dimensions of the last frame (for mouse coordinate conversion)
    last_H, last_W = int(src_h if src_h else 720), int(src_w if src_w else 1280)

    def show_img(img):
        nonlocal cur_win_w, cur_win_h
        if img is None:
            return
        if img.shape[1] != cur_win_w or img.shape[0] != cur_win_h:
            disp = cv2.resize(img, (cur_win_w, cur_win_h), interpolation=cv2.INTER_LINEAR)
        else:
            disp = img
        cv2.imshow(win, disp)

    # ---- Mouse callback: two clicks to set manual ground line (corrects for coordinate scaling) ----------------------
    def on_mouse(event, x, y, flags, param):
        nonlocal ground_mode, ground_y_manual, clicks, last_H, last_W, win
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                _, _, disp_w, disp_h = cv2.getWindowImageRect(win)
            except Exception:
                disp_w, disp_h = 0, 0
            if disp_w and disp_h:
                xi = int(round(x * last_W / float(disp_w)))
                yi = int(round(y * last_H / float(disp_h)))
            else:
                xi, yi = int(x), int(y)
            xi = max(0, min(last_W - 1, xi))
            yi = max(0, min(last_H - 1, yi))
            clicks.append((xi, yi))
            print(f"[DEBUG] click image-coord: x={xi}, y={yi} (frame {last_W}x{last_H})")
            if len(clicks) >= 2:
                ymean = int(0.5 * (clicks[-1][1] + clicks[-2][1]))
                ymean = max(0, min(last_H - 1, ymean))
                ground_y_manual = ymean
                ground_mode = "manual"
                clicks.clear()
                print(f"[INFO] Ground set MANUAL y={ground_y_manual}")

    cv2.setMouseCallback(win, on_mouse)

    t_prev = now_s()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                vis = np.zeros((last_H, last_W, 3), dtype=np.uint8)
                put_text(vis, "Waiting for frames...", (20, 50), (0,255,255), 0.9, 2)
                show_img(vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')): break
                if k in (ord(']'), ord('='), ord('+')):
                    cur_win_w = int(cur_win_w * 1.1); cur_win_h = int(cur_win_h * 1.1)
                    cv2.resizeWindow(win, cur_win_w, cur_win_h)
                if k in (ord('['), ord('-'), ord('_')):
                    cur_win_w = max(320, int(cur_win_w * 0.9))
                    cur_win_h = max(180, int(cur_win_h * 0.9))
                    cv2.resizeWindow(win, cur_win_w, cur_win_h)
                if k == ord('r'):
                    cur_win_w, cur_win_h = (last_W, last_H)
                    cv2.resizeWindow(win, cur_win_w, cur_win_h)
                if k in (ord('f'), ord('F')):
                    fs = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if fs == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
                continue

            # Frame rate limit
            t_now = now_s()
            if (t_now - t_prev) < 1.0/max(1,args.fps):
                time.sleep(max(0, 1.0/args.fps - (t_now - t_prev)))
            t_prev = now_s()

            H, W = frame.shape[:2]
            last_H, last_W = H, W
            vis = frame.copy()

            # (1) AprilTag calibration
            two_ok, centers, ids_pair, vis_tag, tag_count, all_ids = (False, None, None, frame, 0, [])
            if tag_ok:
                two_ok, centers, ids_pair, vis_tag, tag_count, all_ids = detect_two_tags(frame, aruco, dic, params, args.tag_ids)
                vis = vis_tag
                if two_ok:
                    (x1c,y1c),(x2c,y2c) = centers
                    px_per_cm = px_per_cm_from_two_vertical_centers((x1c,y1c), (x2c,y2c), args.known_cm)
                    if px_per_cm:
                        scale_px_per_cm = px_per_cm
                        last_scale_t = now_s()
                        scale_from_two_tags = True
                        trigger_x = int(0.5*(x1c+x2c))

            # Calibration hold
            if scale_px_per_cm is not None and (now_s() - last_scale_t) <= args.keep_scale_s:
                pass
            else:
                if not two_ok:
                    scale_px_per_cm = None
                    scale_from_two_tags = False

            # (2) Ground line
            if ground_mode == "manual" and ground_y_manual is not None:
                ground_y, gconf = int(ground_y_manual), 1.0
            else:
                ground_y, gconf = estimate_ground_y_auto(frame)

            # (3) Detection + Hip-height (highest point in buttock ROI)
            det = process_frame(frame, model, cow_ids, args.conf, tracker)
            height_cm = None
            if det["ok"]:
                x1,y1,x2,y2 = det["box"]; conf = det["conf"]
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                if det["mask"] is not None:
                    over = vis.copy(); over[det["mask"]>0] = (0,200,0)
                    vis = cv2.addWeighted(vis, 0.7, over, 0.3, 0)
                head_y = det["head_y"]

                # --- HIP: Left/Right 1/3 ROI, take the highest point of the mask in that ROI ---
                hip_y = None
                hip_cm = None
                if det["mask"] is not None:
                    hip_y_r, roi_r, dens_r = hip_from_mask_side(det["mask"], (x1,y1,x2,y2), "right")
                    hip_y_l, roi_l, dens_l = hip_from_mask_side(det["mask"], (x1,y1,x2,y2), "left")

                    side = _choose_butt_side(hip_y_r, dens_r, hip_y_l, dens_l)
                    hip_y  = hip_y_r if side == "right" else hip_y_l
                    roi_box = roi_r if side == "right" else roi_l

                    if hip_y is not None:
                        rx1, ry1, rx2, ry2 = roi_box
                        cx_roi = (rx1 + rx2) // 2
                        cv2.circle(vis, (cx_roi, int(hip_y)), 5, (0,0,255), -1)  # Red dot at hip-side ROI center x & highest point y
                        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255,0,0), 2) # Visualize ROI used (can be deleted)

                        px_h = max(0, int(ground_y) - int(hip_y))
                        hip_cm = (px_h / scale_px_per_cm) if scale_px_per_cm else None
                        height_cm = hip_cm if hip_cm is not None else height_cm

                        if hip_cm is not None:
                            put_text(vis, f"Hip: {hip_cm:.1f} cm  [{side}]", (10, 120), (255,255,0), 0.8, 2)
                        else:
                            put_text(vis, f"Hip: -- cm (no scale)  [{side}]", (10, 120), (255,255,0), 0.8, 2)

                        hip_logger.log(
                            hip_cm=hip_cm,
                            px_per_cm=scale_px_per_cm,
                            ground_y=ground_y,
                            dens_right=dens_r,
                            dens_left=dens_l,
                            ids_pair=ids_pair,
                            tag_count=tag_count
                        )

                # Fallback: If the highest hip point is not found in this frame, fall back to the head point (without interrupting the flow)
                if height_cm is None and head_y is not None:
                    cx = (x1 + x2) // 2
                    # cv2.circle(vis, (cx, int(head_y)), 5, (0,0,255), -1)
                    px_h = max(0, int(ground_y) - int(head_y))
                    height_cm = (px_h / scale_px_per_cm) if scale_px_per_cm else None

                # Trigger line: center x of the two tags
                if trigger_x is not None:
                    cx_line = (x1 + x2) // 2
                    cv2.line(vis, (trigger_x, 0), (trigger_x, H-1), (255,0,255), 2)
                    side_trig = 'L' if cx_line < trigger_x else 'R'
                    if last_side and side_trig != last_side and not active:
                        active = True; post_buf = []
                    last_side = side_trig

                # Buffer and take median (uniformly use height_cm: prioritize hip)
                val_cm = height_cm if (height_cm is not None and conf >= args.conf and scale_px_per_cm) else None
                if val_cm is not None:
                    pre_buf.append(val_cm)
                    if active:
                        post_buf.append(val_cm)
                        if len(post_buf) >= args.pre_post_frames:
                            vals = list(pre_buf) + list(post_buf)
                            med = float(np.median(vals)) if vals else float('nan')

                            put_text(vis, f"PASS -> median {med:.1f} cm (frames={len(vals)})",
                                     (10, 60), (0,255,255), 0.8, 2)

                            csv_log(
                                height_cm_median=med,
                                frames_used=len(vals),
                                px_per_cm=scale_px_per_cm,
                                scale_from_two_tags=scale_from_two_tags,
                                ids_pair=ids_pair,
                                tag_count=tag_count,
                                ground_mode=("MANUAL" if ground_mode=="manual" else f"AUTO({gconf:.2f})"),
                                ground_y=ground_y,
                                trigger_x=trigger_x
                            )
                            if scale_px_per_cm:
                                print(f"[PASS] height={med:.1f} cm | frames={len(vals)} | "
                                      f"px/cm={scale_px_per_cm:.6f} | tags={ids_pair} "
                                      f"count={tag_count} | ground={ground_y} {ground_mode}")
                            else:
                                print(f"[PASS] height={med:.1f} cm | frames={len(vals)} | "
                                      f"px/cm=-- | tags={ids_pair} count={tag_count} | "
                                      f"ground={ground_y} {ground_mode}")

                            active = False; post_buf = []; pre_buf.clear()

                # HUD
                tip = (f"ID {det['id']} | Height {height_cm:.1f} cm | conf {conf*100:.1f}%"
                       if height_cm is not None else
                       f"ID {det['id']} | Height -- cm | conf {conf*100:.1f}%")
                put_text(vis, tip, (10, 30), (255,255,255), 0.8, 2)

            # Unified HUD: Ground line / Status bar / Help
            cv2.line(vis, (0, ground_y), (W-1, ground_y), (0,255,255), 2)
            stat = f"px/cm: {scale_px_per_cm:.4f}" if scale_px_per_cm else "px/cm: --"
            if tag_ok:
                live = "LIVE" if scale_from_two_tags else ("CACHE" if scale_px_per_cm else "NONE")
                stat += f"  tag:{live}  count:{tag_count}  ids:{all_ids[:4]}"
            else:
                stat += "  (No AprilTag support: pip install opencv-contrib-python)"
            gmode = "MANUAL" if ground_mode=="manual" else f"AUTO({gconf:.2f})"
            stat += f"  ground:{gmode}"
            put_text(vis, stat, (10, H-12), (200,200,255), 0.7, 2)
            if ground_mode == "manual" and ground_y_manual is None:
                put_text(vis, "Manual ground: click two points on the ground line",
                         (10, 120), (60, 220, 255), 0.8, 2)
            put_text(vis, "Keys: g=manual ground (2 clicks) | a=auto ground | f=fullscreen | +/-=[ ] scale | r=reset | q/ESC=quit",
                     (10, 90), (180,220,255), 0.6, 2)

            # Display (with scaling)
            show_img(vis)

            # Keyboard
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')): break
            if k == ord('g'):
                clicks.clear(); ground_mode = "manual"; ground_y_manual = None
                print("[INFO] Click two points on the ground line in the preview window...")
            if k == ord('a'):
                ground_mode = "auto"; ground_y_manual = None
                print("[INFO] Ground mode -> AUTO")
            if k in (ord(']'), ord('='), ord('+')):
                cur_win_w = int(cur_win_w * 1.1); cur_win_h = int(cur_win_h * 1.1)
                cv2.resizeWindow(win, cur_win_w, cur_win_h)
            if k in (ord('['), ord('-'), ord('_')):
                cur_win_w = max(320, int(cur_win_w * 0.9))
                cur_win_h = max(180, int(cur_win_h * 0.9))
                cv2.resizeWindow(win, cur_win_w, cur_win_h)
            if k == ord('r'):
                cur_win_w, cur_win_h = (W, H)
                cv2.resizeWindow(win, cur_win_w, cur_win_h)
            if k in (ord('f'), ord('F')):
                fs = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if fs == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
        try:
            hip_logger.close()
        except Exception:
            pass
        print(f"[INFO] CSV path: {csv_path}")

if __name__ == "__main__":
    main()