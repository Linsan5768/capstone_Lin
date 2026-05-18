#!/usr/bin/env python3
# oak_depth_debug.py — OAK RGB+Depth visual debugger
# Shows RGB, depth colormap, RANSAC ground plane residuals,
# plane-guard mask (valid head-depth region), and a dedicated
# GROUND INLIER MASK overlay. Optional YOLO head-top + 3D height.




#python -m pip install -U depthai opencv-contrib-python ultralytics numpy

#USB2-friendly, no distance limit
# DEPTHAI_FORCE_USB2=1 \
# python3 oak_depth_debug.py \
# --fps 25 --oak_rgb_res 720p --oak_stereo_res 400p \
# --oak_lr_check --oak_subpixel --conf_thr 180

#Hotkeys:
#1 depth | 2 residual | 3 guard mask | 4 ground inlier mask | d HUD | m mosaic | p probe | S snapshot | +/-/[/] zoom | f fullscreen | space pause | q/ESC quit

import os, time, math, argparse
from pathlib import Path
import numpy as np
import cv2
import depthai as dai
import csv
try:
    from smart_cattle_detector import SmartCattleDetector
    HAVE_SMART = True
except Exception:
    HAVE_SMART = False

# --- YOLO fallback imports ---
try:
    from ultralytics import YOLO as _ULTRA_YOLO
except Exception:
    _ULTRA_YOLO = None

# --- CSV Logger for Depth Cattle Height Measurements ---
class CattleDepthCsvLogger:
    
    def __init__(self):
        """Initialize CSV logger with timestamped filename and header."""
        os.makedirs("logs", exist_ok=True)
        self.path = os.path.join("logs", f"depth_cattle_heights_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "timestamp_iso", "cattle_id", "hip_height_cm", "", "camera_height_m", 
            "confidence", "frame_id", "Zp_plane_depth", "Z_head_depth", 
            "guard_check_ok", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
        ])
        self._f.flush()
        print(f"[CSV] Logging to: {self.path}")

    def log(self, cattle_id, hip_height_cm, camera_height_m, confidence, frame_id, 
            Zp_plane_depth, Z_head_depth, guard_check_ok, bbox):
        """Log a cattle measurement row."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        x1, y1, x2, y2 = bbox if bbox is not None else (None, None, None, None)
        self._w.writerow([
            ts,
            int(cattle_id) if cattle_id is not None else "",
            f"{hip_height_cm:.2f}" if hip_height_cm is not None else "",
            "",  # Blank column for separation
            f"{camera_height_m:.3f}" if camera_height_m is not None else "",
            f"{confidence:.3f}" if confidence is not None else "",
            int(frame_id) if frame_id is not None else "",
            f"{Zp_plane_depth:.3f}" if Zp_plane_depth is not None else "",
            f"{Z_head_depth:.3f}" if Z_head_depth is not None else "",
            "True" if guard_check_ok else "False" if guard_check_ok is not None else "",
            int(x1) if x1 is not None else "",
            int(y1) if y1 is not None else "",
            int(x2) if x2 is not None else "",
            int(y2) if y2 is not None else ""
        ])
        self._f.flush()

    def log_separator(self, reason="AUTO"):
        """Log a separator row to mark new cattle."""
        self._w.writerow([f"NEW_CATTLE_{reason}"] + [""] * 13)
        self._f.flush()

    def close(self):
        """Close the CSV file."""
        try:
            self._f.close()
        except Exception:
            pass

 # --- Demo-style hip-top finder: top-most edge point inside ROI ---
def _hip_top_from_mask_demo(mask: np.ndarray, roi_box: tuple):
    """Return (x,y) of the top-most mask edge within ROI, else None."""
    x1, y1, x2, y2 = map(int, roi_box)
    H, W = mask.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    hip_region = mask[y1:y2, x1:x2]
    if hip_region.size == 0:
        return None

    # 1) Prefer Canny edges (matches demo.py behavior)
    try:
        edges = cv2.Canny(hip_region, 50, 150)
        edge_pts = [(x1+c, y1+r) for r in range(edges.shape[0])
                               for c in range(edges.shape[1]) if edges[r, c] > 0]
        if edge_pts:
            return min(edge_pts, key=lambda p: p[1])
    except Exception:
        pass

    # 2) Fallback: any mask pixel, pick smallest y
    ys, xs = np.where(hip_region > 0)
    if ys.size == 0:
        return None
    idx = int(np.argmin(ys))
    return (int(x1 + xs[idx]), int(y1 + ys[idx]))

# --- HIP HEIGHT HELPERS (from demo.py) ---
def hip_from_mask_side(mask, box, side):
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
    if dens_r > dens_l: return "right"
    if dens_l > dens_r: return "left"
    if hip_r is not None and hip_l is not None:
        return "right" if hip_r < hip_l else "left"
    if hip_r is not None: return "right"
    if hip_l is not None: return "left"
    return "right"

# ---------------- Ground line (simple 2D auto) ----------------
def estimate_ground_y_auto(frame, roi_ratio=0.45):
    H, W = frame.shape[:2]
    y0 = int(H * (1.0 - roi_ratio))
    roi = frame[y0:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    m = float(np.mean(blur))
    lo = max(20, m * 0.4); hi = min(200, m * 1.6)
    edges = cv2.Canny(blur, lo, hi)
    klen = max(15, W // 30)
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

# ================= Utils =================
def apply_colormap_depth(depth_m, zmin=None, zmax=None):
    dm = depth_m.copy()
    bad = ~np.isfinite(dm) | (dm <= 0)
    if zmin is None:
        zmin = np.percentile(dm[~bad], 5) if np.any(~bad) else 0.5
    if zmax is None:
        zmax = np.percentile(dm[~bad], 95) if np.any(~bad) else 5.0
    zmin = max(0.1, float(zmin))
    zmax = max(zmin + 1e-3, float(zmax))
    dm = np.clip((dm - zmin) / (zmax - zmin), 0, 1)
    dm = (dm * 255).astype(np.uint8)
    color = cv2.applyColorMap(dm, cv2.COLORMAP_TURBO)
    color[bad] = (0, 0, 0)
    return color, (zmin, zmax)

def backproject(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

def depth_sample(depth, u, v, k=5):
    H, W = depth.shape
    u = int(round(u)); v = int(round(v))
    if u < 0 or u >= W or v < 0 or v >= H:
        return None
    r = k // 2
    x1, x2 = max(0, u - r), min(W, u + r + 1)
    y1, y2 = max(0, v - r), min(H, v + r + 1)
    patch = depth[y1:y2, x1:x2]
    vals = patch[(patch > 0) & np.isfinite(patch)]
    return float(np.median(vals)) if vals.size else None

def fit_plane_svd(P):
    P = np.asarray(P, np.float32)
    C = P.mean(0)
    _, _, Vt = np.linalg.svd(P - C, full_matrices=False)
    n = Vt[-1]
    n = n / (np.linalg.norm(n) + 1e-9)
    d = -float(n @ C)
    return n, d

def ransac_plane(pts, iters=400, tau=0.02, up=(0, -1, 0), up_cos=0.3, min_in=120):
    """RANSAC + SVD refit. Returns (n, d, inlier_mask)"""
    N = len(pts)
    pts = np.asarray(pts, np.float32)
    if N < 50:
        return None, None, np.zeros(N, bool)
    up = np.asarray(up, np.float32)
    up /= (np.linalg.norm(up) + 1e-9)

    best_in = []
    best = (None, None)
    for _ in range(iters):
        i, j, k = np.random.randint(0, N, 3)
        if len({i, j, k}) < 3:
            continue
        A, B, C = pts[i], pts[j], pts[k]
        n = np.cross(B - A, C - A)
        ln = np.linalg.norm(n)
        if ln < 1e-6:
            continue
        n /= ln
        if abs(n @ up) < up_cos:
            continue
        d = -float(n @ A)
        dist = np.abs(pts @ n + d)
        inliers = np.where(dist < tau)[0]
        if inliers.size > len(best_in):
            best_in = inliers
            best = (n, d)

    if len(best_in) < min_in:
        n, d = fit_plane_svd(pts)
        inmask = np.zeros(N, bool)
        # Recompute inliers at tau for visualization
        dist = np.abs(pts @ n + d)
        inmask[dist < tau] = True
    else:
        n, d = fit_plane_svd(pts[best_in])
        dist = np.abs(pts @ n + d)
        inmask = dist < tau

    # Ensure normal points "downwards" (positive Y is image-down)
    if n @ up < 0:
        n = -n
        d = -d
    n = n / (np.linalg.norm(n) + 1e-9)
    return n, d, inmask

def plane_intersect_depth(u, v, K, n, d):
    fx, fy, cx, cy = K
    r = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], np.float32)
    r /= (np.linalg.norm(r) + 1e-9)
    denom = float(n @ r)
    if abs(denom) < 1e-6:
        return None
    t = -d / denom
    Z = t * r[2]
    return Z if Z > 0 else None

def head_depth_guard(u, v, depth, K, n, d, guard=0.08, adaptive=True, slope=0.02):
    Zp = plane_intersect_depth(u, v, K, n, d)
    if Zp is None:
        return None, None, False
    Z = depth_sample(depth, u, v, k=5)
    if Z is None:
        return Zp, None, False
    g = max(guard, slope * Zp) if adaptive else guard
    ok = (Z <= Zp - g)
    return Zp, Z, ok

def robust_head_depth(u, v, depth, K, n, d, search_down_px=12, band_half_w=6):
    """
    Try harder to get a valid head depth:
    1) Sample a small vertical rod below (u,v) for up to search_down_px pixels.
    2) Within each row, sample a horizontal band (+/- band_half_w) and take the *minimum valid* depth.
    3) Keep the smallest depth that still passes the plane-guard rule (Z <= Zp - g).

    Returns: Z (meters) or None.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K

    # Plane intersection at (u,v) for guard
    Zp = plane_intersect_depth(u, v, K, n, d)
    if Zp is None:
        return None

    # Adaptive guard (same style as your code)
    def guard_for(Zp_): return max(0.08, 0.02 * Zp_)

    best = None
    for dv in range(0, int(search_down_px) + 1):
        vv = int(round(v + dv))
        if vv < 0 or vv >= H: break
        u1, u2 = int(max(0, u - band_half_w)), int(min(W - 1, u + band_half_w))
        row = depth[vv, u1:u2 + 1]
        valid = (row > 0) & np.isfinite(row)
        if not np.any(valid):
            continue
        Zcand = float(np.nanmin(np.where(valid, row, np.nan)))
        if Zcand <= 0 or not np.isfinite(Zcand):
            continue
        # Guard check against plane at the original (u,v)
        g = guard_for(Zp)
        if Zcand <= Zp - g:
            best = Zcand if (best is None or Zcand < best) else best

    return best

def head_top_from_bbox(depth, x1, y1, x2, y2, search_half_width=10):
    """
    Estimate head top (u,v) from a bbox using depth:
    - Scan from the top (y1) downward near the bbox center,
      pick the first row with valid depth (closest plausible surface).
    Fallback to geometric top-center if no depth found.
    """
    H, W = depth.shape[:2]
    x1 = max(0, min(W-1, int(x1))); x2 = max(0, min(W-1, int(x2)))
    y1 = max(0, min(H-1, int(y1))); y2 = max(0, min(H-1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return ((x1 + x2) // 2, y1)

    xc = (x1 + x2) // 2
    xL = max(x1, xc - search_half_width)
    xR = min(x2, xc + search_half_width)

    # Scan top-down inside the bbox for first valid depth pixel
    for v in range(y1, y2):
        row = depth[v, xL:xR+1]
        valid = (row > 0) & np.isfinite(row)
        if np.any(valid):
            # choose the smallest depth (closest) within the small band
            idx = int(np.nanargmin(np.where(valid, row, np.nan)))
            u = xL + idx
            return (u, v)

    # Fallback: geometric top-center slightly inside the box
    v_top = int(y1 + 0.02 * max(1, (y2 - y1)))
    return (xc, v_top)

def ema_update(old, new, alpha=0.2):
    """Exponential Moving Average helper: alpha in [0..1], higher=snappier."""
    if old is None:
        return new
    return (1.0 - alpha) * old + alpha * new

def get_class_ids(model, target_name: str):
    names = model.model.names if hasattr(model, "model") else getattr(model, "names", None)
    if isinstance(names, dict):
        return [i for i, n in names.items() if str(n).lower() == target_name.lower()]
    if isinstance(names, (list, tuple)):
        return [i for i, n in enumerate(names) if str(n).lower() == target_name.lower()]
    return []

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="oak", choices=["oak", "webcam", "video"], help="choose 'oak' for OAK-D, 'webcam', or 'video' file")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--oak_rgb_res", default="720p", choices=["720p", "1080p", "4k"])
    ap.add_argument("--oak_stereo_res", default="400p", choices=["400p", "480p", "720p"])
    ap.add_argument("--oak_lr_check", action="store_true")
    ap.add_argument("--oak_subpixel", action="store_true")
    ap.add_argument("--conf_thr", type=int, default=180)

    ap.add_argument("--roi_bottom", type=float, default=0.60, help="bottom ROI fraction used to fit ground")
    ap.add_argument("--stride", type=int, default=4, help="sampling stride in pixels")
    ap.add_argument("--ransac_iter", type=int, default=400)
    ap.add_argument("--ransac_tau_cm", type=float, default=2.0)
    ap.add_argument("--min_inliers", type=int, default=120)

    ap.add_argument("--use_smart", action="store_true", default=True, help="use SmartCattleDetector (same detector logic as demo.py)")
    ap.add_argument("--smart_model", default="yolo11n-seg.pt", help="model path for SmartCattleDetector")
    ap.add_argument("--smart_conf", type=float, default=0.30, help="confidence threshold for SmartCattleDetector")
    ap.add_argument("--smart_cows_only", action="store_true", default=False,
                    help="if set, restrict SmartCattleDetector to cows only; by default detects both cows and horses")
    ap.add_argument("--smart_high_precision_mask", action="store_true", default=True,
                    help="enable high-precision mask post-processing (matches demo.py purple mask quality)")
    ap.add_argument("--smart_target", choices=["cow", "horse", "both"], default="both",
                    help="target class for SmartCattleDetector: cow (19), horse (17), or both")
    ap.add_argument("--smart_debug", action="store_true",
                    help="print per-frame Smart detections and draw ALL boxes (class/conf) to debug why nothing shows")
    ap.add_argument("--force_yolo_fallback", action="store_true",
                    help="use internal YOLO seg fallback instead of SmartCattleDetector (for demo-like behaviour)")

    ap.add_argument("--show_2d_ground", action="store_true", help="in webcam/video mode, draw 2D heuristic yellow ground line")
    ap.add_argument("--show_mask", action="store_true",
                    help="draw segmentation mask overlay (purple); OFF by default")
    ap.add_argument("--hold_frames", type=int, default=15,
                    help="keep drawing the last good detection for this many frames when detections temporarily disappear")

    ap.add_argument("--save_dir", default="debug_logs")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--groundline_stride", type=int, default=8,
                help="pixel step for drawing the auto ground redline")
    ap.add_argument("--height_k", type=float, default=0.62,
                    help="scale for bbox-based height; try 0.85–0.95")
    ap.add_argument("--hip_frac", type=float, default=0.33,
                    help="fraction from TOP of ROI/bbox to pick hip point (0..1), e.g., 0.33 ≈ 1/3")
    ap.add_argument("--hip_side", default="auto", choices=["auto", "left", "right"],
                    help="which side of the cow to probe for hip-top; auto chooses the heavier lower-body side")
    ap.add_argument("--hip_band_top", type=float, default=0.15,
                    help="top of vertical band (fraction of bbox height from TOP) when searching hip ridge")
    ap.add_argument("--hip_band_bot", type=float, default=0.45,
                    help="bottom of vertical band (fraction of bbox height from TOP) when searching hip ridge")
    ap.add_argument("--hip_exclude_front", type=float, default=0.30,
                help="fraction of bbox width to exclude on the FRONT side when searching ridge (to avoid head/neck)")
    ap.add_argument("--hip_side_window", type=int, default=11,
                    help="odd window size (pixels in ROI) for local-min smoothing when picking hip ridge")
    args = ap.parse_args()
    cv2.setUseOptimized(True)
    # Runtime-adjustable calibration with persistence
    from pathlib import Path
    import json

    os.makedirs(args.save_dir, exist_ok=True)
    smart = None
    if HAVE_SMART and args.use_smart:
        try:
            smart = SmartCattleDetector(
                model_path=args.smart_model,
                confidence_threshold=args.smart_conf,
                detect_cows_only=args.smart_cows_only,
                high_precision_mask=args.smart_high_precision_mask,
            )
            # Apply target override to match demo.py behaviour
            if args.smart_target == "cow":
                smart.detect_cows_only = True
                smart.target_classes = [19]
                tgt_txt = "cow"
            elif args.smart_target == "horse":
                smart.detect_cows_only = False
                smart.target_classes = [17]
                tgt_txt = "horse"
            else:
                smart.detect_cows_only = False
                smart.target_classes = [17, 19]
                tgt_txt = "both"
            print(f"[SMART] Using SmartCattleDetector model={args.smart_model} conf={args.smart_conf} target={tgt_txt} high_prec_mask={args.smart_high_precision_mask}")
        except Exception as e:
            print(f"[SMART] Failed to init SmartCattleDetector: {e}")
            smart = None
    _calib_file = Path(args.save_dir) / "height_calib.json"

    height_k_rt = float(args.height_k)
    # If a saved value exists, prefer it
    if _calib_file.exists():
        try:
            _saved = json.loads(_calib_file.read_text())
            if isinstance(_saved, dict) and "height_k" in _saved:
                height_k_rt = float(_saved["height_k"])
                print(f"[calib] loaded height_k={height_k_rt:.3f} from {_calib_file}")
        except Exception as e:
            print(f"[calib] failed to load {_calib_file}: {e}")
    # --- affine calibration (multi-sample, least squares): true = a_rt * measured + b_rt ---
    a_rt, b_rt = 1.0, 0.0
    _refs = []            # list of (measured_cm, true_cm)
    MAX_REFS = 10

    # Try to load them too if present (including saved refs)
    try:
        if _calib_file.exists():
            _saved = json.loads(_calib_file.read_text())
            if isinstance(_saved, dict):
                if "a" in _saved: a_rt = float(_saved["a"])
                if "b" in _saved: b_rt = float(_saved["b"])
                if "refs" in _saved and isinstance(_saved["refs"], list):
                    # validate tuples
                    for item in _saved["refs"]:
                        if (isinstance(item, (list, tuple)) and len(item) == 2):
                            try:
                                _refs.append((float(item[0]), float(item[1])))
                            except Exception:
                                pass
                    if len(_refs) > MAX_REFS:
                        _refs = _refs[-MAX_REFS:]
                if any(k in _saved for k in ("a","b")):
                    print(f"[calib] loaded a={a_rt:.4f}, b={b_rt:.2f} from {_calib_file} ({len(_refs)} refs)")
    except Exception as e:
        print(f"[calib] failed to load affine params/refs: {e}")
    cv2.setNumThreads(0)  # let OpenCV pick; set to a small int if you see CPU thrash

    os.makedirs(args.save_dir, exist_ok=True)

    # -------- YOLO fallback helper --------
    _yolo_fallback_model = [None]  # single-element list for closure mutability
    def yolo_fallback_detect(rgb_bgr: np.ndarray, model_path: str, conf: float = 0.25):
        """Return list of detections like demo.py using a local YOLO seg model.
        Each item: { 'body_box':(x1,y1,x2,y2), 'cattle_mask':(H,W) uint8, 'class_id':int, 'confidence':float }
        """
        if _ULTRA_YOLO is None:
            return []
        if _yolo_fallback_model[0] is None:
            try:
                _yolo_fallback_model[0] = _ULTRA_YOLO(model_path)
                print(f"[FALLBACK] loaded YOLO seg: {model_path}")
            except Exception as e:
                print(f"[FALLBACK] failed to load YOLO model: {e}")
                return []
        model = _yolo_fallback_model[0]
        H, W = rgb_bgr.shape[:2]
        try:
            res = model(rgb_bgr, conf=conf, classes=[17, 19], verbose=False)
        except Exception as e:
            print(f"[FALLBACK] inference error: {e}")
            return []
        dets = []
        for r in res:
            boxes = getattr(r, 'boxes', None)
            masks = getattr(r, 'masks', None)
            use_masks = (masks is not None)
            mask_data = masks.data.cpu().numpy() if use_masks else None
            if boxes is None:
                continue
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf_b = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy()) if hasattr(b, 'cls') and b.cls is not None else -1
                x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(W, int(x2)); y2 = min(H, int(y2))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                det = {
                    'body_box': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'confidence': conf_b,
                }
                if use_masks and i < mask_data.shape[0]:
                    raw = mask_data[i]
                    m = cv2.resize(raw, (W, H), interpolation=cv2.INTER_CUBIC)
                    thr = 0.2  # demo.py-style lower threshold
                    m = (m > thr).astype(np.uint8) * 255
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
                    det['cattle_mask'] = m
                dets.append(det)
        return dets

    # -------- Webcam/Video support --------
    if args.source in ("webcam", "video"):
        if args.source == "webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise SystemExit("❌ Cannot open webcam (index 0)")
        else:
            # Default to src/cli/1.mov if present, else try current dir
            vid_path = Path(__file__).parent / "1.mov"
            if Path("1.mov").exists():
                vid_path = Path("1.mov")
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                raise SystemExit(f"❌ Cannot open video file: {vid_path}")
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        src_name = "webcam" if args.source == "webcam" else f"video:{vid_path.name}"
        print(f"[INFO] Using {src_name} {W}x{H} @ {args.fps} FPS")

        # --- Detection hold state (for persistence) ---
        hold_bbox = None
        hold_label = None
        hold_conf = None
        hold_mask = None
        hold_ttl = 0
        
        # --- Cattle tracking for CSV logging (webcam/video mode) ---
        csv_logger = CattleDepthCsvLogger()
        cattle_id = 1  # Start with cattle ID 1
        last_cattle_bbox = None  # Previous cattle bounding box
        frames_since_last_detection = 0  # Counter for gap detection
        is_first_cattle = True  # Track if this is the first cattle
        frame_id = 0  # Frame counter for webcam/video mode
        csv_sep_mode = "AUTO"

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis = frame.copy()

            # --- optional 2D auto ground line (yellow) for webcam/video only ---
            if args.show_2d_ground:
                ground_y, _ = estimate_ground_y_auto(frame, roi_ratio=0.45)
                cv2.line(vis, (0, int(ground_y)), (W-1, int(ground_y)), (0, 255, 255), 3)

            last_bbox = None
            last_conf = None
            last_label = None
            mask = None
            dets = []
            if smart is not None:
                if hasattr(smart, "target_classes"):
                    smart.target_classes = [17, 19]
                if args.smart_debug and hasattr(smart, 'confidence_threshold'):
                    smart.confidence_threshold = min(smart.confidence_threshold, max(0.15, args.smart_conf))
                dets = smart.detect_cattle_smart(frame, 0) or []
                if args.smart_debug:
                    print(f"[SMART][video] dets={len(dets)}")
                    for i, d in enumerate(dets):
                        bb = d.get('body_box'); conf = float(d.get('confidence', 0.0))
                        cid = int(d.get('class_id', -1))
                        cname = 'cow' if cid == 19 else ('horse' if cid == 17 else str(cid))
                        if bb is not None:
                            x1, y1, x2, y2 = map(int, bb)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
                            cv2.putText(vis, f"{cname}:{conf:.2f}", (x1, max(0, y1-4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
            # --- Fallback to internal YOLO seg if requested or if Smart returned nothing ---
            if (args.force_yolo_fallback or not dets):
                fb = yolo_fallback_detect(frame, args.smart_model, conf=args.smart_conf)
                if args.smart_debug:
                    print(f"[FALLBACK][video] dets={len(fb)}")
                if fb:
                    dets = fb
            if dets:
                best = None; best_area = -1; best_conf = 0.0
                for d in dets:
                    x1,y1,x2,y2 = d['body_box']
                    area = max(0, x2-x1) * max(0, y2-y1)
                    if area > best_area:
                        best = d; best_area = area; best_conf = float(d.get('confidence', 0.0))
                if best is not None:
                    x1,y1,x2,y2 = map(int, best['body_box'])
                    last_bbox = (x1,y1,x2,y2)
                    last_conf = best_conf
                    # always draw bbox & label for visibility
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cls_id = int(best.get('class_id', -1))
                    cls_name = 'cow' if cls_id == 19 else ('horse' if cls_id == 17 else 'cattle')
                    lbl = f"{cls_name} {last_conf:.2f}" if last_conf is not None else cls_name
                    ty = max(0, y1 - 8)
                    cv2.putText(vis, lbl, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(vis, lbl, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    last_label = lbl
                    # Optional: translucent mask overlay like demo.py if provided by SmartCattleDetector or fallback
                    mask = best.get('cattle_mask', None)
                    if mask is None:
                        mask = best.get('mask', None)
                    if mask is not None and mask.shape[:2] != (H, W):
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    if mask is not None:
                        if args.show_mask:
                            over = vis.copy()
                            over[mask > 0] = (0.3*over[mask>0] + 0.7*np.array([100,0,100])).astype(np.uint8)
                            vis = over
                        bw = max(1, x2 - x1)
                        rx1 = x2 - int(0.30 * bw)
                        ry1 = y1
                        rx2 = x2
                        ry2 = y2
                        roi_box = (rx1, ry1, rx2, ry2)
                        hip_pt = _hip_top_from_mask_demo(mask, roi_box)
                        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                        if hip_pt is not None:
                            cv2.circle(vis, (int(hip_pt[0]), int(hip_pt[1])), 6, (0, 0, 255), -1)
                            head_uv = (int(hip_pt[0]), int(hip_pt[1]))
                    # --- Detection hold: store this detection and reset TTL
                    hold_bbox = last_bbox
                    hold_label = last_label
                    hold_conf = last_conf
                    hold_mask = mask
                    hold_ttl = int(max(0, args.hold_frames))

            # --- If no detection this frame, reuse the last good one while TTL remains
            if last_bbox is None and hold_ttl > 0 and hold_bbox is not None:
                last_bbox = hold_bbox
                last_label = hold_label
                last_conf = hold_conf
                mask = hold_mask
                hold_ttl -= 1
                # draw bbox & label to make it visible during hold
                x1, y1, x2, y2 = last_bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
                if last_label:
                    ty = max(0, y1 - 8)
                    cv2.putText(vis, last_label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(vis, last_label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                # also re-draw mask/ROI/hip if mask is available
                if mask is not None:
                    if mask.shape[:2] != (H, W):
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    if args.show_mask:
                        over = vis.copy()
                        over[mask > 0] = (0.3*over[mask>0] + 0.7*np.array([100,0,100])).astype(np.uint8)
                        vis = over
                    x1, y1, x2, y2 = last_bbox
                    bw = max(1, x2 - x1)
                    rx1 = x2 - int(0.30 * bw); ry1 = y1; rx2 = x2; ry2 = y2
                    roi_box = (rx1, ry1, rx2, ry2)
                    hip_pt = _hip_top_from_mask_demo(mask, roi_box)
                    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                    if hip_pt is not None:
                        cv2.circle(vis, (int(hip_pt[0]), int(hip_pt[1])), 6, (0, 0, 255), -1)
                        head_uv = (int(hip_pt[0]), int(hip_pt[1]))

            # ---- Cattle tracking and CSV logging (webcam/video mode) ----
            current_bbox = last_bbox  # Use the current detection bbox
            new_cattle_detected = False
            
            # Update frames counter
            if current_bbox is not None:
                frames_since_last_detection = 0  # Reset counter when cattle detected
                
                # Check if this is a new cattle
                if last_cattle_bbox is not None:
                    # Calculate bbox center movement
                    x1, y1, x2, y2 = current_bbox
                    last_x1, last_y1, last_x2, last_y2 = last_cattle_bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    last_center_x = (last_x1 + last_x2) / 2
                    last_center_y = (last_y1 + last_y2) / 2
                    
                    # Calculate distance moved
                    distance_moved = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                    
                    # New cattle if moved more than 150 pixels
                    if distance_moved > 150:
                        new_cattle_detected = True
                else:
                    # First cattle detection
                    if not is_first_cattle:
                        new_cattle_detected = True
            else:
                # No detection this frame, increment counter
                frames_since_last_detection += 1
                
                # New cattle if gap of 30+ frames and then detection appears
                if frames_since_last_detection >= 30 and last_cattle_bbox is not None:
                    # We'll detect new cattle on next detection
                    pass
            
            # Handle new cattle detection
            if new_cattle_detected and csv_sep_mode == "AUTO":
                cattle_id += 1
                csv_logger.log_separator("AUTO")
                print(f"[CSV] New cattle detected: ID {cattle_id}")
            
            if current_bbox is not None:
                # For webcam/video mode, we don't have depth information, so use placeholder values
                csv_logger.log(
                    cattle_id=cattle_id,
                    hip_height_cm=None,  
                    camera_height_m=None, 
                    confidence=last_conf,
                    frame_id=frame_id,
                    Zp_plane_depth=None, 
                    Z_head_depth=None,  
                    guard_check_ok=None,  
                    bbox=current_bbox
                )
            
            # Update tracking variables
            if current_bbox is not None:
                last_cattle_bbox = current_bbox
                is_first_cattle = False
            
            frame_id += 1

            cv2.imshow("Webcam Cattle Detection", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            if key in (ord('t'), ord('T')):
                if csv_sep_mode == "AUTO":
                        csv_sep_mode = "MANUAL"
                else:
                    csv_sep_mode ="AUTO"
                print(f"[CSV] CSV separator mode: {csv_sep_mode}")

            # manually adding cattle csv break 
            if key in (ord('b'), ord('B')):
                 if csv_sep_mode == "MANUAL":
                    csv_logger.log_separator("MANUAL")
                    cattle_id += 1
                    last_cattle_bbox = None
                    frames_since_last_detection = 0
                    print(f"[CSV] Manual break: New cattle ID {cattle_id}")
        cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup CSV logger
        csv_logger.close()
        return

    # -------- DepthAI pipeline --------
    p = dai.Pipeline()

    cam = p.createColorCamera()
    cam.setFps(args.fps)
    if args.oak_rgb_res == "720p":
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    else:
        if args.oak_rgb_res == "1080p":
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        elif args.oak_rgb_res == "4k":
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

    monoL = p.createMonoCamera(); monoR = p.createMonoCamera()
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    if args.oak_stereo_res == "400p":
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    elif args.oak_stereo_res == "480p":
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    else:
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    stereo = p.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(args.oak_lr_check)
    stereo.setSubpixel(args.oak_subpixel)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.initialConfig.setConfidenceThreshold(args.conf_thr)
    monoL.out.link(stereo.left); monoR.out.link(stereo.right)

    x_rgb = p.createXLinkOut(); x_rgb.setStreamName("rgb"); cam.video.link(x_rgb.input)
    x_dep = p.createXLinkOut(); x_dep.setStreamName("depth"); stereo.depth.link(x_dep.input)

    with dai.Device(p) as dev:
        # Output queues
        q_rgb = dev.getOutputQueue("rgb", 4, False)
        q_dep = dev.getOutputQueue("depth", 4, False)

        # Resolution
        vs = cam.getVideoSize()
        W, H = (int(vs[0]), int(vs[1])) if isinstance(vs, tuple) else (int(vs.width), int(vs.height))

        # Camera intrinsics from calibration (RGB-aligned)
        calib = dev.readCalibration()
        try:
            K33 = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        except:
            K33 = calib.getDefaultCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        fx, fy, cx, cy = float(K33[0][0]), float(K33[1][1]), float(K33[0][2]), float(K33[1][2])
        K = (fx, fy, cx, cy)

        print(f"[INFO] USB={dev.getUsbSpeed()} | RGB={W}x{H} | FPS={args.fps}")

        # (YOLO support removed)

        # GUI
        win = "OAK Depth Debug"
        if not args.headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, min(1920, W*2), min(1080, H*2))

        show_depth = True
        show_residual = False
        show_guard = False
        show_groundmask = True  # <--- NEW: ground inlier mask toggle
        show_hud = True
        mosaic = False
        zmin = None; zmax = None

        probe = [W // 2, H // 2]

        # Plane smoothing
        n_ema = None; d_ema = None; alpha = 0.2
        frame_id = 0
        last_print = 0
        ground_mask_bin = None  # (H,W) uint8 mask of RANSAC inliers, dilated for stability
        foot_y_ema = None  # smoothed y coordinate for horizontal ground line
        # Smoothed head point (for drawing only)
        head_u_ema = None
        head_v_ema = None

        # Smoothed foot x (anchor dot)
        foot_x_ema = None
        height_cm_plane_ema = None  # smoothed plane-based height estimate
        height_cm_bbox_ema = None   # smoothed bbox-derived fallback height

        # Cache morphology kernel once (used for ground mask dilate)
        dil_k = np.ones((max(3, args.stride // 2 * 2 + 1),
                        max(3, args.stride // 2 * 2 + 1)), np.uint8)

        last_label = None
        # --- Detection hold state for DepthAI ---
        hold_bbox = None
        hold_label = None
        hold_conf = None
        hold_mask = None
        hold_ttl = 0
        
        # --- Cattle tracking for CSV logging ---
        csv_logger = CattleDepthCsvLogger()
        cattle_id = 1  # Start with cattle ID 1
        last_cattle_bbox = None  # Previous cattle bounding box
        frames_since_last_detection = 0  # Counter for gap detection
        is_first_cattle = True  # Track if this is the first cattle
        csv_sep_mode = "AUTO"
        while True:
            f_rgb = q_rgb.tryGet(); f_dep = q_dep.tryGet()
            if f_rgb is None or f_dep is None:
                if args.headless:
                    time.sleep(0.01); continue
                vis = np.zeros((H, W, 3), np.uint8)
                cv2.putText(vis, "Waiting...", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(win, vis)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
                continue

            rgb = f_rgb.getCvFrame()
            depth = f_dep.getFrame().astype(np.float32) / 1000.0  # mm -> m
            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

            # ---- Fit ground plane in a bottom ROI ----
            y0 = int(H * (1.0 - args.roi_bottom))
            ys, xs = np.mgrid[y0:H:args.stride, 0:W:args.stride]
            z = depth[ys, xs]
            valid = (z > 0) & np.isfinite(z)

            xs_v = xs[valid]; ys_v = ys[valid]; z_v = z[valid]
            pts = []
            for u, v, Z in zip(xs_v.flat, ys_v.flat, z_v.flat):
                pts.append(backproject(float(u), float(v), float(Z), fx, fy, cx, cy))

            inlier_ratio = 0.0
            camH = float("nan")
            n, d, inmask = (None, None, None)
            if len(pts) >= 50:
                n, d, inmask = ransac_plane(
                    pts,
                    iters=args.ransac_iter,
                    tau=args.ransac_tau_cm / 100.0,
                    up=(0, -1, 0), up_cos=0.3,
                    min_in=args.min_inliers
                )
                if n is not None:
                    if n_ema is None:
                        n_ema, d_ema = n, d
                    else:
                        n_ema = (1 - alpha) * n_ema + alpha * n
                        n_ema = n_ema / (np.linalg.norm(n_ema) + 1e-9)
                        d_ema = (1 - alpha) * d_ema + alpha * d
                    inlier_ratio = float(np.count_nonzero(inmask)) / float(len(pts))
                    camH = abs(float(d_ema))
            # --- Build a binary ground inlier mask for this frame (even if we don't visualize it) ---
            ground_mask_bin = np.zeros((H, W), np.uint8)
            if inmask is not None and len(pts) > 0:
                # Use the same xs_v / ys_v sample coordinates you already computed
                xs_s = xs_v.flatten()
                ys_s = ys_v.flatten()
                in_idx = np.where(inmask)[0]
                for idx in in_idx:
                    u = int(xs_s[idx]); v = int(ys_s[idx])
                    if 0 <= u < W and 0 <= v < H:
                        ground_mask_bin[v, u] = 255
                # Thicken slightly to be tolerant to sampling sparsity
                # k = max(3, args.stride // 2 * 2 + 1)
                ground_mask_bin = cv2.dilate(ground_mask_bin, dil_k, iterations=1)

            # ---- Detector (SmartCattleDetector only) head-top ----
            head_uv = None
            last_bbox, last_conf = None, None
            mask = None
            dets = []
            if smart is not None:
                if args.smart_debug and hasattr(smart, 'confidence_threshold'):
                    smart.confidence_threshold = min(smart.confidence_threshold, max(0.15, args.smart_conf))
                dets = smart.detect_cattle_smart(rgb, frame_id) or []
                if args.smart_debug:
                    print(f"[SMART][oak] dets={len(dets)}")
                    for i, d in enumerate(dets):
                        bb = d.get('body_box'); conf = float(d.get('confidence', 0.0))
                        cid = int(d.get('class_id', -1))
                        cname = 'cow' if cid == 19 else ('horse' if cid == 17 else str(cid))
                        if bb is not None:
                            x1, y1, x2, y2 = map(int, bb)
                            cv2.rectangle(rgb, (x1, y1), (x2, y2), (128, 128, 128), 1)
                            cv2.putText(rgb, f"{cname}:{conf:.2f}", (x1, max(0, y1-4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
            # --- Fallback to internal YOLO seg if requested or if Smart returned nothing ---
            if (args.force_yolo_fallback or not dets):
                fb = yolo_fallback_detect(rgb, args.smart_model, conf=args.smart_conf)
                if args.smart_debug:
                    print(f"[FALLBACK][oak] dets={len(fb)}")
                if fb:
                    dets = fb
            if dets:
                best = None; best_area = -1; best_conf = 0.0
                for d in dets:
                    x1,y1,x2,y2 = d['body_box']
                    area = max(0, x2-x1) * max(0, y2-y1)
                    if area > best_area:
                        best = d; best_area = area; best_conf = float(d.get('confidence', 0.0))
                if best is not None:
                    x1,y1,x2,y2 = map(int, best['body_box'])
                    last_bbox = (x1, y1, x2, y2)
                    last_conf = best_conf
                    cls_id = int(best.get('class_id', -1))
                    last_label = f"{'cow' if cls_id == 19 else ('horse' if cls_id == 17 else 'cattle')} {last_conf:.2f}" if last_conf is not None else ('cow' if cls_id == 19 else ('horse' if cls_id == 17 else 'cattle'))
                    mask = best.get('cattle_mask', None)
                    if mask is None:
                        mask = best.get('mask', None)
                    if mask is not None and mask.shape[:2] != (H, W):
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    # --- Detection hold: store this detection and reset TTL
                    hold_bbox = last_bbox
                    hold_label = last_label
                    hold_conf = last_conf
                    hold_mask = mask
                    hold_ttl = int(max(0, args.hold_frames))

            # --- If no detection this frame, reuse the last good one while TTL remains
            if last_bbox is None and hold_ttl > 0 and hold_bbox is not None:
                last_bbox = hold_bbox
                last_label = hold_label
                last_conf = hold_conf
                mask = hold_mask
                hold_ttl -= 1

            # placeholders to draw later in the visualization block
            roi_box_draw = None
            hip_pt_draw = None
            mask_draw = None

            # --- HIP HEIGHT DETECTION from mask (demo-style) ---
            if mask is not None and last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                bw = max(1, x2 - x1)
                rx1 = x2 - int(0.30 * bw)
                ry1 = y1
                rx2 = x2
                ry2 = y2
                roi_box = (rx1, ry1, rx2, ry2)
                hip_pt = _hip_top_from_mask_demo(mask, roi_box)
                # store for later drawing
                roi_box_draw = roi_box
                hip_pt_draw = hip_pt
                mask_draw = mask
                if hip_pt is not None:
                    head_uv = (int(hip_pt[0]), int(hip_pt[1]))

            # Only fall back to manual probe if YOLO produced no bbox at all
            if head_uv is None and last_bbox is None:
                head_uv = (probe[0], probe[1])
            
            # --- Smooth head point for nicer visuals (drawing only) ---
            head_uv_draw = None
            if head_uv is not None:
                head_u_ema = ema_update(head_u_ema, float(head_uv[0]), alpha=0.30)
                head_v_ema = ema_update(head_v_ema, float(head_uv[1]), alpha=0.30)
                head_uv_draw = (int(head_u_ema), int(head_v_ema))
            # --- Prepare a foot anchor early (for height calc), without drawing ---
            foot_x_tmp, foot_y_tmp = None, None
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                foot_x_tmp = (x1 + x2) // 2
                # Snap to ground mask in that column if possible; otherwise use bbox bottom
                if ground_mask_bin is not None:
                    col = ground_mask_bin[:, foot_x_tmp]
                    idx = np.where(col > 0)[0]
                    if idx.size > 0:
                        foot_y_tmp = int(idx[np.argmin(np.abs(idx - y2))])
                    else:
                        foot_y_tmp = int(y2)
                else:
                    foot_y_tmp = int(y2)
            # ---- Direct head-to-foot height (constrained to ground plane) ----
            height_direct = None  # for HUD later
            _use_fx = foot_x_ema if foot_x_ema is not None else foot_x_tmp
            _use_fy = foot_y_ema if foot_y_ema is not None else foot_y_tmp

            if head_uv is not None and _use_fx is not None and _use_fy is not None:
                Z_head = depth_sample(depth, head_uv[0], head_uv[1])
                if Z_head:
                    P_head = backproject(head_uv[0], head_uv[1], Z_head, fx, fy, cx, cy)

                    P_foot = None
                    if n_ema is not None and d_ema is not None and abs(n_ema[1]) > 1e-6:
                        # re-anchor foot onto ground plane under head X,Z
                        Xh, _, Zh = P_head
                        Yf = -(n_ema[0]*Xh + n_ema[2]*Zh + d_ema) / n_ema[1]
                        P_foot = np.array([Xh, Yf, Zh], dtype=np.float32)

                    # fallback to raw depth if no plane
                    if P_foot is None:
                        Z_foot = depth_sample(depth, _use_fx, _use_fy)
                        if Z_foot:
                            P_foot = backproject(_use_fx, _use_fy, Z_foot, fx, fy, cx, cy)

                    if P_foot is not None:
                        height_direct = abs(P_head[1] - P_foot[1]) * 100.0
                        # Clamp to plausible human range
                        height_direct = float(np.clip(height_direct, 80.0, 200.0))
                        if "height_direct_old" not in locals():
                            height_direct_old = height_direct
                        height_direct = ema_update(height_direct_old, height_direct, alpha=0.2)
                        height_direct_old = height_direct

                        # cv2.putText(vis, f"HeightDirect={height_direct:.1f} cm",
                        #             (20, H - 40),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # ---- Robust plane-based height (always try to produce a value) ----
            Zp = None; Z = None
            height_cm_plane = None
            ok = None
            if n_ema is not None and head_uv is not None:
                # Use a point slightly BELOW the head to avoid horizon (denom ~ 0)
                delta_v = max(6, H // 50)                           # ~1–2% of image height
                v_anchor = float(min(H - 1, head_uv[1] + delta_v))  # nudge ray downward

                # 1) Plane intersection using the anchored v (much more stable)
                Zp = plane_intersect_depth(float(head_uv[0]), v_anchor, K, n_ema, d_ema)

                # 2) Try to get a robust head depth around/below the head
                if Zp is not None:
                    Z = robust_head_depth(float(head_uv[0]), float(head_uv[1]),
                                        depth, K, n_ema, d_ema,
                                        search_down_px=12, band_half_w=6)

                # 4) If we have *some* Z now, compute height from plane distance and smooth it
                if Z is not None:
                    P = backproject(float(head_uv[0]), float(head_uv[1]), float(Z), fx, fy, cx, cy)
                    height_raw = abs(float(n_ema @ P + d_ema)) * 100.0  # cm
                    height_raw = float(np.clip(height_raw, 120.0, 220.0))  # clamp plausible range
                    height_cm_plane_ema = ema_update(height_cm_plane_ema, height_raw, alpha=0.15) if height_cm_plane_ema is not None else height_raw
                    height_cm_plane = height_cm_plane_ema
                    if Zp is not None:
                        g_disp = max(0.08, 0.02 * Zp)
                        ok = (Z <= Zp - g_disp)
                else:
                    # No reliable head depth this frame → feed bbox height into the plane EMA
                    if height_cm_bbox_ema is not None:
                        height_cm_plane_ema = (
                            ema_update(height_cm_plane_ema, height_cm_bbox_ema, alpha=0.10)
                            if height_cm_plane_ema is not None else height_cm_bbox_ema
                        )
                    height_cm_plane = height_cm_plane_ema

            # Decide what height value to surface this frame
            height_cm = height_cm_plane if height_cm_plane is not None else (
                height_cm_plane_ema if height_cm_plane_ema is not None else height_cm_bbox_ema
            )
            if height_cm is not None:
                height_cm += 2 ## added height weight

            # ---- Cattle tracking and CSV logging ----
            current_bbox = last_bbox 
            new_cattle_detected = False
            
            if current_bbox is not None:
                frames_since_last_detection = 0  # Reset counter when cattle detected
                
                # Check if this is a new cattle
                if last_cattle_bbox is not None:
                    # Calculate bbox center movement
                    x1, y1, x2, y2 = current_bbox
                    last_x1, last_y1, last_x2, last_y2 = last_cattle_bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    last_center_x = (last_x1 + last_x2) / 2
                    last_center_y = (last_y1 + last_y2) / 2
                    
                    # Calculate distance moved
                    distance_moved = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                    
                    # New cattle if moved more than 150 pixels
                    if distance_moved > 150:
                        new_cattle_detected = True
                else:
                    # First cattle detection
                    if not is_first_cattle:
                        new_cattle_detected = True
            else:
                # No detection this frame, increment counter
                frames_since_last_detection += 1
                
                # New cattle if gap of 30+ frames and then detection appears
                if frames_since_last_detection >= 30 and last_cattle_bbox is not None:
                    # Detect new cattle on next detection
                    pass
            
            # Handle new cattle detection
            if new_cattle_detected and csv_sep_mode == "AUTO":
                cattle_id += 1
                csv_logger.log_separator("AUTO")
                print(f"[CSV] New cattle detected: ID {cattle_id}")
            
            # Log measurement if we have valid data
            if current_bbox is not None and height_cm is not None:
                csv_logger.log(
                    cattle_id=cattle_id,
                    hip_height_cm=height_cm,
                    camera_height_m=camH if np.isfinite(camH) else None,
                    confidence=last_conf,
                    frame_id=frame_id,
                    Zp_plane_depth=Zp,
                    Z_head_depth=Z,
                    guard_check_ok=ok,
                    bbox=current_bbox
                )
            
            # Update tracking variables
            if current_bbox is not None:
                last_cattle_bbox = current_bbox
                is_first_cattle = False

            # ---- Visualization ----
            if args.headless:
                now = time.time()
                if now - last_print > 0.5:
                    print(f"[{frame_id}] CamH={camH:.3f}m inliers={inlier_ratio:.2f} "
                          f"| probe=({head_uv[0]},{head_uv[1]}) "
                          f"Zp={None if Zp is None else round(Zp,3)} "
                          f"Z={None if Z is None else round(Z,3)} ok={ok} "
                          f"height={None if height_cm is None else round(height_cm,1)}cm")
                    last_print = now
            else:
                vis = rgb.copy()

                # --- draw stored mask/ROI/hip (moved here so drawings are not lost) ---
                if mask_draw is not None:
                    if args.show_mask:
                        over = vis.copy()
                        over[mask_draw > 0] = (0.3 * over[mask_draw > 0] + 0.7 * np.array([100, 0, 100])).astype(np.uint8)
                        vis = over
                if roi_box_draw is not None:
                    rx1, ry1, rx2, ry2 = map(int, roi_box_draw)
                    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                if hip_pt_draw is not None:
                    cv2.circle(vis, (int(hip_pt_draw[0]), int(hip_pt_draw[1])), 5, (0, 0, 255), -1)

                # Depth colormap
                dep_img = None
                if show_depth:
                    dep_img, (zmin, zmax) = apply_colormap_depth(depth, zmin, zmax)

                # Residual heatmap (|n·P + d|)
                res_img = None
                if show_residual and n_ema is not None:
                    res = np.full((H, W), np.nan, np.float32)
                    ys2, xs2 = np.mgrid[0:H:args.stride, 0:W:args.stride]
                    zz = depth[ys2, xs2]
                    val = (zz > 0) & np.isfinite(zz)
                    for (u, v, Z_) in zip(xs2[val].flat, ys2[val].flat, zz[val].flat):
                        P = backproject(float(u), float(v), float(Z_), fx, fy, cx, cy)
                        res[int(v), int(u)] = abs(float(n_ema @ P + d_ema))
                    tmp = res.copy()
                    bad = ~np.isfinite(tmp)
                    if np.any(~bad):
                        p1, p2 = np.nanpercentile(tmp, 5), np.nanpercentile(tmp, 95)
                        tmp = (np.clip(tmp, p1, p2) - p1) / max(1e-6, (p2 - p1))
                        tmp[bad] = 0
                        res_img = cv2.applyColorMap((tmp * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
                        res_img[bad] = (0, 0, 0)

                # Plane-guard mask (valid points: Z <= Zp - guard)
                guard_img = None
                if show_guard and n_ema is not None:
                    guard_img = np.zeros_like(rgb)
                    ys3, xs3 = np.mgrid[0:H:args.stride, 0:W:args.stride]
                    for (u, v) in zip(xs3.flat, ys3.flat):
                        Zp_ = plane_intersect_depth(float(u), float(v), K, n_ema, d_ema)
                        if Zp_ is None:
                            continue
                        Z_ = depth_sample(depth, u, v, 3)
                        if Z_ is None:
                            continue
                        g = max(0.08, 0.02 * Zp_)
                        if Z_ <= Zp_ - g:
                            guard_img[int(v), int(u)] = (0, 255, 0)

                # ----- Ground inlier mask overlay (reuse prebuilt mask) -----
                if show_groundmask and n_ema is not None and ground_mask_bin is not None:
                    overlay = vis.copy()
                    # blue-ish tint on inlier pixels
                    overlay[ground_mask_bin > 0] = (overlay[ground_mask_bin > 0] * 0.3 + np.array([255, 128, 0]) * 0.7).astype(np.uint8)
                    vis = overlay
                
                # ----- draw YOLO person bbox (if detected) -----
                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox
                    # box
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # label
                    lbl = last_label if last_label is not None else (f"cow {last_conf:.2f}" if last_conf is not None else "cow")
                    ty = max(0, y1 - 8)
                    cv2.putText(vis, lbl, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(vis, lbl, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

                # Head/probe marker
                if head_uv is not None:
                    if last_bbox is None:
                        # Manual probe: show red X
                        cv2.drawMarker(vis, (int(head_uv_draw[0]), int(head_uv_draw[1])), (0, 0, 255),
                                    cv2.MARKER_TILTED_CROSS, 16, 2)

                ## ----- Auto ground redline: constrained to RANSAC ground mask -----
                # Draw a line along the TOP boundary of the ground inlier mask (closest to the person),
                # clipped to mask support and split into contiguous segments.
                if ground_mask_bin is not None and np.any(ground_mask_bin):
                    # (optional) slight smoothing/cleanup so the contour is less jagged
                    kernel = np.ones((3, 3), np.uint8)
                    mask_for_line = cv2.morphologyEx(ground_mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask_for_line = cv2.dilate(mask_for_line, kernel, iterations=1)

                    step = max(2, int(args.groundline_stride))
                    pts_line = []

                    # For each column, pick the TOPMOST ground pixel (smallest v)
                    for u in range(0, W, step):
                        col = mask_for_line[:, u]
                        idx = np.where(col > 0)[0]
                        if idx.size == 0:
                            continue
                        v_i = int(idx.min())  # top boundary of ground in this column
                        pts_line.append([u, v_i])

                    # # draw only contiguous runs so we don't bridge gaps
                    # if len(pts_line) >= 2:
                    #     pts_line = sorted(pts_line, key=lambda p: p[0])
                    #     segments = []
                    #     seg = [pts_line[0]]
                    #     for i in range(1, len(pts_line)):
                    #         if abs(pts_line[i][0] - pts_line[i-1][0]) <= step * 2:
                    #             seg.append(pts_line[i])
                    #         else:
                    #             if len(seg) > 1:
                    #                 segments.append(np.array(seg, np.int32).reshape(-1, 1, 2))
                    #             seg = [pts_line[i]]
                    #     if len(seg) > 1:
                    #         segments.append(np.array(seg, np.int32).reshape(-1, 1, 2))

                    #     for seg in segments:
                    #         cv2.polylines(vis, [seg], isClosed=False, color=(0, 0, 255), thickness=2)

                    # --- anchor near the detected person’s feet and draw a straight ground line ---
                    if last_bbox is not None:
                        x1, y1, x2, y2 = last_bbox

                        # Smooth foot X (anchor dot) from bbox center
                        foot_x_raw = (x1 + x2) // 2
                        foot_x_ema = int(ema_update(foot_x_ema, float(foot_x_raw), alpha=0.25)) if foot_x_ema is not None else foot_x_raw

                        foot_y = None
                        if 'pts_line' in locals() and pts_line:
                            # Preferred: use the mask-derived line point nearest the smoothed foot_x
                            foot_pt = min(pts_line, key=lambda p: abs(p[0] - foot_x_ema))
                            foot_y = int(foot_pt[1])
                        else:
                            # Fallback: bbox bottom, try to snap to ground mask in that column
                            foot_y = int(y2)
                            if ground_mask_bin is not None:
                                col = ground_mask_bin[:, foot_x_ema]
                                idx = np.where(col > 0)[0]
                                if idx.size > 0:
                                    # choose ground pixel closest to bbox bottom
                                    foot_y = int(idx[np.argmin(np.abs(idx - y2))])

                        # Smooth the horizontal ground line Y and draw it
                        if foot_y is not None:
                            foot_y_ema = int(ema_update(float(foot_y_ema), float(foot_y), alpha=0.15)) if foot_y_ema is not None else foot_y
                            cv2.line(vis, (0, int(foot_y_ema)), (W - 1, int(foot_y_ema)), (0, 0, 255), 2)

                # --- derive 'ok' for display (guard check), even in robust path ---
                if ok is None and Zp is not None and Z is not None:
                    g_disp = max(0.08, 0.02 * Zp)   # same guard you use elsewhere
                    ok = (Z <= Zp - g_disp)
                # HUD
                if show_hud:
                    def hud(t, y, c=(255, 255, 255)):
                        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                    c, 2, cv2.LINE_AA)

                    y_line = 28
                    hud(f"CamH={camH:.3f} m | inliers={inlier_ratio:.2f}", y_line)
                    y_line += 30
                    hud(f"Probe (u,v)=({head_uv[0]},{head_uv[1]})  "
                        f"Zp={None if Zp is None else round(Zp,3)}  "
                        f"Z={None if Z is None else round(Z,3)}  ok={ok}", y_line, (180, 255, 180))
                    y_line += 30
                    h_show = height_cm_plane_ema if height_cm_plane_ema is not None else height_cm_bbox_ema
                    hud(f"HipHeight={None if h_show is None else round(h_show,1)} cm", y_line, (255, 220, 180))
                    y_line += 30
                    hud(f"[affine a={a_rt:.4f}, b={b_rt:.1f}]", y_line, (170, 220, 255))
                    y_line += 30
                    hud(f"[refs {len(_refs)}/{MAX_REFS}]", y_line, (220, 220, 170))
                    y_line += 30
                    hud("Keys: 1=Depth  2=Residual  3=Guard  4=GroundMask  d=HUD  m=Mosaic b=add_break_in_csv "
                        "p=Probe  S=Snapshot  space=Pause  f=Full  q/ESC=Quit", y_line, (200, 220, 255))
                    y_line += 24
                    hud("Calib: ,/. adjust k  c=auto→140cm r=record  k=solve(LS)  L=list  X=clear  C=save  R=reset  [+/-/[/]] zoom", y_line, (200, 220, 255))
                    y_line += 30
                    hud(f"CSV Mode: {csv_sep_mode}", y_line, (255, 255, 180))

                # Compose panes
                panes = [vis]
                if dep_img is not None: panes.append(dep_img)
                if res_img is not None: panes.append(res_img)
                if guard_img is not None: panes.append(guard_img)

                if mosaic and len(panes) > 1:
                    row1 = cv2.hconcat(panes[:2]) if len(panes) >= 2 else cv2.hconcat([panes[0], panes[0]])
                    row2 = cv2.hconcat(panes[2:4]) if len(panes) >= 4 else cv2.hconcat([panes[0], panes[0]])
                    show = cv2.vconcat([row1, row2])
                else:
                    show = vis

                cv2.imshow(win, show)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')): break
                if k == ord('1'): show_depth = not show_depth
                if k == ord('2'): show_residual = not show_residual
                if k == ord('3'): show_guard = not show_guard
                if k == ord('4'): show_groundmask = not show_groundmask  # toggle ground inlier mask
                # --- live calibration of height_k_rt ---
                if k == ord(','):  # decrease scale
                    height_k_rt = max(0.40, height_k_rt - 0.01)
                if k == ord('.'):  # increase scale
                    height_k_rt = min(1.20, height_k_rt + 0.01)
                if k == ord('c'):  # one-tap auto-calibrate to 180cm
                    if 'h_show' in locals() and h_show is not None and h_show > 0:
                        height_k_rt = float(np.clip(height_k_rt * (140.0 / h_show), 0.60, 1.20))
                # --- Multi-sample affine calibration (up to 10 cattle) ---
                # Press 'r' to record (measured -> true) pair for current frame
                if k == ord('r'):
                    # Block additional input once MAX_REFS reached
                    if len(_refs) >= MAX_REFS:
                        print(f"[calib] {MAX_REFS} references already recorded. Press 'k' to solve or 'R' to reset.")
                    elif 'h_show' in locals() and h_show is not None and h_show > 0:
                        try:
                            true_str = input("[calib] Enter TRUE height in cm for this cattle (e.g., 140): ").strip()
                            true_cm = float(true_str)
                            meas = float(h_show)
                            _refs.append((meas, true_cm))
                            print(f"[calib] added ref #{len(_refs)}: measured={meas:.1f} → true={true_cm:.1f}")
                            if len(_refs) == MAX_REFS:
                                print(f"[calib] Collected {MAX_REFS} references. Press 'k' to solve or 'R' to reset.")
                        except Exception as e:
                            print(f"[calib] invalid input: {e}")
                    else:
                        print("[calib] no height estimate available yet to record (h_show is None)")

                # Press 'L' to list current references
                if k in (ord('L'), ord('l')):
                    if _refs:
                        print("[calib] current refs:")
                        for i, (m, t) in enumerate(_refs, 1):
                            print(f"  #{i:02d}: measured={m:.1f} → true={t:.1f}")
                    else:
                        print("[calib] no refs recorded yet")

                # Press 'X' to clear references
                if k in (ord('X'), ord('x')):
                    _refs.clear()
                    print("[calib] cleared all refs")

                # Press 'R' to reset calibration file and start fresh
                if k == ord('R'):
                    try:
                        ans = input("[calib] Reset calibration file and clear refs? (y/n): ").strip().lower()
                        if ans == 'y':
                            if _calib_file.exists():
                                os.remove(_calib_file)
                                print(f"[calib] removed calibration file: {_calib_file}")
                            _refs.clear()
                            a_rt, b_rt = 1.0, 0.0
                            height_k_rt = 0.62  # reset to default baseline
                            print("[calib] calibration reset to default (k=0.62, a=1.0, b=0.0)")
                        else:
                            print("[calib] reset cancelled")
                    except Exception as e:
                        print(f"[calib] failed to reset calibration: {e}")

                # Press 'k' to solve affine using least squares over all refs
                if k == ord('k'):
                    if len(_refs) >= 2:
                        xs = np.array([m for (m, _) in _refs], dtype=np.float64)
                        ys = np.array([t for (_, t) in _refs], dtype=np.float64)
                        A = np.vstack([xs, np.ones_like(xs)]).T
                        sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
                        a_rt, b_rt = float(sol[0]), float(sol[1])
                        # Report fit quality
                        pred = a_rt * xs + b_rt
                        mae = float(np.mean(np.abs(pred - ys)))
                        print(f"[calib] solved affine (LS) over {len(_refs)} refs: a={a_rt:.4f}, b={b_rt:.2f} | MAE={mae:.2f} cm")
                    else:
                        print("[calib] need at least 2 refs before solving (press 'r' to add)")

                # Extend 'C' to save affine params too
                if k == ord('C'):
                    try:
                        _calib_file.write_text(json.dumps({
                            "height_k": float(height_k_rt),
                            "a": float(a_rt),
                            "b": float(b_rt),
                            "refs": [(float(m), float(t)) for (m, t) in _refs]
                        }, indent=2))
                        print(f"[calib] saved k={height_k_rt:.3f}, a={a_rt:.4f}, b={b_rt:.2f} with {len(_refs)} refs to {_calib_file}")
                    except Exception as e:
                        print(f"[calib] failed to save {_calib_file}: {e}")
                if k in (ord('d'), ord('D')): show_hud = not show_hud
                if k == ord('m'): mosaic = not mosaic
                if k == ord('f'):
                    fs = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        win, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if fs == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                    )
                if k in (ord('+'), ord('='), ord(']')):
                    w = int(show.shape[1] * 1.1); h = int(show.shape[0] * 1.1)
                    cv2.resizeWindow(win, w, h)
                if k in (ord('-'), ord('_'), ord('[')):
                    w = max(320, int(show.shape[1] * 0.9)); h = max(240, int(show.shape[0] * 0.9))
                    cv2.resizeWindow(win, w, h)
                if k == ord(' '):  # pause
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 in (ord(' '), ord('q'), 27):
                            break
                if k == ord('p'):
                    clicked = [None]
                    def cb(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            clicked[0] = (x, y)
                    cv2.setMouseCallback(win, cb)
                    print("[probe] Click to set probe position…")
                    while True:
                        k3 = cv2.waitKey(10) & 0xFF
                        if clicked[0] is not None:
                            probe[:] = clicked[0]
                            cv2.setMouseCallback(win, lambda *a: None)
                            break
                        if k3 in (27, ord('q')):
                            cv2.setMouseCallback(win, lambda *a: None)
                            break
                if k in (ord('S'), ord('s')):
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    base = f"{args.save_dir}/{ts}"
                    cv2.imwrite(base + "_rgb.jpg", rgb)
                    dep_u16 = (np.clip(depth, 0, 65.535) * 1000).astype(np.uint16)
                    cv2.imwrite(base + "_depthmm.png", dep_u16)
                    np.savez_compressed(
                        base + "_meta.npz",
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        n=None if n_ema is None else n_ema,
                        d=None if d_ema is None else d_ema,
                        camH=camH,
                        probe=np.array(head_uv),
                        Zp=Zp, Z=Z,
                        inliers=inlier_ratio
                    )
                    print(f"[save] {base}_rgb.jpg / _depthmm.png / _meta.npz")

                if k in (ord('b'), ord('B')):
                     if csv_sep_mode == "MANUAL":
                        csv_logger.log_separator("MANUAL")
                        cattle_id += 1
                        last_cattle_bbox = None
                        frames_since_last_detection = 0 
                        print(f"[CSV] manual break added")

                if k in (ord('t'), ord('T')):
                    if csv_sep_mode == "AUTO":
                        csv_sep_mode = "MANUAL"
                    else:
                        csv_sep_mode ="AUTO"
                    print(f"CSV separator mode: {csv_sep_mode}")

            frame_id += 1

        csv_logger.close()

        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
