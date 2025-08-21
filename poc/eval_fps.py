import os, json, math, csv, cv2, argparse
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import traceback
from unicodedata import normalize
import glob
import time

# ---- how to use -----
'''
python eval.py \
  --json-dir /home/hiperwall/Ai_modules/Ai/poc/labels \
  --video-dir /home/hiperwall/Ai_modules/Ai/poc/videos \
  --out-root ./eval_out \
  --iou 0.3 --conf 0.25 \
  --tracker ocsort \
  --nproc 4
'''

# ---- 파이프라인 의존 (당신 코드) ----
from tracking_module.detection_and_tracking import DetectorAndTracker

try:
    import motmetrics as mm
    _HAS_MOT = True
except Exception:
    _HAS_MOT = False

# ----------------- 파일명 보정 -----------------
def resolve_video_path(video_dir: Path, file_name: str) -> Path:
    file_name = file_name.strip()
    file_name = normalize("NFC", file_name)
    p_exact = (video_dir / file_name)
    if p_exact.exists():
        return p_exact

    stem, ext = os.path.splitext(file_name)
    cand = []
    if stem.endswith("_1"):
        cand.append(video_dir / (stem[:-2] + ext))
    else:
        cand.append(video_dir / (stem + "_1" + ext))
    cand.extend([
        video_dir / (stem + "_2" + ext),
        video_dir / (stem + "_01" + ext),
        video_dir / (stem.replace(" ", "") + ext),
        video_dir / (stem.replace(" ", "") + "_1" + ext),
    ])
    for c in cand:
        if c.exists():
            return c

    for path in video_dir.glob(f"{stem}*{ext}"):
        if path.is_file():
            return path

    patt = os.path.join(str(video_dir), f"{stem}*")
    hit = sorted([Path(p) for p in glob.glob(patt)])
    for path in hit:
        if path.is_file() and path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            return path

    for path in video_dir.rglob("*"):
        if not path.is_file():
            continue
        name_norm = normalize("NFC", path.name)
        if name_norm.lower() == file_name.lower():
            return path
        if name_norm.lower().startswith(stem.lower()) and path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            return path

    raise FileNotFoundError(f"Video not found for '{file_name}' under {video_dir}")

# ----------------- 공통 유틸 -----------------
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    union = aw*ah + bw*bh - inter
    return inter / max(union, 1e-6)

def _is_person_id(ann_id: str):
    return isinstance(ann_id, str) and ann_id.startswith("person_")

# --------- 좌표 판별/정규화 (xywh vs xyxy 자동) ----------
def _normalize_bbox_list(bbox, img_w, img_h):
    """
    bbox(list/tuple[4])가 xywh인지 xyxy인지 자동 추정해서 xywh로 리턴
    """
    x0, y0, a2, b2 = map(float, bbox[:4])

    # 후보 1: 입력이 xywh라고 가정
    cand_xywh = (x0, y0, a2, b2)

    # 후보 2: 입력이 xyxy라고 가정
    w2 = max(0.0, a2 - x0)
    h2 = max(0.0, b2 - y0)
    cand_xyxy_as_xywh = (x0, y0, w2, h2)

    def plausible(x, y, w, h):
        if not all(map(math.isfinite, [x,y,w,h])): return False
        if w <= 0 or h <= 0: return False
        return (0 <= x <= img_w) and (0 <= y <= img_h) and (w <= img_w*1.1) and (h <= img_h*1.1)

    p1 = plausible(*cand_xywh)
    p2 = plausible(*cand_xyxy_as_xywh)

    if p1 and not p2:
        x,y,w,h = cand_xywh
    elif p2 and not p1:
        x,y,w,h = cand_xyxy_as_xywh
    elif p1 and p2:
        over1 = (cand_xywh[0]+cand_xywh[2] > img_w+1e-3) or (cand_xywh[1]+cand_xywh[3] > img_h+1e-3)
        over2 = (cand_xyxy_as_xywh[0]+cand_xyxy_as_xywh[2] > img_w+1e-3) or (cand_xyxy_as_xywh[1]+cand_xyxy_as_xywh[3] > img_h+1e-3)
        if over1 and not over2:
            x,y,w,h = cand_xyxy_as_xywh
        elif over2 and not over1:
            x,y,w,h = cand_xywh
        else:
            area1 = cand_xywh[2]*cand_xywh[3]
            area2 = cand_xyxy_as_xywh[2]*cand_xyxy_as_xywh[3]
            x,y,w,h = (cand_xyxy_as_xywh if area2 < area1 else cand_xywh)
    else:
        return None

    x = max(0.0, min(x, img_w-1.0))
    y = max(0.0, min(y, img_h-1.0))
    w = max(1.0, min(w, img_w - x))
    h = max(1.0, min(h, img_h - y))
    return (x,y,w,h)

def _pick_bbox(ann, img_w, img_h):
    if all(k in ann for k in ("x1","y1","x2","y2")):
        x1,y1,x2,y2 = float(ann["x1"]),float(ann["y1"]),float(ann["x2"]),float(ann["y2"])
        x,y,w,h = x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)
    elif any(k in ann for k in ("bbox","box","xywh")):
        raw = ann.get("bbox") or ann.get("box") or ann.get("xywh")
        if isinstance(raw, (list,tuple)) and len(raw) >= 4:
            norm = _normalize_bbox_list(raw, img_w, img_h)
            if norm is None:
                return None
            x,y,w,h = norm
        else:
            return None
    elif all(k in ann for k in ("x","y","w","h")):
        x,y,w,h = float(ann["x"]),float(ann["y"]),float(ann["w"]),float(ann["h"])
    else:
        return None

    x = max(0.0, min(x, img_w-1.0))
    y = max(0.0, min(y, img_h-1.0))
    w = max(1.0, min(w, img_w - x))
    h = max(1.0, min(h, img_h - y))
    return (x,y,w,h)

def json_to_mot(json_path: Path, out_dir: Path, *, img_w: int, img_h: int):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    anns = data.get("annotations", []) or []
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"gt").mkdir(parents=True, exist_ok=True)

    id_map = {}
    next_id = 1
    rows = []
    for ann in anns:
        ann_id = ann.get("id")
        if not _is_person_id(ann_id):
            continue
        bb = _pick_bbox(ann, img_w, img_h)
        if bb is None:
            continue
        x,y,w,h = bb
        if w <= 0 or h <= 0:
            continue
        frame = int(ann.get("frame", 0))
        if ann_id not in id_map:
            id_map[ann_id] = next_id; next_id += 1
        tid = id_map[ann_id]
        rows.append(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0")

    # (out_dir/"gt"/"gt.txt").write_text("\n".join(rows), encoding="utf-8")  # ← 파일 저장 주석
    data["_gt_rows"] = rows  # 파일 안 만들고 데이터로 보관
    return data

def load_gt_from_data(data):
    """json_to_mot(...)가 돌려준 data에서 gt rows를 읽어 메모리로 구성"""
    per_frame = defaultdict(list)
    rows = data.get("_gt_rows", [])
    for ln in rows:
        if not ln.strip(): continue
        parts = ln.split(",")
        frame = int(float(parts[0]))
        tid   = int(float(parts[1]))
        x,y,w,h = map(float, parts[2:6])
        conf  = float(parts[6]) if len(parts)>6 else 1.0
        cls   = int(float(parts[7])) if len(parts)>7 else 1
        if conf <= 0 or cls != 1 or w<=0 or h<=0:
            continue
        per_frame[frame].append((tid, [x,y,w,h]))
    return per_frame

# ---- 워커 전역 (프로세스마다 딱 하나) ----
_DT = None
_CAM_ID = None
_TRACKER_KIND = None
_CONF_THR = None

def _init_worker(camera_id: int, tracker: str, conf_thr: float):
    global _DT, _CAM_ID, _TRACKER_KIND, _CONF_THR
    _CAM_ID = int(camera_id)
    _TRACKER_KIND = tracker
    _CONF_THR = float(conf_thr)

    _DT = DetectorAndTracker(cameraID=_CAM_ID, tracker_type=_TRACKER_KIND, conf_threshold=_CONF_THR)
    print(f"[worker] camera_id={_CAM_ID} detector-tracker ready")


def _process_one(json_path: str, video_dir: str, out_root: str, iou_thr: float):
    """
    전체 프레임 수(frame_idx)와 영상 처리 구간의 경과 시간(elapsed_sec)만으로 FPS 계산.
    파일 저장 루틴은 전부 주석 처리.
    """
    global _DT, _CAM_ID
    json_path = Path(json_path)
    seq_name = json_path.stem
    out_dir = Path(out_root) / seq_name
    # res_det = out_dir / "res_det.txt"
    # res_trk = out_dir / "res_track.txt"
    # out_dir.mkdir(parents=True, exist_ok=True)  # ← 저장 안 함

    summary = {
        "seq": seq_name, "video": None, "camera_id": _CAM_ID,
        "frames": 0, "elapsed_sec": 0.0, "fps_total": 0.0,
        "error": None
    }
    try:
        # JSON에서 먼저 file_name만 뽑아 비디오 경로/크기 확인
        data_raw = json.loads(json_path.read_text(encoding="utf-8"))
        file_name = (data_raw.get("video") or {}).get("file_name")
        if not file_name:
            raise RuntimeError("JSON.video.file_name not found")
        video_path = resolve_video_path(Path(video_dir), file_name)
        summary["video"] = str(video_path)

        # 비디오 크기 획득
        cap_probe = cv2.VideoCapture(str(video_path))
        if not cap_probe.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        img_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        img_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap_probe.release()

        # 1) JSON -> MOT GT (좌표 정규화 포함, 파일은 만들지 않음)
        data = json_to_mot(json_path, out_dir, img_w=img_w, img_h=img_h)
        gt = load_gt_from_data(data)

        # 2) Video open
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        dt = _DT

        frame_idx = 0
        t0 = time.perf_counter()  # ← 영상 처리 시작
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # detection + tracking만 수행. 결과 파일 기록은 주석.
            _vis, _timing2, _bx, _sc, _ci, tracks = dt.detect_and_track(frame, debug=False, return_vis=False)
            # if res_det.exists(): pass
            # if res_trk.exists(): pass

        t1 = time.perf_counter()  # ← 영상 처리 끝
        cap.release()

        elapsed = max(1e-9, t1 - t0)
        fps = frame_idx / elapsed
        elapsed = max(1e-9, t1 - t0)
        fps = frame_idx / elapsed

        summary.update({
            "frames": frame_idx,
            "elapsed_sec": elapsed,
            "fps_total": fps
        })

        # (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")  # ← 파일 저장 주석
        print(f"[OK][cam={_CAM_ID}] {seq_name}: frames={frame_idx} elapsed={elapsed:.3f}s FPS={fps:.2f}")
        return summary

    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        # (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")  # ← 파일 저장 주석
        print(f"[ERR][cam={_CAM_ID}] {seq_name}: {summary['error']}")
        traceback.print_exc()
        return summary


def _worker_main(video_jobs, video_dir, out_root, iou_thr):
    """
    워커 안에서 여러 영상을 순차 처리.
    """
    results = []
    for jp in video_jobs:
        r = _process_one(jp, video_dir, out_root, iou_thr)
        results.append(r)
    return results

def _spawn_worker(cam_id, tracker, conf_thr, jobs, video_dir, out_root, iou_thr, out_conn):
    """
    각 프로세스 엔트리. 초기화 → 배정된 영상들 처리 → 결과 리스트를 파이프로 부모에 전달.
    """
    try:
        _init_worker(cam_id, tracker, conf_thr)
        out = _worker_main(jobs, video_dir, out_root, iou_thr)
        out_conn.send(out)
    except Exception as e:
        fallback = [{"seq": Path(j).stem, "video": None, "camera_id": cam_id, "frames": 0, "elapsed_sec": 0.0, "fps_total": 0.0, "error": f"{type(e).__name__}: {e}"} for j in jobs]
        try:
            out_conn.send(fallback)
        except Exception:
            pass
    finally:
        try:
            out_conn.close()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="ocsort", choices=["ocsort","bytetrack"])
    ap.add_argument("--nproc", type=int, default=1)
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    video_dir = Path(args.video_dir)
    out_root = Path(args.out_root)
    # out_root.mkdir(parents=True, exist_ok=True)  # ← 결과 파일 저장 안 하므로 주석

    json_list = sorted(json_dir.glob("*.json"))
    if len(json_list)==0:
        raise RuntimeError(f"No JSON found in {json_dir}")

    # 카메라 ID 1,2,3,4 고정. nproc은 최대 4만 의미 있게.
    camera_ids = [1,2,3,4][:max(1, min(args.nproc, 4))]

    # 영상 단위로 라운드로빈 "배정"만 함. 프레임 단위 아님.
    buckets = [[] for _ in camera_ids]
    for i, jp in enumerate(json_list):
        buckets[i % len(camera_ids)].append(str(jp))

    procs = []
    pipes = []

    for cam_idx, cam_id in enumerate(camera_ids):
        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(
            target=_spawn_worker,
            args=(cam_id, args.tracker, args.conf, buckets[cam_idx], str(video_dir), str(out_root), float(args.iou), child_conn),
            name=f"worker-cam{cam_id}",
            daemon=True
        )
        p.start()
        procs.append(p)
        pipes.append(parent_conn)

    # 결과 수집
    results = []
    for conn in pipes:
        try:
            results.extend(conn.recv())
        except EOFError:
            pass

    for p in procs:
        p.join()

    # conf별/전체 FPS 요약 출력
    ok = [r for r in results if not r.get("error")]
    if len(ok):
        mean_fps = sum(r.get("fps_total", 0.0) for r in ok) / len(ok)
        mean_elapsed = sum(r.get("elapsed_sec", 0.0) for r in ok) / len(ok)
        total_frames = sum(r.get("frames", 0) for r in ok)
        print(f"[SUMMARY] videos={len(ok)} frames={total_frames} mean_elapsed={mean_elapsed:.3f}s mean_FPS={mean_fps:.2f}")
    else:
        print("[SUMMARY] no successful results")

     # 파일 저장 로직 전부 주석 (요약 CSV 등)
    sum_dir = out_root / "_summary"; sum_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sum_dir / "timing.csv"
    fields = ["seq","video","camera_id","frames","elapsed_sec","fps_total","error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k:r.get(k) for k in fields})
    print(f"[DONE] Wrote timing CSV -> {csv_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
