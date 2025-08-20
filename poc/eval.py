import os, json, math, csv, cv2, argparse
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import traceback
# ... (중략: util, resolve_video_path 등 기존 함수 그대로 유지) ...

from tracking_module.detection_and_tracking import DetectorAndTracker

try:
    import motmetrics as mm
    _HAS_MOT = True
except Exception:
    _HAS_MOT = False

# ---- how to use -----
'''
python eval.py \
  --json-dir /home/hiperwall/Ai_modules/Ai/poc/labels \
  --video-dir /home/hiperwall/Ai_modules/Ai/poc/videos \
  --out-root ./eval_out \
  --iou 0.3 --conf 0.25 \
  --tracker ocsort \
  --nproc 4 \
  --camera-ids 1,2,3,4
'''

# --- 새로 추가 ---
from unicodedata import normalize
import glob

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

def _pick_bbox(ann):
    for k in ("bbox","box","xywh"):
        if k in ann and isinstance(ann[k], (list,tuple)) and len(ann[k])>=4:
            x,y,w,h = ann[k][:4]
            return float(x),float(y),float(w),float(h)
    if all(k in ann for k in ("x","y","w","h")):
        return float(ann["x"]),float(ann["y"]),float(ann["w"]),float(ann["h"])
    if all(k in ann for k in ("x1","y1","x2","y2")):
        x1,y1,x2,y2 = float(ann["x1"]),float(ann["y1"]),float(ann["x2"]),float(ann["y2"])
        return x1,y1,max(0.0,x2-x1),max(0.0,y2-y1)
    return None

def _is_person_id(ann_id: str):
    return isinstance(ann_id, str) and ann_id.startswith("person_")

def json_to_mot(json_path: Path, out_dir: Path):
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
        bb = _pick_bbox(ann)
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

    (out_dir/"gt"/"gt.txt").write_text("\n".join(rows), encoding="utf-8")
    return data

def load_gt(gt_txt: Path):
    per_frame = defaultdict(list)
    if not gt_txt.exists():
        return per_frame
    for ln in gt_txt.read_text(encoding="utf-8").splitlines():
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

try:
    import motmetrics as mm
    _HAS_MOT = True
except Exception:
    _HAS_MOT = False

# ---- 워커 전역 (프로세스마다 딱 하나) ----
_DT = None
_CAM_ID = None
_TRACKER_KIND = None
_CONF_THR = None

def _init_worker(camera_id: int, tracker: str, conf_thr: float):
    """
    프로세스 시작 시 한 번만 DetectorAndTracker 생성해서 전역으로 보관.
    각 워커는 고정된 camera_id로 카프카/리아이디 응답 컨슈머를 붙는다.
    """
    global _DT, _CAM_ID, _TRACKER_KIND, _CONF_THR
    _CAM_ID = int(camera_id)
    _TRACKER_KIND = tracker
    _CONF_THR = float(conf_thr)

    # 한 번만 생성
    _DT = DetectorAndTracker(cameraID=_CAM_ID, tracker_type=_TRACKER_KIND, conf_threshold=_CONF_THR)
    print(f"[worker] camera_id={_CAM_ID} detector-tracker ready")


def _process_one(json_path: str, video_dir: str, out_root: str, iou_thr: float):
    """
    기존 run_one에서 DetectorAndTracker 생성/해제 부분만 제거.
    워커 전역 _DT 를 사용.
    """
    global _DT, _CAM_ID
    json_path = Path(json_path)
    seq_name = json_path.stem
    out_dir = Path(out_root) / seq_name
    res_det = out_dir / "res_det.txt"
    res_trk = out_dir / "res_track.txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "seq": seq_name, "video": None, "camera_id": _CAM_ID,
        "tp":0,"fp":0,"fn":0,"precision":0.0,"recall":0.0,
        "idf1":None,"idp":None,"idr":None,"mota":None,"motp":None,"num_switches":None,
        "error": None
    }
    try:
        # 1) JSON -> MOT GT
        data = json_to_mot(json_path, out_dir)
        file_name = (data.get("video") or {}).get("file_name")
        if not file_name:
            raise RuntimeError("JSON.video.file_name not found")
        video_path = resolve_video_path(Path(video_dir), file_name)
        summary["video"] = str(video_path)
        gt = load_gt(out_dir/"gt"/"gt.txt")

        # 2) Video open
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # 워커 전역 DetectorAndTracker 사용
        dt = _DT

        acc = mm.MOTAccumulator(auto_id=True) if _HAS_MOT else None
        tp=fp=fn=0
        with res_det.open("w", encoding="utf-8") as f_det, res_trk.open("w", encoding="utf-8") as f_trk:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                frame_idx += 1

                # detection
                boxes, scores, cids, _ = dt.infer(frame, debug=False)
                dets=[]
                for (x,y,w,h), s, cid in zip(boxes, scores, cids):
                    if int(cid)!=0: continue
                    if not math.isfinite(float(s)) or float(s) < _CONF_THR: continue
                    f_det.write(f"{frame_idx},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{float(s):.6f},-1,-1,-1\n")
                    dets.append([float(x),float(y),float(w),float(h),float(s)])

                # tracking
                vis, timing2, _bx, _sc, _ci, tracks = dt.detect_and_track(frame, debug=False, return_vis=False)
                trk_ids, trk_boxes = [], []
                for t in tracks:
                    l,t0,r,b = t.to_ltrb() if hasattr(t,"to_ltrb") else t.ltrb
                    x,y,w,h = float(l),float(t0),float(r-l),float(b-t0)
                    tid = int(t.track_id)
                    f_trk.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1\n")
                    trk_ids.append(tid); trk_boxes.append([x,y,w,h])

                # detection PR
                gts = gt.get(frame_idx, [])
                gt_boxes = [g[1] for g in gts]
                matched=set()
                for dx,dy,dw,dh,_ in dets:
                    best=-1
                    for gi, gb in enumerate(gt_boxes):
                        if gi in matched: continue
                        if iou_xywh([dx,dy,dw,dh], gb) >= iou_thr:
                            best=gi; break
                    if best>=0:
                        matched.add(best); tp+=1
                    else:
                        fp+=1
                fn += (len(gt_boxes)-len(matched))

                # MOT metrics
                if _HAS_MOT:
                    import numpy as np
                    gt_ids = [g[0] for g in gts]
                    if len(gt_ids)==0 and len(trk_ids)==0:
                        acc.update([],[],[])
                    else:
                        C = np.full((len(gt_ids), len(trk_ids)), np.inf, dtype=float)
                        for i,gb in enumerate(gt_boxes):
                            for j,tb in enumerate(trk_boxes):
                                iou = iou_xywh(gb, tb)
                                if iou >= iou_thr:
                                    C[i,j] = 1.0 - iou
                        acc.update(gt_ids, trk_ids, C)

        cap.release()

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        summary.update({"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec})

        if _HAS_MOT:
            mh = mm.metrics.create()
            metr = mh.compute(acc, metrics=["idf1","idp","idr","mota","motp","num_switches"], name=seq_name)
            row = metr.loc[seq_name]
            summary.update({
                "idf1": float(row.get("idf1", float("nan"))),
                "idp": float(row.get("idp", float("nan"))),
                "idr": float(row.get("idr", float("nan"))),
                "mota": float(row.get("mota", float("nan"))),
                "motp": float(row.get("motp", float("nan"))),
                "num_switches": float(row.get("num_switches", float("nan"))),
            })

        (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK][cam={_CAM_ID}] {seq_name}: P={prec:.3f} R={rec:.3f}  -> {out_dir}")
        return summary

    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    json_list = sorted(json_dir.glob("*.json"))
    if len(json_list)==0:
        raise RuntimeError(f"No JSON found in {json_dir}")

    # 카메라 ID 1,2,3,4 로 고정. nproc은 최소 1, 최대 4만 의미있게 씀.
    camera_ids = [1,2,3,4][:max(1, min(args.nproc, 4))]

    # 영상 단위로 라운드로빈 "배정"만 함. 프레임 단위가 아님.
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

    # 요약 CSV
    sum_dir = out_root / "_summary"; sum_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sum_dir / "metrics.csv"
    fields = ["seq","video","camera_id","precision","recall","tp","fp","fn","idf1","idp","idr","mota","motp","num_switches","error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k:r.get(k) for k in fields})
    print(f"[DONE] Wrote summary CSV -> {csv_path}")


def _spawn_worker(cam_id, tracker, conf_thr, jobs, video_dir, out_root, iou_thr, out_conn):
    """
    각 프로세스 엔트리. 초기화 → 배정된 영상들 처리 → 결과 리스트를 파이프로 부모에 전달.
    """
    try:
        _init_worker(cam_id, tracker, conf_thr)
        out = _worker_main(jobs, video_dir, out_root, iou_thr)
        out_conn.send(out)
    except Exception as e:
        # 실패한 경우에도 형식을 맞춘 결과를 보내 부모가 CSV 만들 수 있게 함
        fallback = [{"seq": Path(j).stem, "video": None, "camera_id": cam_id, "error": f"{type(e).__name__}: {e}"} for j in jobs]
        try:
            out_conn.send(fallback)
        except Exception:
            pass
    finally:
        try:
            out_conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
