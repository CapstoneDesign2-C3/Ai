import cv2
import numpy as np
import os
import time
import argparse
from build_engine import load_trt_engine, infer_with_trt
from scipy.optimize import linear_sum_assignment

# IoU 계산
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (areaA + areaB - interArea + 1e-6)

# 후처리: NMS + person 필터
def postprocess(outputs, iou_threshold):
    dets = outputs[0]
    boxes = dets[:, :4]
    scores = dets[:, 4]
    classes = dets[:, 5].astype(int)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        if classes[i] != 0:
            idxs = idxs[1:]
            continue
        keep.append(i)
        rest = idxs[1:]
        if len(rest) == 0:
            break
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious < iou_threshold]
    results = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        results.append({'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'score': float(scores[i])})
    return results

# 매칭: IoU 기반 Hungarian
def match_detections(tracklets, detections, iou_threshold):
    if not tracklets:
        return [], list(range(len(detections)))
    track_boxes = np.array([t['bbox'] for t in tracklets])
    det_boxes = np.array([d['bbox'] for d in detections])
    cost = 1.0 - np.array([[compute_iou(tb, db) for db in det_boxes] for tb in track_boxes])
    row_idx, col_idx = linear_sum_assignment(cost)
    matches, unmatched_dets = [], []
    used_det = set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] < (1.0 - iou_threshold):
            matches.append((r, c))
            used_det.add(c)
    unmatched_dets = [j for j in range(len(detections)) if j not in used_det]
    return matches, unmatched_dets

# 전처리: FP16 엔진을 고려하여 float16 캐스팅
def preprocess(frame):
    img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
    tensor = np.transpose(img, (2, 0, 1))[None, ...]
    return tensor.astype(np.float16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True, help='Path to yolov11.engine')
    parser.add_argument('--video', required=True, help='Path to input MP4 file')
    parser.add_argument('--output_dir', default='output_crops', help='Directory to save crops')
    parser.add_argument('--camera_id', default='CAM_TEST', help='Dummy camera ID')
    parser.add_argument('--iou_thresh', type=float, default=0.3)
    parser.add_argument('--max_age', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 엔진 로드
    engine, context = load_trt_engine(args.engine)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.video}")

    tracklets = []
    next_id = 1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = time.time()

        inp = preprocess(frame)
        outputs = infer_with_trt(context, inp)
        detections = postprocess(outputs, args.iou_thresh)

        matches, unmatched = match_detections(tracklets, detections, args.iou_thresh)
        # Update matched
        for trk_idx, det_idx in matches:
            trk = tracklets[trk_idx]
            trk['bbox'] = detections[det_idx]['bbox']
            trk['age'] = 0
            trk['hits'] += 1
        # Create new tracklets
        for det_idx in unmatched:
            bbox = detections[det_idx]['bbox']
            trk = {'id': next_id, 'bbox': bbox, 'age': 0, 'hits': 1}
            tracklets.append(trk)
            # crop 저장
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            track_dir = os.path.join(args.output_dir, f"id_{next_id}")
            os.makedirs(track_dir, exist_ok=True)
            cv2.imwrite(os.path.join(track_dir, f"frame_{frame_idx}.jpg"), crop)
            next_id += 1
        # Age & prune
        to_remove = []
        for trk in tracklets:
            if not any(trk is tracklets[r] for r,_ in matches):
                trk['age'] += 1
            if trk['age'] > args.max_age:
                to_remove.append(trk)
        for trk in to_remove:
            tracklets.remove(trk)

        # 시각화
        for trk in tracklets:
            x1, y1, x2, y2 = map(int, trk['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{trk['id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow('Test DetectionTracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
