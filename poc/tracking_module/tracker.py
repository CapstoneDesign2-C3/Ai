import cv2
import numpy as np
import time
import base64
import json
import argparse
from build_engine import load_trt_engine, infer_with_trt
from scipy.optimize import linear_sum_assignment
from kafka import KafkaProducer

class Tracklet:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = np.array(bbox, dtype=float)
        self.age = 0
        self.hits = 1

class DetectionTrackingWorker:
    def __init__(self, rtsp_url, camera_id, kafka_servers, reid_topic,
                 engine_path='yolov11.engine', max_age=30, iou_threshold=0.3):
        # RTSP Stream Capture
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream {rtsp_url}")
        # TensorRT Engine
        self.engine, self.context = load_trt_engine(engine_path)
        # Tracking state
        self.tracklets = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        # Kafka Producer for new track dispatch
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.camera_id = camera_id
        self.reid_topic = reid_topic

    '''
    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW, add batch dimension
        return np.transpose(img, (2, 0, 1))[None, ...]
    '''

    # for TensorRT-fp16 engine
    def preprocess(self, frame):
        # 1) Resize & Normalize (0–1)
        img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        # 2) HWC -> CHW, add batch dim
        tensor = np.transpose(img, (2, 0, 1))[None, ...]
        # 3) Cast to float16 for FP16 engine
        return tensor.astype(np.float16)

    def postprocess(self, outputs):
        # outputs shape: (1, N, 6) -> [x1, y1, x2, y2, conf, cls]
        dets = outputs[0]
        boxes = dets[:, :4]
        scores = dets[:, 4]
        classes = dets[:, 5]
        # Simple NMS
        idxs = np.argsort(scores)[::-1]
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            rest = idxs[1:]
            ious = np.array([self.compute_iou(boxes[i], boxes[j]) for j in rest])
            idxs = rest[ious < self.iou_threshold]
        results = []
        for i in keep:
            if int(classes[i]) != 0:  # person class only
                continue
            x1, y1, x2, y2 = boxes[i]
            results.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'score': float(scores[i])
            })
        return results

    def compute_iou(self, boxA, boxB):
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

    def match_detections(self, detections):
        if not self.tracklets:
            return [], list(range(len(detections)))
        track_boxes = np.array([t.bbox for t in self.tracklets])
        det_boxes = np.array([d['bbox'] for d in detections])
        cost = 1.0 - np.array([[self.compute_iou(tb, db) for db in det_boxes] for tb in track_boxes])
        row_idx, col_idx = linear_sum_assignment(cost)
        matches, unmatched_dets = [], []
        used_det = set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < (1.0 - self.iou_threshold):
                matches.append((r, c))
                used_det.add(c)
        unmatched_dets = [j for j in range(len(detections)) if j not in used_det]
        return matches, unmatched_dets

    def update_tracks(self, detections, frame, timestamp):
        matches, unmatched = self.match_detections(detections)
        # Update matched tracklets
        for trk_idx, det_idx in matches:
            trk = self.tracklets[trk_idx]
            trk.bbox = np.array(detections[det_idx]['bbox'], dtype=float)
            trk.age = 0
            trk.hits += 1
        # Create and dispatch new tracklets
        for det_idx in unmatched:
            bbox = detections[det_idx]['bbox']
            trk = type('T', (), {})()
            trk.id = self.next_id
            trk.bbox = np.array(bbox, dtype=float)
            trk.age = 0
            trk.hits = 1
            self.tracklets.append(trk)
            self.dispatch(trk, frame, timestamp)
            self.next_id += 1
        # Age out old tracklets
        removed = []
        for trk in self.tracklets:
            if all((trk.id != m[0] for m in matches)):
                trk.age += 1
            if trk.age > self.max_age:
                removed.append(trk)
        for trk in removed:
            self.tracklets.remove(trk)

    def dispatch(self, trk, frame, timestamp):
        x1, y1, x2, y2 = map(int, trk.bbox)
        crop = frame[y1:y2, x1:x2]
        _, enc = cv2.imencode('.jpg', crop)
        payload = {
            'camera_id': self.camera_id,
            'timestamp': timestamp,
            'tracklet_id': trk.id,
            'crop_jpg': base64.b64encode(enc.tobytes()).decode('ascii')
        }
        self.producer.send(self.reid_topic, payload)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = time.time()
            inp = self.preprocess(frame)
            outputs = infer_with_trt(self.context, inp)
            dets = self.postprocess(outputs)
            self.update_tracks(dets, frame, ts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp_url', required=True)
    parser.add_argument('--camera_id', required=True)
    parser.add_argument('--kafka_servers', nargs='+', default=['localhost:9092'])
    parser.add_argument('--reid_topic', default='new_tracks')
    args = parser.parse_args()
    worker = DetectionTrackingWorker(
        rtsp_url=args.rtsp_url,
        camera_id=args.camera_id,
        kafka_servers=args.kafka_servers,
        reid_topic=args.reid_topic
    )
    worker.run()

