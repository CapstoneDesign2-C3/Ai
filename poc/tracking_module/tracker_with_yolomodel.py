import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class DetectorTracker:
    """
    Uses a YOLO model for detection and DeepSort for tracking.
    """
    def __init__(self,
                 model_path: str,
                 input_size=(640, 640),
                 conf_thresh=0.25,
                 max_age=30,
                 max_iou_distance=0.7,
                 nn_budget=100,
                 device: str = 'cuda'):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device

        # Detection config
        self.input_size = input_size
        self.conf_thresh = conf_thresh

        # Initialize DeepSort tracker
        self.tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            nn_budget=nn_budget,
            embedder="mobilenet",  # default embedder
            half=True
        )

    def infer(self, frame: np.ndarray):
        # Run detection: returns a Results object
        results = self.model.predict(
            source=[frame],
            imgsz=self.input_size,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )
        return results[0]

    def detect_and_track(self, frame: np.ndarray):
        # 1) Detect
        res = self.infer(frame)
        dets = []  # list of ([x1,y1,x2,y2], confidence, class)

        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy().tolist()
            score = float(conf.cpu().numpy())
            cls_id = int(cls.cpu().numpy())
            # Only keep 'person' class (0)
            if cls_id != 0:
                continue
            dets.append(([x1, y1, x2, y2], score, cls_id))

        # 2) Track
        tracks = self.tracker.update_tracks(dets, frame=frame)
        results = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, r, b = tr.to_ltrb()
            results.append({
                'local_id': tr.track_id,
                'bbox': [int(l), int(t), int(r), int(b)],
                'score': tr.det_conf,
                'class': tr.det_class
            })
        return results

    def cleanup(self):
        # DeepSort has no GPU-allocated context to free explicitly
        pass

    def __del__(self):
        self.cleanup()
