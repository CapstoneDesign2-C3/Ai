# tracking_module/trackers.py
# ByteTrack / OC-SORT 어댑터 (외부 라이브러리 래핑)
# 설치:
#   ByteTrack: pip install yolox==0.3.0 lapx cython_bbox onemetric
#   OC-SORT : pip install ocsort lapx cython_bbox onemetric

import sys
try:
    import lap  # ocsort가 기대
except Exception:
    import lapx as lap  # 대체
    sys.modules['lap'] = lap  # ocsort가 import lap 할 때 lapx를 쓰게 함

from dataclasses import dataclass

@dataclass
class _TrackView:
    track_id: int
    ltrb: tuple  # (l, t, r, b)
    _confirmed: bool = True

    def to_ltrb(self):
        return self.ltrb

    def is_confirmed(self):
        return self._confirmed


# --------------------------
# ByteTrack Adapter
# --------------------------
class ByteTrackAdapter:
    """
    YOLO xywh → ByteTrack(xyxy) 변환하여 업데이트
    - 입력: detections = [([x,y,w,h], score, class_id), ...]
    - 출력: List[_TrackView]
    """
    def __init__(
        self,
        track_thresh: float = 0.5,   # detection score threshold (high)
        track_buffer: int = 30,      # max_age 비슷한 개념 (초 단위가 아님)
        match_thresh: float = 0.8,   # IOU match threshold
        mot20: bool = False,
    ):
        try:
            from yolox.tracker.byte_tracker import BYTETracker
        except Exception as e:
            raise ImportError(
                "BYTETracker import 실패. `pip install yolox==0.3.0 lapx cython_bbox onemetric` 필요"
            ) from e
        self._BYTETracker = BYTETracker
        self.tracker = BYTETracker(track_thresh, track_buffer, match_thresh, mot20=mot20)

    @staticmethod
    def _xywh_to_xyxy(x, y, w, h):
        l = x
        t = y
        r = x + w
        b = y + h
        return l, t, r, b

    def update_tracks(self, detections, frame=None):
        """
        detections 형식: [([x,y,w,h], score, cls), ...] (정수/실수 ok, 사람 class 필터는 호출측에서)
        ByteTrack은 np.ndarray 입력:
          - dets: (N, 5) [x1, y1, x2, y2, score]
        """
        import numpy as np

        if not detections:
            online_targets = self.tracker.update(np.empty((0, 5), dtype=np.float32), (frame.shape[0], frame.shape[1]), (frame.shape[0], frame.shape[1]))
            return []

        dets = []
        for (x, y, w, h), s, _cid in detections:
            l, t, r, b = self._xywh_to_xyxy(x, y, w, h)
            dets.append([l, t, r, b, float(s)])
        dets = np.asarray(dets, dtype=np.float32)

        # tracker.update의 입력: dets, img_info(H,W), img_size(H,W)
        H, W = (frame.shape[0], frame.shape[1]) if frame is not None else (1080, 1920)
        online_targets = self.tracker.update(dets, (H, W), (H, W))

        tracks = []
        for t in online_targets:
            # yolox.tracker.byte_tracker. STrack의 tlwh, tlbr 제공
            tlbr = t.tlbr.astype(float)  # (l,t,r,b)
            l, t0, r, b = tlbr.tolist()
            tid = int(t.track_id)
            # ByteTrack은 확정 상태만 반환하는 편이지만, 명시적으로 True
            tracks.append(_TrackView(track_id=tid, ltrb=(l, t0, r, b), _confirmed=True))
        return tracks


# --------------------------
# OC-SORT Adapter
# --------------------------
class OCSORTAdapter:
    """
    YOLO xywh → OCSORT(xyxy) 변환하여 업데이트
    - 입력: detections = [([x,y,w,h], score, class_id), ...]
    - 출력: List[_TrackView]
    """
    def __init__(
        self,
        det_thresh: float = 0.5,     # confidence threshold
        iou_threshold: float = 0.3,  # association IOU
        max_age: int = 30,
        min_hits: int = 3,
        delta_t: int = 3,
        use_byte: bool = False,      # 내부 한계상 Byte 정책 일부 사용
    ):
        try:
            from ocsort import OCSort
        except Exception as e:
            raise ImportError(
                "OCSort import 실패. `pip install ocsort lapx cython_bbox onemetric` 필요"
            ) from e
        self._OCSort = OCSort
        self.tracker = OCSort(
            det_thresh=det_thresh,
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
            delta_t=delta_t,
            use_byte=use_byte,
        )

    @staticmethod
    def _xywh_to_xyxy(x, y, w, h):
        l = x
        t = y
        r = x + w
        b = y + h
        return l, t, r, b

    def update_tracks(self, detections, frame=None):
        import numpy as np
        import torch  # ★

        # 1) (N,6): [x1,y1,x2,y2,score,cls]
        if detections:
            rows = []
            for (x, y, w, h), s, cid in detections:
                l, t, r, b = self._xywh_to_xyxy(x, y, w, h)
                cls = int(cid) if cid is not None else 0
                rows.append([float(l), float(t), float(r), float(b), float(s), float(cls)])
            dets = np.asarray(rows, dtype=np.float32)
            dets[~np.isfinite(dets)] = 0.0
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        # 2) 이미지 크기
        if frame is not None:
            H, W = frame.shape[:2]
        else:
            H, W = 1080, 1920

        # 3) numpy → torch Tensor로 변환 (ocsort가 .numpy()를 호출하므로 Tensor로 줘야 함) ★
        dets_t = torch.as_tensor(dets, dtype=torch.float32, device="cpu")

        # 4) 다양한 시그니처 호환
        try:
            outputs = self.tracker.update(dets_t, (H, W))              # ★
        except TypeError:
            try:
                outputs = self.tracker.update(dets_t, (H, W), (H, W))  # ★
            except TypeError:
                outputs = self.tracker.update(dets_t)                  # ★

        # 5) 반환 파싱
        tracks = []
        if outputs is None or len(outputs) == 0:
            return tracks

        if isinstance(outputs, np.ndarray):
            for row in outputs:
                if row.shape[0] >= 5:
                    l, t0, r, b, tid = row[:5].tolist()
                    tracks.append(_TrackView(track_id=int(tid),
                                            ltrb=(float(l), float(t0), float(r), float(b)),
                                            _confirmed=True))
        else:
            # 객체 리스트 포맷도 호환
            for obj in outputs:
                if hasattr(obj, "tlbr") and hasattr(obj, "track_id"):
                    l, t0, r, b = [float(v) for v in obj.tlbr]
                    tracks.append(_TrackView(track_id=int(obj.track_id),
                                            ltrb=(l, t0, r, b),
                                            _confirmed=True))
        return tracks