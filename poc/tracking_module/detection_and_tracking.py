import os
import time
import json
import atexit
from contextlib import contextmanager
from pathlib import Path
  
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import torch
import threading
import io, base64

from dotenv import load_dotenv
from kafka_util import consumers, producers
from db_util.db_util import PostgreSQL
from tracking_module.trackers import ByteTrackAdapter, OCSORTAdapter

from PIL import Image
from datetime import datetime


class DetectorAndTracker:
    def __init__(self, class_names_path=None, conf_threshold=0.45, iou_threshold=0.35, cameraID=None,
                 tracker_type: str = "bytetrack"):
        """
        TensorRT YOLOv11 + DeepSORT (PyTorch) 통합
        - Torch가 만든 primary CUDA context를 PyCUDA/TensorRT가 공유
        - GPU 만지는 구간에서만 push/pop (누수 방지)
        """
        load_dotenv("../env/aws.env")

        # --- Kafka ---
        # self.frame_consumer = consumers.FrameConsumer(camera_id=cameraID)
        # self.track_result_producer = producers.TrackResultProducer(cameraID=cameraID)
        self.result_producer = producers.create_track_result_producer(camera_id=cameraID)

        # --- 로컬 ID 세트 ---
        self.camera_id = cameraID
        self.local_id_set = set()
        self.pending_reid = set()               # 전송했지만 응답 대기
        self.local_to_global = {}               # local_id -> global_id
        self.track_start_ts = {}                # local_id -> appeared_time(ms)
        self.local_to_detection = {}      
        # lifecycle: lid -> {"start_ms": int, "status": "reid_pending"|"active"|"end_scheduled", "gid": Optional[int], "detection_id": Optional[int]}  
        self.track_lifecycle = {}               
        self._state_lock = threading.Lock()     # 멀티 스레드 환경에서 race condition 관리
        self.inflight_reid = set()  
        self.end_deadlines = {}  # lid -> deadline_ms
        self.end_grace_ms = int(os.getenv("END_GRACE_MS", "1500"))  # 1.5s 기본



        self.db = PostgreSQL(os.getenv('DB_HOST'), os.getenv('DB_NAME'),
                             os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_PORT'))
        
        self.reid_consumer = consumers.create_reid_response_consumer(cameraID, handler=self.on_reid_response)
        threading.Thread(target=self.reid_consumer.run, daemon=True).start()

        # --- CUDA 컨텍스트: primary retain만, push/pop은 with 블록에서 ---
        _ = torch.zeros(1, device='cuda')     # Torch로 primary context 생성 트리거
        cuda.init()
        self.cuda_ctx = cuda.Device(0).retain_primary_context()
        atexit.register(self._cleanup_cuda)

        # --- 설정값 ---
        self.engine_path = os.getenv('ENGINE_PATH')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # --- 클래스 이름 ---
        self.class_names = self._load_class_names(class_names_path)
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

        # --- TensorRT 객체/버퍼: 활성 컨텍스트 블록 안에서 생성 ---
        with self._cuda():
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.engine = self._load_engine()
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        # --- Tracker 선택 --- #
        self.tracker_type = tracker_type.lower().strip()
        if self.tracker_type == "bytetrack":
            # ByteTrack: 빠르고 ID 스위치 적음 (appearance 없이도 강함)
            self.tracker = ByteTrackAdapter(
                track_thresh=0.5,     # YOLO confidence 기준
                track_buffer=60,      # 끊김 허용 프레임(상황 따라 30~90)
                match_thresh=0.8
            )
        elif self.tracker_type == "ocsort":
            # OC-SORT: 가림에 더 강함
            self.tracker = OCSORTAdapter(
                det_thresh=0.5,
                iou_threshold=0.3,
                max_age=120,
                min_hits=3,
                delta_t=3,
                use_byte=False
            )
        else:
            raise ValueError(f"Unsupported tracker_type: {tracker_type}")

        
        # 안전한 CPU 임베더(0벡터/NaN 방지)
        # self.embedder = self._cpu_embedder_safe

        self._print_engine_info()


    # --------------------- UTIL: CUDA ctx ---------------------
    @contextmanager
    def _cuda(self):
        """이 블록 안에서만 primary CUDA context 활성화(push) → 작업 → sync → pop"""
        self.cuda_ctx.push()
        try:
            yield
        finally:
            try:
                if hasattr(self, "stream"):
                    self.stream.synchronize()
            finally:
                self.cuda_ctx.pop()

    def _cleanup_cuda(self):
        try:
            self.cuda_ctx.detach()
        except Exception:
            pass


    # --------------------- ENGINE/IO --------------------------
    def _load_class_names(self, class_names_path):
        if class_names_path and Path(class_names_path).exists():
            try:
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    if class_names_path.endswith('.json'):
                        data = json.load(f)
                        return data.get('names', list(data.values()))
                    else:
                        return [line.strip() for line in f.readlines() if line.strip()]
            except Exception as e:
                print(f"⚠️ 클래스 파일 로드 실패: {e}")

        # COCO 80 classes 기본
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def _load_engine(self):
        try:
            with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            print(f"✅ TensorRT 엔진 로드 완료: {self.engine_path}")
            return engine
        except Exception as e:
            print(f"❌ 엔진 로드 실패: {e}")
            raise

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        if hasattr(self.engine, 'num_io_tensors'):
            # TRT 8.5+ API
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.context.get_tensor_shape(name)
                size = trt.volume(shape)

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))

                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_shape = shape
                    self.input_name = name
                    inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': name})
                    self.output_name = name
        else:
            # TRT 7/8 이전
            for binding in self.engine:
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                shape = self.context.get_binding_shape(binding)
                size = trt.volume(shape)

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))

                if self.engine.binding_is_input(binding):
                    self.input_shape = shape
                    self.input_name = binding
                    inputs.append({'host': host_mem, 'device': device_mem, 'name': binding})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': binding})
                    self.output_name = binding

        return inputs, outputs, bindings, stream

    def _print_engine_info(self):
        print(f"🔧 TensorRT 엔진 정보:")
        print(f"   - TensorRT 버전: {trt.__version__}")
        print(f"   - 입력 크기: {self.input_shape}")
        print(f"   - 클래스 수: {len(self.class_names)}")
        print(f"   - 신뢰도 임계값: {self.conf_threshold}")
        print(f"   - IoU 임계값: {self.iou_threshold}")

        methods = []
        for m in ("execute_async_v3", "execute_async_v2", "execute_async", "execute_v2", "execute"):
            if hasattr(self.context, m):
                methods.append(m)
        print(f"   - 사용 가능한 실행 메서드: {', '.join(methods)}")

# --------------------- embedding --------------------------
    def _cpu_embedder_safe(self, chips):
        """
        HSV 히스토그램 기반 128-d 임베딩 (CPU)
        - 각 chip 당 (H:48 + S:32 + V:48) = 128 차원
        - L2 정규화 시 ε로 0분모 방지
        """
        embs = []
        for chip in chips:
            try:
                hsv = cv2.cvtColor(chip, cv2.COLOR_BGR2HSV)
                h_hist = cv2.calcHist([hsv], [0], None, [48], [0, 180]).flatten()
                s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
                v_hist = cv2.calcHist([hsv], [2], None, [48], [0, 256]).flatten()
                vec = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)  # (128,)
                norm = np.linalg.norm(vec)
                if not np.isfinite(norm) or norm == 0.0:
                    # 완전 평면/검은 이미지 등 → 안전한 기본값
                    vec[:] = 1.0 / np.sqrt(128)
                else:
                    vec /= norm
                embs.append(vec)
            except Exception:
                # 변환 실패 시에도 안전한 기본값
                embs.append(np.full(128, 1.0 / np.sqrt(128), np.float32))

        return np.vstack(embs) if embs else None

# ----------------------- Image Util -----------------------
    def _pil_to_jpeg_bytes(self, img: Image.Image, quality=90) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)

        return buf.getvalue()
    

# --------------------- track/reid util --------------------------
    def on_reid_response(self, data: dict):
        # parsing
        try:
            lid = int(data.get("local_id", data.get("track_id", -1)))
            gid = int(data.get("global_id", -1))
            crop = self._pil_to_jpeg_bytes(Image.open(io.BytesIO(base64.b64decode(data.get("crop_img")))).convert('RGB'))
        except (TypeError, ValueError):
            return
        if gid <= 0 or lid < 0:
            return
        
        with self._state_lock:
            # 이미 처리됐으면 중복 방지
            if self.local_to_detection.get(lid) or lid in self.inflight_reid:
                return
            
            # 매핑/대기 해제
            self.local_to_global[lid] = gid
            self.pending_reid.discard(lid)

            # appeared_ms 결정
            if lid not in self.track_start_ts:
                self.track_start_ts[lid] = int(time.time()*1000)
            appeared_ms = self.track_start_ts[lid]
            self.track_lifecycle.setdefault(lid, {})["status"] = "active"

            # ✅ DB 삽입 진행중 표식
            self.inflight_reid.add(lid)

        try:
            det_id = self.db.addNewDetection(uuid=str(gid), appeared_time=appeared_ms, exit_time=None, crop_img=crop)
        except Exception as e:
            # 실패 시 inflight 해제만 하고 리턴(재시도 전략은 후속)
            with self._state_lock:
                self.inflight_reid.discard(lid)
            print(f"addNewDetection failed: lid={lid}, gid={gid}, err={e}")
            return

        # 성공 후에만 확정
        with self._state_lock:
            self.local_to_detection[lid] = det_id
            self.inflight_reid.discard(lid)
            status = self.track_lifecycle.get(lid, {}).get("status")

        if status == "end_scheduled":
            self._finalize_track(lid)


    def _finalize_track(self, lid: int, *, exit_ms: int | None = None):
        """
        트랙 종료 처리: 
        - ReID 대기중이면 'end_scheduled'만 표시
        - detection_id가 있으면 exit_time 업데이트
        - 상태 정리는 여기서 일괄 처리
        """
        now_ms = int(time.time() * 1000) if exit_ms is None else exit_ms

        # 1) 상태 점검/결정은 락 안에서
        with self._state_lock:
            if lid not in self.local_to_detection and lid in self.pending_reid:
                self.track_lifecycle.setdefault(lid, {})["status"] = "end_scheduled"
                return
            det_id = self.local_to_detection.get(lid)

        # 2) DB 업데이트는 락 밖에서
        success = False
        if det_id:
            try:
                self.db.updateDetectionExitTime(det_id, now_ms, camera_id=self.camera_id)
                success = True
            except Exception as e:
                print(f"updateDetectionExitTime failed: det_id={det_id}, err={e}")

        # 3) 성공한 경우에만 정리 (락 재획득)
        if success:
            with self._state_lock:
                self.local_to_detection.pop(lid, None)
                self.local_to_global.pop(lid, None)
                self.track_start_ts.pop(lid, None)
                self.pending_reid.discard(lid)
                self.local_id_set.discard(lid)
                self.track_lifecycle.pop(lid, None)


    # --------------------- PRE/POST ---------------------------
    def preprocess(self, image):
        # 입력 shape: (N,C,H,W)
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        img_h, img_w = image.shape[:2]
        scale = min(input_w / img_w, input_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)

        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, scale, pad_x, pad_y


    def postprocess(self, outputs, scale, pad_x, pad_y, debug=False):
        t_pp0 = time.perf_counter()

        out = outputs[0]
        if out.ndim == 3:
            out = out[0]
            if out.shape[0] < out.shape[1]:
                out = out.T
        out = np.asarray(out, dtype=np.float32, order='C')

        # ---- 1) parsing (사람 클래스만 사용) ----
        t_parse0 = time.perf_counter()
        cls = out[:, 4:]                 # (N, num_classes)
        scores = cls[:, 0]               # person 점수만
        class_ids = np.zeros_like(scores, dtype=np.int32)

        mask = scores > self.conf_threshold
        if not np.any(mask):
            t_parse1 = time.perf_counter()
            t_pp1 = time.perf_counter()
            return (
                np.empty((0, 4), np.float32),
                np.empty((0,),  np.float32),
                np.empty((0,),  np.int32),
                {
                    "parse": (t_parse1 - t_parse0),
                    "nms": 0.0,
                    "postprocess": (t_pp1 - t_pp0),
                }
            )

        b = out[mask, :4]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 벡터화된 보정 + 좌상단 기준 변환
        x_c, y_c, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x = (x_c - pad_x) / scale
        y = (y_c - pad_y) / scale
        w = w / scale
        h = h / scale
        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        boxes_xywh = np.stack([x1, y1, w, h], axis=1).astype(np.float32)
        t_parse1 = time.perf_counter()

        # Top-K로 후보 캡 (선택적으로 조정)
        K = 300
        if scores.shape[0] > K:
            idx_topk = np.argpartition(scores, -K)[-K:]
            boxes_xywh = boxes_xywh[idx_topk]
            scores     = scores[idx_topk]
            class_ids  = class_ids[idx_topk]

        # ---- 2) NMS ----
        t_nms0 = time.perf_counter()
        idxs = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.astype(np.float32).tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        t_nms1 = time.perf_counter()

        if len(idxs) == 0:
            t_pp1 = time.perf_counter()
            return (
                np.empty((0, 4), np.float32),
                np.empty((0,),  np.float32),
                np.empty((0,),  np.int32),
                {
                    "parse": (t_parse1 - t_parse0),
                    "nms": (t_nms1 - t_nms0),
                    "postprocess": (t_pp1 - t_pp0),
                }
            )
        
        if isinstance(idxs, tuple):
            idxs = idxs[0]
        idxs = np.asarray(idxs).reshape(-1)

        sel_boxes = boxes_xywh[idxs]
        sel_scores = scores[idxs]
        sel_ids = class_ids[idxs]

        t_pp1 = time.perf_counter()
        timing_pp = {
            "parse": (t_parse1 - t_parse0),
            "nms": (t_nms1 - t_nms0),
            "postprocess": (t_pp1 - t_pp0),
        }
        return sel_boxes, sel_scores, sel_ids, timing_pp

    # --------------------- INFER ------------------------------
    def infer(self, image, debug=False):
        start_total = time.time()

        # CPU: preprocess
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)

        # GPU: HtoD -> TRT -> DtoH
        start_inf = time.time()
        with self._cuda():
            np.copyto(self.inputs[0]['host'], input_tensor.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            try:
                if hasattr(self.context, 'execute_async_v3'):
                    self.context.set_tensor_address(self.input_name, self.inputs[0]['device'])
                    for o in self.outputs:
                        self.context.set_tensor_address(o['name'], o['device'])
                    self.context.execute_async_v3(stream_handle=self.stream.handle)
                else:
                    self.context.execute_v2(bindings=self.bindings)
            except Exception:
                self.context.execute_v2(bindings=self.bindings)

            for o in self.outputs:
                cuda.memcpy_dtoh_async(o['host'], o['device'], self.stream)

        inference_time = time.time() - start_inf

        # CPU: postprocess (상세 타이밍 사용)
        out_data = [o['host'].reshape(o['shape']) for o in self.outputs]
        boxes, scores, class_ids, pp_timing = self.postprocess(out_data, scale, pad_x, pad_y, debug)

        total_time = time.time() - start_total
        return boxes, scores, class_ids, {
            'preprocess': total_time - inference_time - pp_timing['postprocess'],
            'inference': inference_time,
            'postprocess': pp_timing['postprocess'],
            'nms': pp_timing['nms'],
            'parse': pp_timing['parse'],
            'total': total_time
        }

            
    # core
    # DetectorAndTracker 클래스에 추가할 메서드들

    def detect_and_track(self, frame, debug=False, return_vis=False):
        # 0) Inference
        boxes, scores, class_ids, timing_info = self.infer(frame, debug)

        # 1) YOLO → tracker detections (사람만)
        h_img, w_img = frame.shape[:2]
        detections = []
        for box, score, cid in zip(boxes, scores, class_ids):
            if int(cid) != 0:
                continue
            x1, y1, w, h = box
            # 정수화 + 경계클램프
            x1 = max(0, min(int(round(x1)), w_img - 1))
            y1 = max(0, min(int(round(y1)), h_img - 1))
            w  = max(1, min(int(round(w)),  w_img - x1))
            h  = max(1, min(int(round(h)),  h_img - y1))
            detections.append(([x1, y1, w, h], float(score), int(cid)))

        # 2) Update tracks (ByteTrack/OC-SORT는 embeds 안 씀)
        tracks = self.tracker.update_tracks(detections, frame=frame)

        '''  
        # 2) 임베딩
        # 임베딩 (없으면 None)
        embeds = self.embedder(object_chips) if object_chips else None

        # 방어 로직
        if embeds is not None:
            embeds = np.asarray(embeds, dtype=np.float32)
            if embeds.ndim == 1:
                embeds = embeds[None, :]
            # 비유한값(NaN/Inf) -> 0
            embeds[~np.isfinite(embeds)] = 0.0
            # L2 정규화 (ε로 0 분모 방지)
            norms = np.linalg.norm(embeds, axis=1, keepdims=True)
            safe_norms = np.maximum(norms, 1e-6)
            embeds = embeds / safe_norms
            # 전부 0/무효면 appearance 없이 진행
            if not np.isfinite(embeds).all() or np.all(norms < 1e-6):
                embeds = None
        
        if embeds is not None and len(embeds) != len(detections):
                embeds = None
        

        # 3) Track 업데이트
        tracks = self.tracker.update_tracks(detections, embeds=embeds, frame=frame)
        '''

        #  Confirmed 신규 track만 ReID 요청용 crop 생성
        # TODO: first appearance time - end appearance time time stamp
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id

            # local_id_set: 같은 local id에 대해서 요청을 한번만 보내기 위한 set
            # pending_reid: ReID 요청을 보냈지만 아직 응답(= global_id 매핑)을 받지 못한 local_id 집합.
            if tid in self.local_id_set or tid in self.pending_reid:
                continue

            l, t, r, b = track.to_ltrb()
            l = int(round(l)); t = int(round(t)); r = int(round(r)); b = int(round(b))
            w = max(1, r - l); h = max(1, b - t)
            crop = frame[max(t,0):max(b,0), max(l,0):max(r,0)]
            if crop.size == 0:
                continue

            # Kafka 전송 (base64 JSON)
            # tracker 처음 들어왔을 때만 보내는 위치로
            idemp = f"{self.camera_id}:{tid}"
            self.result_producer.send_message(
                crop,
                track_id=tid,
                bbox=[l, t, w, h],
                class_name="person",
                encoding="base64",
                idempotency_key=idemp
            )     

            with self._state_lock:
                self.pending_reid.add(tid)
                self.local_id_set.add(tid)
                self.track_start_ts.setdefault(tid, int(time.time() * 1000))
                self.track_lifecycle[tid] = {
                    "start_ms": self.track_start_ts[tid],
                    "status": "reid_pending",
                    "gid": None,
                    "detection_id": None,
                }
        
        # 객체가 마지막으로 관측된 시간 DB에 업데이트
        curr_ids = {t.track_id for t in tracks if t.is_confirmed()}
        now_ms = int(time.time() * 1000)
        # 2-1) 현재 보이는 트랙은 데드라인 제거
        with self._state_lock:
            for lid in list(self.end_deadlines.keys()):
                if lid in curr_ids:
                    self.end_deadlines.pop(lid, None)

        # 2-2) 이번 프레임에 사라진 lid는 데드라인 설정(최초 1회)
        ended = set(self.track_start_ts.keys()) - curr_ids
        with self._state_lock:
            for lid in ended:
                self.end_deadlines.setdefault(lid, now_ms + self.end_grace_ms)

        # 2-3) 데드라인이 지난 트랙만 실제 finalize
        for lid, ddl in list(self.end_deadlines.items()):
            if now_ms >= ddl:
                self._finalize_track(lid, exit_ms=ddl)
                with self._state_lock:
                    self.end_deadlines.pop(lid, None)

        # 5) return_vis = True인 경우 시각화 프레임 return 
        if return_vis:
            vis = self.draw_tracks(frame, tracks)
        else:
            vis = None
        return vis, timing_info, boxes, scores, class_ids, tracks
    
    def draw_tracks(self, image, tracks):
        vis = image.copy()
        H, W = vis.shape[:2]
        for t in tracks:
            if not t.is_confirmed(): 
                continue
            l, t0, r, b = map(int, t.to_ltrb())
            l = max(0, min(l, W-1)); r = max(0, min(r, W-1))
            t0 = max(0, min(t0, H-1)); b = max(0, min(b, H-1))
            if r <= l or b <= t0: 
                continue
            cv2.rectangle(vis, (l, t0), (r, b), (0,255,0), 2)
            cv2.putText(vis, f'ID:{t.track_id}', (l, max(0, t0-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return vis

    def draw_detections(self, image, boxes, scores, class_ids, debug=False):
        if debug:
            print(f"🎨 그리기 디버그: 이미지={image.shape}, 박스={len(boxes)}")
        if len(boxes) == 0:
            return image

        res = image.copy()
        img_h, img_w = image.shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, w, h = box.astype(int)
            x2, y2 = x1 + w, y1 + h

            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            color = self.colors[class_id % len(self.colors)]

            cv2.rectangle(res, (x1, y1), (x2, y2), color, 3)
            label = f"{class_name}: {score:.2f}"

            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            bg_x1, bg_y1 = x1, max(0, y1 - th - base - 10)
            bg_x2, bg_y2 = min(img_w, x1 + tw + 10), y1
            cv2.rectangle(res, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(res, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return res
