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

from dotenv import load_dotenv
from kafka_util import consumers, producers
from deep_sort_realtime.deepsort_tracker import DeepSort

# (extract_crop_from_frame에서 사용)
from PIL import Image
import io, base64


class DetectorAndTracker:
    def __init__(self, class_names_path=None, conf_threshold=0.25, iou_threshold=0.45, cameraID=None):
        """
        TensorRT YOLOv11 + DeepSORT (PyTorch) 통합
        - Torch가 만든 primary CUDA context를 PyCUDA/TensorRT가 공유
        - GPU 만지는 구간에서만 push/pop (누수 방지)
        """
        load_dotenv(override=True)

        # --- Kafka ---
        self.frame_consumer = consumers.FrameConsumer()
        self.result_producer = producers.DetectedResultProducer()
        # self.track_result_producer = producers.TrackResultProducer(cameraID=cameraID)

        # --- 로컬 ID 세트 ---
        self.local_id_set = set()

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

        # --- DeepSORT ---
        self.tracker = DeepSort(max_age=5)
        
        # 안전한 CPU 임베더(0벡터/NaN 방지)
        self.embedder = self._cpu_embedder_safe

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
        if debug:
            print(f"🔍 후처리 디버그: 출력 {len(outputs)}개")

        out = outputs[0]
        if len(out.shape) == 3:
            out = out[0]
            if out.shape[0] < out.shape[1]:
                out = out.T  # (84,8400)->(8400,84)

        boxes, scores, class_ids = [], [], []
        for i, det in enumerate(out):
            x_center, y_center, width, height = det[:4]
            class_confs = det[4:]
            max_conf = float(np.max(class_confs))
            if max_conf <= self.conf_threshold:
                continue
            class_id = int(np.argmax(class_confs))

            # pad/scale 보정
            x_center = (x_center - pad_x) / scale
            y_center = (y_center - pad_y) / scale
            width = width / scale
            height = height / scale

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            boxes.append([x1, y1, width, height])
            scores.append(max_conf)
            class_ids.append(class_id)

        if not boxes:
            return [], [], []

        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                   self.conf_threshold, self.iou_threshold)
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0] if len(indices) > 0 else []
            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                return boxes[indices], scores[indices], class_ids[indices]

        return [], [], []

    # --------------------- INFER ------------------------------
    def infer(self, image, debug=False):
        start_total = time.time()

        # CPU: 전처리
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
            except Exception as e:
                print(f"추론 실행 오류: {e}")
                self.context.execute_v2(bindings=self.bindings)

            for o in self.outputs:
                cuda.memcpy_dtoh_async(o['host'], o['device'], self.stream)

        inference_time = time.time() - start_inf

        # CPU: 후처리
        out_data = [o['host'].reshape(o['shape']) for o in self.outputs]
        boxes, scores, class_ids = self.postprocess(out_data, scale, pad_x, pad_y, debug)

        total_time = time.time() - start_total
        return boxes, scores, class_ids, {
            'preprocess': total_time - inference_time,
            'inference': inference_time,
            'postprocess': 0.0,  # 필요시 분리 계산 가능
            'total': total_time
        }

    # --------------------- MISC -------------------------------
    def extract_crop_from_frame(self, frame: np.ndarray, bbox: list) -> str:
        """Extract person crop and return base64 JPEG"""
        try:
            x, y, w, h = bbox
            h_img, w_img = frame.shape[:2]
            x = int(round(max(0, min(x, w_img - 1))))
            y = int(round(max(0, min(y, h_img - 1))))
            w = int(round(max(1, min(w_img - x, w))))
            h = int(round(max(1, min(h_img - y, h))))
            crop = frame[y:y + h, x:x + w]

            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            crop_pil.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Failed to extract crop: {e}")
            return ""
    
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

    def detect_and_track(self, frame, debug=False):
        # 0) Inference
        boxes, scores, class_ids, timing_info = self.infer(frame, debug)
        if debug:
            print(f"Inference timing: {timing_info}")

        # 1) DeepSORT 입력으로 변환 (int + 경계 클램프)
        h_img, w_img = frame.shape[:2]
        detections = []
        object_chips = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, w, h = box
            x1 = int(round(x1)); y1 = int(round(y1))
            w = int(round(w)); h = int(round(h))
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            w = max(1, min(w_img - x1, w))
            h = max(1, min(h_img - y1, h))

            detections.append(([x1, y1, w, h], float(score), int(class_id)))
            chip = frame[y1:y1 + h, x1:x1 + w]
            if chip.size > 0:
                object_chips.append(chip)

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

        # 4) Confirmed 신규 track만 ReID 요청용 crop 생성
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            if tid in self.local_id_set:
                continue

            l, t, r, b = track.to_ltrb()
            l = int(round(l)); t = int(round(t)); r = int(round(r)); b = int(round(b))
            l = max(0, min(l, w_img - 1)); r = max(0, min(r, w_img))
            t = max(0, min(t, h_img - 1)); b = max(0, min(b, h_img))
            if r <= l or b <= t:
                continue

            crop = frame[t:b, l:r]
            if crop.size == 0:
                continue

            # self.track_result_producer.send_message(crop)  # 필요시 전송
            self.local_id_set.add(tid)

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
