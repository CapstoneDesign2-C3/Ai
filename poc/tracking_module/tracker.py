import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
# pycuda.autoinit 제거하고 수동으로 CUDA 초기화
from deep_sort_realtime.deepsort_tracker import DeepSort

class DetectorTracker:
    """
    Uses a TensorRT YOLO engine for detection and DeepSort for tracking.
    """
    def __init__(self,
                 engine_path="/home/hiperwall/Ai_modules/Ai/poc/yolo11m_fp16.engine",
                 input_size=(640, 640),
                 max_age=30,
                 max_iou_distance=0.7,
                 nn_budget=100):
        
        # CUDA 초기화 및 컨텍스트 생성
        try:
            cuda.init()
            self.cuda_ctx = cuda.Device(0).make_context()
            print("[*] CUDA context initialized successfully")
        except Exception as e:
            print(f"[Error] CUDA initialization failed: {e}")
            raise
            
        self.input_size = input_size
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            print(f"[*] TensorRT engine loaded: {engine_path}")
        except Exception as e:
            print(f"[Error] Failed to load TensorRT engine: {e}")
            raise

        # Set input shape explicitly (TensorRT 8.6+)
        h, w = self.input_size
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, (1, 3, h, w))

        # Allocate device buffers
        try:
            for binding in self.engine:
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
                shape = self.context.get_tensor_shape(binding)
                size = trt.volume(shape)
                dev_mem = cuda.mem_alloc(size * dtype().itemsize)
                self.bindings.append(int(dev_mem))
                if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    self.inputs.append((binding, dev_mem, size, dtype))
                else:
                    self.outputs.append((binding, dev_mem, size, dtype))
            print("[*] GPU memory allocated successfully")
        except Exception as e:
            print(f"[Error] GPU memory allocation failed: {e}")
            raise

        # DeepSort Tracker
        self.tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            nn_budget=nn_budget
        )

        print("=== TensorRT Engine IO Tensors ===")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)   # INPUT / OUTPUT
            dtype = self.engine.get_tensor_dtype(name)
            try:
                shape = self.context.get_tensor_shape(name)
            except Exception:
                shape = self.engine.get_tensor_shape(name)  # fallback for static shape
            print(f"{mode.name}: {name} | dtype: {dtype} | shape: {shape}")
    
    def __del__(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'cuda_ctx'):
                self.cuda_ctx.pop()
                print("[*] CUDA context cleaned up")
        except Exception as e:
            print(f"[Warning] CUDA context cleanup failed: {e}")

    def preprocess(self, frame: np.ndarray):
        h, w = self.input_size
        img = cv2.resize(frame, (w, h)).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
        return tensor

    def infer(self, frame: np.ndarray):
        try:
            # CUDA 컨텍스트 푸시 (멀티프로세싱 환경에서 필요)
            self.cuda_ctx.push()
            
            data = self.preprocess(frame)  # shape: (1, 3, 640, 640)
            input_name = self.engine.get_tensor_name(0)  # "images"
            output_name = self.engine.get_tensor_name(1)  # "output0"

            # Get input/output info
            input_binding = self.inputs[0]
            output_binding = self.outputs[0]

            # Copy input to device
            cuda.memcpy_htod_async(input_binding[1], data.ravel(), self.stream)

            # Set tensor addresses explicitly
            self.context.set_tensor_address(input_name, input_binding[1])
            self.context.set_tensor_address(output_name, output_binding[1])

            # Run inference
            success = self.context.execute_async_v3(self.stream.handle)
            if not success:
                raise RuntimeError("TensorRT inference failed")

            # Retrieve output
            host_out = np.empty(output_binding[2], dtype=output_binding[3])
            cuda.memcpy_dtoh_async(host_out, output_binding[1], self.stream)
            self.stream.synchronize()

            return host_out
            
        except Exception as e:
            print(f"[Error] Inference failed: {e}")
            return np.array([])  # 빈 배열 반환
        finally:
            # CUDA 컨텍스트 팝
            self.cuda_ctx.pop()

    def postprocess(self, output: np.ndarray, orig_shape):
        if output.size == 0:
            return []  # 빈 결과 반환
            
        h_img, w_img = orig_shape
        output = output.reshape(84, -1).T  # shape: (8400, 84)

        boxes = output[:, 0:4]
        obj_conf = output[:, 4]
        class_scores = output[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        conf = obj_conf * class_conf

        conf_threshold = 0.4
        keep = conf > conf_threshold

        boxes = boxes[keep]
        conf = conf[keep]
        class_ids = class_ids[keep]

        # Convert xywh to x1y1x2y2
        results = []
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            x1 = int((cx - w / 2) * w_img / 640 * w_img)  # 스케일링 수정
            y1 = int((cy - h / 2) * h_img / 640 * h_img)
            x2 = int((cx + w / 2) * w_img / 640 * w_img)
            y2 = int((cy + h / 2) * h_img / 640 * h_img)
            
            # 경계 체크
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(conf[i]),
                'class': int(class_ids[i])
            })

        return results

    def detect_and_track(self, frame: np.ndarray):
        try:
            # Detection 수행
            dets = self.postprocess(self.infer(frame), frame.shape[:2])
            
            # Detection 결과 로깅 (처음 10개 프레임만)
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 1
                
            if self.frame_count <= 10 or self.frame_count % 100 == 0:
                print(f"[Detection] Frame {self.frame_count}: Found {len(dets)} detections")
                for i, d in enumerate(dets):
                    if i < 3:  # 처음 3개만 로깅
                        print(f"  Detection {i}: bbox={d['bbox']}, score={d['score']:.3f}, class={d['class']}")
            
            det_list = []
            for d in dets:
                x1, y1, x2, y2 = d['bbox']
                det_list.append(([x1, y1, x2, y2], d['score'], d['class']))

            # Tracking 수행
            tracks = self.tracker.update_tracks(det_list, frame=frame)
            
            # Tracking 결과 처리
            results = []
            new_tracks = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                l, t, r, b = track.to_ltrb()
                is_new = track.is_new()
                if is_new:
                    new_tracks += 1
                    
                results.append({
                    'local_id': track.track_id,
                    'bbox': [int(l), int(t), int(r), int(b)],
                    'score': track.det_conf,
                    'class': track.det_class,
                    'is_new': is_new
                })
            
            # Tracking 결과 로깅
            if self.frame_count <= 10 or self.frame_count % 100 == 0:
                print(f"[Tracking] Frame {self.frame_count}: {len(results)} confirmed tracks, {new_tracks} new")
                
            return results
            
        except Exception as e:
            print(f"[Error] detect_and_track failed on frame {getattr(self, 'frame_count', 0)}: {e}")
            import traceback
            traceback.print_exc()
            return []