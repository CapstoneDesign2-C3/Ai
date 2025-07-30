import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def initialize_cuda_context():
    try:
        cuda.init()
        device = cuda.Device(0)
        context = device.make_context()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _ = torch.zeros(1).cuda()
        return context
    except Exception as e:
        print(f"CUDA 초기화 실패: {e}")
        return None

class DetectorTracker:
    def __init__(self, engine_path: str, input_size=(640, 640), conf_thresh=0.25, max_age=10, max_iou_distance=0.4, nn_budget=50):
        self.input_size = input_size
        self.conf_thresh = conf_thresh

        print(f"Initializing DetectorTracker with engine: {engine_path}")

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        h, w = self.input_size
        try:
            self.context.set_input_shape('images', (1, 3, h, w))
        except Exception:
            name = self.engine.get_tensor_name(0)
            self.context.set_input_shape(name, (1, 3, h, w))

        self.bindings, self.inputs, self.outputs = [], [], []
        self.stream = cuda.Stream()

        for name in self.engine:
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            dev_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(dev_mem))
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append((name, dev_mem, size, dtype))
                self.input_host_mem = cuda.pagelocked_empty(size, dtype)
            else:
                self.outputs.append((name, dev_mem, size, dtype))
                self.output_host_mem = cuda.pagelocked_empty(size, dtype)

        self.tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, nn_budget=nn_budget)
        print("DetectorTracker initialized successfully")

    def preprocess(self, frame: np.ndarray):
        h, w = self.input_size
        self.orig_h, self.orig_w = frame.shape[:2]
        img = cv2.resize(frame, (w, h)).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)

    def infer_async(self, frame: np.ndarray):
        tensor = self.preprocess(frame)
        np.copyto(self.input_host_mem, tensor.ravel())

        in_name, in_ptr, _, _ = self.inputs[0]
        out_name, out_ptr, _, _ = self.outputs[0]

        cuda.memcpy_htod_async(in_ptr, self.input_host_mem, self.stream)
        self.context.set_tensor_address(in_name, in_ptr)
        self.context.set_tensor_address(out_name, out_ptr)
        self.context.execute_async_v3(self.stream.handle)
        return out_ptr

    def postprocess(self, output: np.ndarray, frame_shape):
        h_img, w_img = frame_shape[:2]
        try:
            if len(output.shape) == 1:
                if output.size % 84 == 0:
                    num_boxes = output.size // 84
                    data = output.reshape(84, num_boxes).T
                else:
                    print(f"Unexpected output size: {output.size}")
                    return []
            elif len(output.shape) == 3:
                data = output[0].T
            elif len(output.shape) == 2:
                data = output.T
            else:
                print(f"Unexpected output shape: {output.shape}")
                return []

            boxes = data[:, :4]
            scores = data[:, 4:]
            cls_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)
            keep = (max_scores > self.conf_thresh) & (cls_ids == 0)

            results = []
            if not np.any(keep):
                return results

            input_h, input_w = self.input_size
            scale_x = w_img / input_w
            scale_y = h_img / input_h

            for i in np.where(keep)[0]:
                cx, cy, bw, bh = boxes[i]
                score = float(max_scores[i])
                cls_id = int(cls_ids[i])

                cx_scaled = cx * scale_x
                cy_scaled = cy * scale_y
                bw_scaled = bw * scale_x
                bh_scaled = bh * scale_y

                x1 = int(cx_scaled - bw_scaled / 2)
                y1 = int(cy_scaled - bh_scaled / 2)
                x2 = int(cx_scaled + bw_scaled / 2)
                y2 = int(cy_scaled + bh_scaled / 2)

                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img - 1))
                y2 = max(0, min(y2, h_img - 1))

                if x2 > x1 + 5 and y2 > y1 + 5:
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'score': score,
                        'class': cls_id
                    })

            return results

        except Exception as e:
            print(f"[Error] postprocess failed: {e}")
            return []

    def detect_and_track(self, frame: np.ndarray):
        try:
            out_ptr = self.infer_async(frame)
            self.stream.synchronize()
            cuda.memcpy_dtoh(self.output_host_mem, out_ptr)
            detections = self.postprocess(self.output_host_mem.copy(), frame.shape)

            det_list = [([*d['bbox']], d['score'], d['class']) for d in detections]
            tracks = self.tracker.update_tracks(det_list, frame=frame)

            results = []
            for track in tracks:
                if not track.is_confirmed():
                    continue

                l, t, r, b = track.to_ltrb()
                track_id = getattr(track, 'track_id', None)
                score = getattr(track, 'det_conf', 0.5)
                cls = getattr(track, 'det_class', 0)
                is_new_value = getattr(track, 'age', 0) == 1

                results.append({
                    'local_id': track_id,
                    'bbox': [int(l), int(t), int(r), int(b)],
                    'score': score,
                    'class': cls,
                    'is_new': is_new_value
                })
            return results

        except Exception as e:
            print(f"[Error] detect_and_track failed: {e}")
            return []
    
    def detect_only(self, frame: np.ndarray):
        """
        추적 없이 검출만 수행 (디버깅용)
        """
        try:
            output = self.infer(frame)
            detections = self.postprocess(output, frame.shape)
            return detections
        except Exception as e:
            print(f"[Error] detect_only failed: {e}")
            return []

    def visualize_results(self, frame: np.ndarray, results: list):
        """
        결과를 프레임에 시각화
        """
       
        annotated_frame = frame.copy()

        for result in results:
            try:
                bbox = result.get('bbox', [])
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = bbox
                
                # 바운딩 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 라벨 텍스트 구성 (None 값 체크)
                label_parts = ["Person"]
                
                # ID 추가
                local_id = result.get('local_id')
                if local_id is not None:
                    label_parts.append(f"ID:{local_id}")
                
                # 점수 추가
                score = result.get('score')
                if score is not None:
                    label_parts.append(f"{score:.2f}")
                
                # NEW 표시
                is_new = result.get('is_new', False)
                if is_new:
                    label_parts.append("NEW")
                
                label = " ".join(label_parts)
                
                # 라벨 배경
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # 라벨 텍스트
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
            except Exception as viz_error:
                print(f"⚠️  Visualization error for result: {viz_error}")
                continue
        
        return annotated_frame
    
    def get_engine_info(self):
        """
        엔진 정보 반환 (디버깅용)
        """
        info = {
            'input_size': self.input_size,
            'conf_thresh': self.conf_thresh,
            'num_inputs': len(self.inputs),
            'num_outputs': len(self.outputs)
        }
        return info

    def __del__(self):
        """
        메모리 정리
        """
        try:
            # GPU 메모리 해제
            if hasattr(self, 'inputs'):
                for _, dev_mem, _, _ in self.inputs:
                    dev_mem.free()
            if hasattr(self, 'outputs'):
                for _, dev_mem, _, _ in self.outputs:
                    dev_mem.free()
        except Exception as e:
            print(f"[Warning] Memory cleanup failed: {e}")
            pass
