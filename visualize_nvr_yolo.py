import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from poc.nvr_util.nvr_client import NVRClient

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class YoloTRT:
    def __init__(self, engine_path='poc/yolo_engine/yolo11m_fp16.engine', input_size=(640, 640)):
        self.input_size = input_size

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        h, w = self.input_size
        self.context.set_input_shape("images", (1, 3, h, w))

        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for name in self.engine:
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            dev_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(dev_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((name, dev_mem, size, dtype))
            else:
                self.outputs.append((name, dev_mem, size, dtype))

    def preprocess(self, frame):
        h, w = self.input_size
        img = cv2.resize(frame, (w, h)).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
        return tensor

    def infer(self, frame):
        tensor = self.preprocess(frame)
        input_name, input_ptr, _, _ = self.inputs[0]
        output_name, output_ptr, output_size, output_dtype = self.outputs[0]

        cuda.memcpy_htod_async(input_ptr, tensor.ravel(), self.stream)
        self.context.set_tensor_address(input_name, input_ptr)
        self.context.set_tensor_address(output_name, output_ptr)

        self.context.execute_async_v3(self.stream.handle)
        host_out = np.empty(output_size, dtype=output_dtype)
        cuda.memcpy_dtoh_async(host_out, output_ptr, self.stream)
        self.stream.synchronize()

        return host_out

    def postprocess(self, output, frame_shape, conf_thresh=0.4):
        h_img, w_img = frame_shape[:2]
        output = output.reshape(84, -1).T  # shape (8400, 84)

        boxes = output[:, :4]
        obj_conf = output[:, 4]
        class_scores = output[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        conf = obj_conf * class_conf
        keep = conf > conf_thresh

        results = []
        for i in np.where(keep)[0]:
            cx, cy, bw, bh = boxes[i]
            x1 = int((cx - bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img)
            x2 = int((cx + bw / 2) * w_img)
            y2 = int((cy + bh / 2) * h_img)
            results.append((x1, y1, x2, y2, float(conf[i]), int(class_ids[i])))

        return results


def main():
    yolo = YoloTRT()

    nvr = NVRClient()
    channel = nvr.NVRChannelList[0]
    channel.connect()

    while True:
        ret, frame = channel.cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        output = yolo.infer(frame)
        detections = yolo.postprocess(output, frame.shape)

        for x1, y1, x2, y2, score, cls in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls}:{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    channel.disconnect()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
