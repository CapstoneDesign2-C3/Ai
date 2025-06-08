import av
import cv2
from io import BytesIO
import numpy as np
from fractions import Fraction
import uuid
from collections import deque
from app.Config import ORIGINAL_VIDEOS

class KeyFrameExtractor:
    def __init__(self):
        self.FLOW_STD_COEF = 0.5
        self.MSE_STD_COEF = 1
        self.MIN_MSE_STD_COEF = -3
        self.MAX_MSE_STD_COEF = 2
        self.RELAX_STD_DIVISOR = 1.2
        self.RECOVERY_STD_DIVISOR = 15
        self.Z_SCORE_TRHESHOLD = 3.5
        self.output_fps = 10  # 1초당 1프레임 처리
        self.input_segment_sec = 300  # 5분
        self.window_divisor = 15 # segment 길이의 1/15
        self.window_size = self.input_segment_sec * self.output_fps // 15
        self.mse_window_dict = {}
        self.flow_window_dict = {}

    def open_video(self, video_bytes):
        container = av.open(BytesIO(video_bytes))
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)
        fps_int = int(fps)
        return container,video_stream,fps_int

    def get_or_create_windows(self, camera_id):
        created = False
        if camera_id not in self.mse_window_dict:
            self.mse_window_dict[camera_id] = deque(maxlen=self.window_size)
            created = True

        if camera_id not in self.flow_window_dict:
            self.flow_window_dict[camera_id] = deque(maxlen=self.window_size)
            created = True
        
        return self.mse_window_dict[camera_id], self.flow_window_dict[camera_id], created
    
    def fill_windows(self, first_gray_frame, container, fps_int, flow_window, mse_window):
        frame_index = 0
        init_frames = 16
        cur_frames = []
        post_frames = []
        prev_gray_resized = first_gray_frame
        for frame in container.decode(video=0):
            if frame_index % fps_int == 0:
                gray_resized = self.convert_gray_and_resize(frame)

                if len(cur_frames) < init_frames // 2:
                    cur_flow = self.get_cur_flow(prev_gray_resized, gray_resized)
                    flow_window.append(cur_flow)

                    # 평균 프레임 누적
                    cur_frames.append(gray_resized.astype(np.float32))
                    prev_gray_resized = gray_resized

                elif len(post_frames) < init_frames // 2:
                    post_frames.append(gray_resized)

                if len(cur_frames) + len(post_frames) >= init_frames:
                    break
            frame_index += 1

        mean_frame = np.mean(np.stack(cur_frames), axis=0)
        cur_window_size = len(cur_frames)

        # ▶️ 초기 MSE 샘플 수집

        for gray_resized in post_frames:
            mse = self.get_mse(mean_frame, gray_resized)
            mse_window.append(mse)

            # 평균 프레임 업데이트 (고정된 window_size로 가정)
            mean_frame, cur_window_size = self.update_mean_frame(self.window_size, mean_frame, gray_resized, cur_window_size)
        return prev_gray_resized, mean_frame
    
    
    def set_segment_factors(self, mse_window, flow_window):
        flow_threshold = self.adapt_flow_threshold(flow_window)
        mean = np.mean(mse_window)
        std = np.std(mse_window)
        mse_threshold = mean + self.MSE_STD_COEF * std # 조정 필요 : 
        min_mse_threshold = mean + self.MIN_MSE_STD_COEF * std # 조정 필요 : 초기에 정하면 나중에 문제 발생할 여지
        max_mse_threshold = mean + self.MAX_MSE_STD_COEF * std # 조정 필요 : 초기에 정하면 나중에 문제 발생할 여지
        recovery_factor = std / self.RECOVERY_STD_DIVISOR # 조정 필요 : general 한 값이 필요
        relax_factor = std / self.RELAX_STD_DIVISOR # 조정 필요 : general 한 값이 필요

        return flow_threshold, mse_threshold, min_mse_threshold, max_mse_threshold, recovery_factor, relax_factor

    def update_mean_frame(self, window_size, mean_frame, gray_resized, cur_window_size):
        gray_resized = gray_resized.astype(np.float32)
        if cur_window_size < window_size:
            cur_window_size += 1
            mean_frame = mean_frame * ((cur_window_size - 1) / cur_window_size) + gray_resized * (1 / cur_window_size)
        else:
            mean_frame = mean_frame * ((window_size - 1) / window_size) + gray_resized * (1 / window_size)

        return mean_frame, cur_window_size


    #첫 프레임은 이전 프레임이 없음 => 그냥 prev를 첫 프레임으로
    def get_first_gray_frame(self, video_bytes):
        with av.open(BytesIO(video_bytes)) as container:
            for frame in container.decode(video=0):
                return self.convert_gray_and_resize(frame)

    def convert_gray_and_resize(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray_resized = cv2.resize(gray, (int(w * 0.25), int(h * 0.25)))
        return gray_resized

    # window 평균과 현재 프레임의 mse 계산
    def get_mse(self, mean_frame, gray_resized):
        mse = np.mean((gray_resized.astype(np.float32) - mean_frame) ** 2)
        return mse

    # 현재 프레임과 이전 프레임의 optical flow 계산
    def get_cur_flow(self, prev_gray_resized, gray_resized):
        flow = cv2.calcOpticalFlowFarneback(prev_gray_resized, gray_resized, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        cur_flow = np.mean(mag)
        return cur_flow

    # optical flow outlier 제거
    def remove_outlier(self, flow_window, cur_flow):
        median = np.median(flow_window)
        mad = np.median([np.abs(f - median) for f in flow_window])
        if mad < 1e-6:
            z_score = 0
        else:
            z_score = 0.6745 * (cur_flow - median) / mad

        if abs(z_score) <= self.Z_SCORE_TRHESHOLD:
            flow_window.append(cur_flow)
        return z_score

    def adapt_flow_threshold(self, flow_window):
        return np.mean(flow_window) + self.FLOW_STD_COEF * np.std(flow_window) # 조정 필요 : general 한 값이 필요

    def adapt_mse_threshold(self, min_mse_threshold, max_mse_threshold, mse_threshold, flow_threshold, relax_factor, recovery_factor, cur_flow, frame_gap):
        if cur_flow > flow_threshold:
            delta = relax_factor * (cur_flow - flow_threshold)
            frame_gap = 1
            mse_threshold = max(mse_threshold - delta, min_mse_threshold)
        else:
            frame_gap += 1
            recovery = recovery_factor * (frame_gap ** 0.5) # 조정 필요 : general 한 값이 필요
            mse_threshold += recovery
            mse_threshold = min(mse_threshold, max_mse_threshold)
        return mse_threshold, frame_gap

    # mse > msethreshold 이면 추출
    def extract_key_frame(self, output_fps, selected_count, mse_threshold, output, out_stream, frame, mse):
        if mse > mse_threshold:
            frame.pts = selected_count
            frame.time_base = Fraction(1, output_fps)
            packet = out_stream.encode(frame)
            if packet:
                output.mux(packet)
            selected_count += 1
        
        return selected_count
    
    def end_segment(self, output, out_stream, selected_count):
        if output:
            if selected_count > 0:
                packet = out_stream.encode(None)
                if packet:
                    output.mux(packet)
                output.close()
            else:
                output.close()

    def init_new_segment_output(self, video_stream, output_path):
        output = av.open(output_path, mode='w')
        out_stream = output.add_stream("libx264", rate=self.output_fps)
        out_stream.width = video_stream.width
        out_stream.height = video_stream.height
        out_stream.pix_fmt = 'yuv420p'
        out_stream.time_base = Fraction(1, self.output_fps)
        return output,out_stream
    
    def process(self, video_data, camera_id, frame_queue):
        video_uuid = str(uuid.uuid4())
        output_path = f"{ORIGINAL_VIDEOS}/{video_uuid}.mp4"
        video_bytes = video_data.read()
        container, video_stream, fps_int = self.open_video(video_bytes)
        output, out_stream = self.init_new_segment_output(video_stream, output_path)


        flow_window, mse_window, is_new = self.get_or_create_windows(camera_id)
        first_gray_frame = self.get_first_gray_frame(video_bytes)
        if is_new:
            prev_gray_resized, mean_frame = self.fill_windows(first_gray_frame, container, fps_int, flow_window, mse_window)
            
        cur_window_size = len(mse_window)
        flow_threshold, mse_threshold, min_mse_threshold, max_mse_threshold, recovery_factor, relax_factor = self.set_segment_factors(mse_window, flow_window)
        prev_gray_resized = first_gray_frame
        frame_gap = 0
        frame_index = 0
        segment_idx = 0
        selected_count = 0
        for frame in container.decode(video=0):
            if frame_index % fps_int != 0:
                frame_index += 1
                continue

            if frame_index % (self.input_segment_sec * fps_int) == 0:
                self.end_segment(output, out_stream, selected_count)
                frame_queue.put(output_path)

                video_uuid = str(uuid.uuid4())
                output_path = f"{ORIGINAL_VIDEOS}/{video_uuid}.mp4"
                output, out_stream = self.init_new_segment_output(video_stream, output_path)

                flow_threshold, mse_threshold, min_mse_threshold, max_mse_threshold, recovery_factor, relax_factor = self.set_segment_factors(mse_window, flow_window)
                print(f"[Segment {segment_idx}] flow_th: {flow_threshold:.2f}, mse_th: {mse_threshold:.2f}, min_mse_th: {min_mse_threshold:.2f}, "
                        f"max_mse_th: {max_mse_threshold:.2f}, relax: {relax_factor:.4f}, recover: {recovery_factor:.4f}")
                segment_idx += 1

            gray_resized = self.convert_gray_and_resize(frame)
            mse = self.get_mse(mean_frame, gray_resized)
            cur_flow = self.get_cur_flow(prev_gray_resized, gray_resized)
            self.remove_outlier(flow_window, cur_flow)
            flow_threshold = self.adapt_flow_threshold(flow_window)

            mse_threshold, frame_gap = self.adapt_mse_threshold(min_mse_threshold, max_mse_threshold, mse_threshold, flow_threshold, relax_factor, recovery_factor, cur_flow, frame_gap)
            selected_count = self.extract_key_frame(self.output_fps, selected_count, mse_threshold, output, out_stream, frame, mse)

            #다음 처리 준비
            prev_gray_resized = gray_resized.copy()
            mse_window.append(mse)
            mean_frame, cur_window_size = self.update_mean_frame(self.window_size, mean_frame, gray_resized, cur_window_size)

            frame_index += 1

        packet = out_stream.encode(None)
        if packet:
            output.mux(packet)
        
        container.close()
        output.close()

        frame_queue.put(output_path)
        frame_queue.put(None)
        return