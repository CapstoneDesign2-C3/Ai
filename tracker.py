# tracker.py
from Yolov7_StrongSORT_OSNet.track import run

class TrackerModule:
    def __init__(self):
        # YOLO 모델 경로 (원하면 인자로도 받을 수 있음)
        self.yolo_weights = 'Yolov7_StrongSORT_OSNet/yolov7/yolov7.pt'

    def run(self, video_path, output_dir):
        print(f"[Tracker] Running on video: {video_path}")
        print(f"[Tracker] Output dir: {output_dir}")

        run(
            yolo_weights=[self.yolo_weights],  # 리스트로 줘야 맞음
            source=video_path,
            save_vid=True,
            project=output_dir,
            name='',
            exist_ok=True,
            device='0'  # GPU 사용 (필요 시 'cpu' 도 가능)
        )

        print("[Tracker] Done.")
