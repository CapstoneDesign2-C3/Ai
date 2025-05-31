import asyncio
from concurrent.futures import ThreadPoolExecutor
from BackendClient import *

def run_pipeline(keyframeExtractor, yolo, vlm, frame_queue, yolo_queue, video_path, camera_id):
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.submit(keyframeExtractor.process, video_path, camera_id)
        executor.submit(yolo_worker, yolo, frame_queue, yolo_queue)
        executor.submit(vlm_worker, vlm, yolo_queue)

def yolo_worker(yolo, frame_queue, yolo_queue):
    while True:
        image = frame_queue.get()
        if image is None:
            yolo_queue.put(None)
            break
        result = yolo.process(image)
        yolo_queue.put(result)

def vlm_worker(vlm, yolo_queue):
    while True:
        detection = yolo_queue.get()
        if detection is None:
            break
        result = vlm.process(detection)
        asyncio.create_task(post_vlm_summary(result))