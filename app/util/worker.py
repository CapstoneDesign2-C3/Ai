import asyncio
from flask import current_app
from concurrent.futures import ThreadPoolExecutor
from app.util.backendClient import *
from queue import Queue

def run_pipeline(video_data, camera_id):
    frame_queue = Queue()
    yolo_queue = Queue()

    keyframe_extractor = current_app.key_frame_extractor
    tracker = current_app.tracker
    vlm = current_app.vlm
    backend_client = current_app.backent_client

    start_time = "2025-06-07T12:00:00Z"
    thumbnail_url = f"s3://somebucket/thumbnails/{camera_id}.jpg"
    status = "completed"

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.submit(keyframe_extractor.process, video_data, camera_id, frame_queue)
        executor.submit(yolo_worker, tracker, frame_queue, yolo_queue)
        executor.submit(
            vlm_worker,
            vlm,
            backend_client,
            yolo_queue,
            camera_id,
            start_time,
            thumbnail_url,
            status
        )


def yolo_worker(tracker, frame_queue, yolo_queue):
    while True:
        video_url = frame_queue.get()
        if video_url is None:
            yolo_queue.put(None)
            break
        result = tracker.run(video_url)
        yolo_queue.put(result)

def vlm_worker(vlm, backend_client, yolo_queue, camera_id, start_time, thumbnail_url, status):
    while True:
        video_url = yolo_queue.get()
        if video_url is None:
            break
        summary = vlm.vlm_summary(video_url)

        asyncio.run(backend_client.post_summary(
            camera_id=camera_id,
            summary=summary,
            video_url=video_url,
            start_time=start_time,
            thumbnail_url=thumbnail_url,
            status=status
        ))