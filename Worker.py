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
        print("VLM Output:", result)