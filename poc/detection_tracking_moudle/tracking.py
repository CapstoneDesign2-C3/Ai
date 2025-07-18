import build_engine

# 1) TensorRT 엔진 로드
engine = load_trt_engine("yolov11m_fp16.engine")
context = engine.create_execution_context()

# 2) Tracker 인스턴스 생성 (SORT)
from sort import Sort
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# 처음 나타난 객체인지를 판별하는 set
seen_ids = set()

while True:
    frame = capture.read()
    inp = preprocess(frame)                # resize, normalize, to_tensor
    outputs = infer_with_trt(context, inp) # TensorRT inference
    detections = yolo_decode(outputs)      # NMS + class filtering(person only)

    # detections: np.ndarray of shape (N, 5) → [x1,y1,x2,y2,score]
    tracks = tracker.update(detections)    # returns np.ndarray (M,6): [x1,y1,x2,y2,track_id,score]


    # 화면에 표시하는 기능
    '''
    for x1,y1,x2,y2,track_id,score in tracks:
        cv2.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)
        cv2.putText(frame, str(int(track_id)), (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    '''

    # 처음 등장한 track_id 만을 보낸다.
    current_ids = { int(t[4]) for t in tracks }
    new_ids = current_ids - seen_ids

    for x1,y1,x2,y2,track_id,score in tracks:
        if track_id in new_ids:
            crop = frame[y1:y2, x1:x2]
            send_to_reid(crop, track_id)
    seen_ids |= new_ids

    # display(frame)

