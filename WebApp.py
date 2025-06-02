from flask import Flask, jsonify, request, make_response
from vlm import *
from s3Uploader import *
from tracker import *
import os
import uuid
import json

app = Flask(__name__)

# 모델 로드
vlm = VLM()
tracker = TrackerModule()

# 영상 경로
DATA_ROOT = './data'
ORIGINAL_VIDEOS = os.path.join(DATA_ROOT, 'original_videos')
TRACKING_RESULTS = os.path.join(DATA_ROOT, 'tracking_results')
VLM_SUMMARIES = os.path.join(DATA_ROOT, 'vlm_summaries')

# 디렉토리 생성
os.makedirs(ORIGINAL_VIDEOS, exist_ok=True)
os.makedirs(TRACKING_RESULTS, exist_ok=True)
os.makedirs(VLM_SUMMARIES, exist_ok=True)


@app.route('/api/v1/transfer', methods=['POST'])
def video_transfer():
    if 'file' not in request.files:
        return 'File is missing', 404
    
    video_data = request.files['file']
    video_uuid = str(uuid.uuid4())  # UUID 생성
    save_path = os.path.join(ORIGINAL_VIDEOS, f"{video_uuid}.mp4")
    video_data.save(save_path)

    # 반환: video_uuid
    response = make_response(jsonify({
        'video_uuid': video_uuid
    }))
    response.status_code = 200

    return response

# Step 2: YOLO Tracker 실행
@app.route('/api/v1/yolo', methods=['POST'])
def yolo():
    body = request.get_json()
    video_uuid = body.get('video_uuid')
    if not video_uuid:
        return 'video_uuid is required', 400
    
    video_path = os.path.join(ORIGINAL_VIDEOS, f"{video_uuid}.mp4")
    if not os.path.exists(video_path):
        return 'Video path invalid', 400

    # Tracker 결과 output 디렉토리 생성
    tracker_output_dir = os.path.join(TRACKING_RESULTS, video_uuid)
    os.makedirs(tracker_output_dir, exist_ok=True)

    # Tracker 실행
    print(f"Running tracker on: {video_path}")
    tracker.run(video_path, output_dir=tracker_output_dir)
    print("Tracker completed.")

    response = make_response(jsonify({
        'status': 'Tracker completed',
        'video_uuid': video_uuid
    }))
    response.status_code = 200

    return response

@app.route('/api/v1/vlm_summary', methods=['POST'])
def vlm_summary():
    body = request.get_json()
    video_uuid = body.get('video_uuid')
    angle = body.get('angle')

    if not video_uuid:
        return 'video_uuid is required', 400
    
    video_path = os.path.join(ORIGINAL_VIDEOS, f"{video_uuid}.mp4")
    if not os.path.exists(video_path):
        return 'Video path invalid', 400

    print(f"Running VLM summary on: {video_path}, angle: {angle}")
    summary_result = vlm.vlm_summary(angle, video_path)

    # summary 저장
    summary_save_path = os.path.join(VLM_SUMMARIES, f"{video_uuid}.json")
    with open(summary_save_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary_result}, f, ensure_ascii=False, indent=2)

    response = make_response(jsonify({
        'summary': summary_result,
        'video_uuid': video_uuid
    }))
    response.status_code = 200

    return response

@app.route('/api/v1/vlm_feature', methods=['POST'])
def vlm_feature():
  body = request.get_json()
  image_url = body.get('imageUrl')
  response = make_response(jsonify(vlm.vlm_feature(image_data=image_url)))
  response.status_code = 200

  return response

@app.route('/api/v1/s3_upload', methods=['POST'])
def s3_upload():
  body = request.get_json()

  video_uuid = body['video_uuid']
  camera_id = body['camera_id']
  detect_time = body['detect_time']

  response = make_response()
  if s3uploder.upload_file(video_uuid, camera_id, detect_time):
    response.status_code = 200
  else:
    response.status_code = 400

  return response


if __name__ == '__main__':
    vlm = VLM()
    s3uploder = S3Uploader()
    app.run(host="0.0.0.0", port=5000)
    