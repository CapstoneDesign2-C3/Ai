# from flask import Flask, jsonify, request, make_response
# from app.util.vlm import *
# from app.util.s3Uploader import *
# from app.util.tracker import *
# from app.util.BackendClient import *
# import os
# import uuid
# import json
# from app.util.Worker import *
# from app.util.KeyframeExtractor import *

# load_dotenv()

# app = Flask(__name__)

# db_user = os.getenv("POSTGRESQL_USERNAME")
# db_pass = os.getenv("POSTGRESQL_PASSWORD")
# db_host = os.getenv("POSTGRESQL_URL")
# db_port = os.getenv("POSTGRESQL_PORT")
# db_name = os.getenv("POSTGRESQL_DATABASE")

# # 모델 로드
# vlm = VLM()
# tracker = TrackerModule()

# @app.route('/api/v1/transfer', methods=['POST'])
# def video_transfer():
#     if 'file' not in request.files:
#         return 'File is missing', 404
    
#     video_data = request.files['file']
#     video_uuid = str(uuid.uuid4())  # UUID 생성 여기에 날짜나 카메라 번호 등 입력하면 좋을 듯 함. format 정해야 할 듯.
#     save_path = os.path.join(ORIGINAL_VIDEOS, f"{video_uuid}.mp4")
#     video_data.save(save_path)

#     # 반환: json화된 video_uuid
#     response = make_response(jsonify({
#         'video_uuid': video_uuid
#     }))
#     response.status_code = 200

#     return response

# # YOLO Tracker 실행
# @app.route('/api/v1/yolo', methods=['POST'])
# def yolo():
#     body = request.get_json()
#     video_uuid = body.get('video_uuid') # 일단 body에 담는 방식으로 작성함.
#     if not video_uuid:
#         return 'video_uuid is required', 400
    
#     video_path = os.path.join(ORIGINAL_VIDEOS, f"{video_uuid}.mp4")
#     if not os.path.exists(video_path):
#         return 'Video path invalid', 400

#     # Tracker 결과 output 디렉토리 생성
#     tracker_output_dir = os.path.join(TRACKING_RESULTS, video_uuid) # 이 부분 날짜/카메라 번호로 수정 필요함.
#     os.makedirs(tracker_output_dir, exist_ok=True)

#     # Tracker 실행
#     print(f"Running tracker on: {video_path}")
#     tracker.run(video_path, output_dir=tracker_output_dir)
#     print("Tracker completed.")

#     response = make_response(jsonify({
#         'status': 'Tracker completed',
#         'video_uuid': video_uuid
#     }))
#     response.status_code = 200

#     return response

# @app.route('/api/v1/vlm_summary', methods=['POST'])
# def vlm_summary():
#     body = request.get_json()
#     video_uuid = body.get('video_uuid')
    
#     angle = body.get('angle')

#     if not video_uuid:
#         return 'video_uuid is required', 400
    
#     video_path = os.path.join(ORIGINAL_VIDEOS, f"{video_uuid}.mp4")
#     if not os.path.exists(video_path):
#         return 'Video path invalid', 400

#     print(f"Running VLM summary on: {video_path}, angle: {angle}")
#     summary = vlm.vlm_summary(angle, video_path)

#     cameraId = body.get('cameraId')
#     startTime = body.get('startTime')
#     thumbnailUrl = 'temp' # thumbnailUrl 없어서 일단 temp로
#     status = body.get('status')

#     response = post_summary(cameraId=cameraId, 
#                  summary=summary, 
#                  videoUrl=video_path, 
#                  startTime=startTime, 
#                  thumbnailUrl=thumbnailUrl, 
#                  status=status)

#     return response

# @app.route('/api/v1/vlm_feature', methods=['POST'])
# def vlm_feature():
#   body = request.get_json()
#   image_url = body.get('imageUrl')
#   reId = body.get('reId')
#   startFrame = body.get('startFrame')
#   endFrame = body.get('endFrame')
#   videoUrl = body.get('videoUrl')
#   cameraId = body.get('cameraId')
#   status = body.get('status')
#   feature = vlm.vlm_feature(image_data=image_url)
  
#   response = post_feature(reId=reId, 
#                         feature=feature, 
#                         startFrame=startFrame,
#                         endFrame=endFrame, 
#                         videoUrl=videoUrl, 
#                         cameraId=cameraId,
#                         status=status)

#   return response

# @app.route('/api/v1/s3_upload', methods=['POST'])
# def s3_upload():
#   body = request.get_json()

#   video_uuid = body['video_uuid']
#   camera_id = body['camera_id']
#   detect_time = body['detect_time']

#   response = make_response()
#   if s3uploder.upload_file(video_uuid, camera_id, detect_time):
#     response.status_code = 200
#   else:
#     response.status_code = 400

#   return response

# if __name__ == '__main__':
#     s3uploder = S3Uploader()
#     app.run(host="0.0.0.0", port=5000)
    