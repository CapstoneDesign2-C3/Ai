from flask import Flask, jsonify, request, make_response
from vlm import *
from s3Uploader import *

app = Flask(__name__)

@app.route('/api/v1/transfer', methods=['POST'])
def video_transfer():
  if 'file' not in request.files:
    return 'File is missing', 404
  
  video_data = request.files['file']


  result = "temp"
  response = make_response(jsonify(result))
  response.status_code = 200

  return response


@app.route('/api/v1/yolo', methods=['POST'])
def yolo():
  body = request.get_json()

  response = make_response()
  response.status_code = 200

  return response

@app.route('/api/v1/vlm_summary', methods=['POST'])
def vlm_summary():
  body = request.get_json()
  
  video_url = body.get('videoUrl')
  angle = body.get('angle')
  response = make_response(jsonify(vlm.vlm_feature(angle=angle, video_data=video_url)))
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
    