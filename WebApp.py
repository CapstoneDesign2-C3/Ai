from flask import Flask, jsonify, request, make_response
from vlm import *

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
  
  response = make_response(jsonify(vlm.vlm_summary(video_data=body)))
  response.status_code = 200

  return response

@app.route('/api/v1/vlm_feature', methods=['POST'])
def vlm_feature():
  body = request.get_json()

  response = make_response(jsonify(vlm.vlm_feature(video_data=body)))
  response.status_code = 200

  return response

if __name__ == '__main__':
    vlm = VLM()
    app.run(host="0.0.0.0", port=5000)
    