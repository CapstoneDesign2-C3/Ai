from flask import Blueprint, request, jsonify
from app.util.worker import *
from threading import Thread

video_bp = Blueprint("video", __name__, url_prefix="/api/v1/video")

@video_bp.route("/process", methods=["POST"])
def process_video():
    try:
        camera_id = request.form.get("camera_id")
        if not camera_id:
            return jsonify({"error": "camera_id is required"}), 400

        video_file = request.files.get("video")
        if not video_file:
            return jsonify({"error": "video file is required"}), 400

        Thread(target=run_pipeline, args=(video_file, camera_id)).start()

        return jsonify({"message": "Video processing started"}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500