from flask import Flask
from app.extension import db
from dotenv import load_dotenv
import os
from flask_sqlalchemy import SQLAlchemy
from app.Config import *

from app.util.keyFrameExtractor import KeyFrameExtractor
from app.util.s3Uploader import S3Uploader
from app.util.tracker import TrackerModule
from app.util.vlm import VLM
from app.util.backendClient import BackendClient

from app.routes.process_route import video_bp


load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    os.makedirs(ORIGINAL_VIDEOS, exist_ok=True)
    os.makedirs(TRACKING_RESULTS, exist_ok=True)

    app.key_frame_extractor = KeyFrameExtractor()
    app.s3_uploader = S3Uploader(
        access_key = os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region = os.getenv("AWS_DEFAULT_REGION"),
        bucket_name = os.getenv("S3_BUCKET")
    )
    app.tracker = TrackerModule()
    app.vlm = VLM()
    app.backent_client = BackendClient(os.getenv('BACKEND_URL'))

    app.register_blueprint(video_bp)

    return app