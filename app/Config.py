import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_VIDEOS = os.path.join(BASE_DIR, 'extracted_videos')
TRACKING_RESULTS = os.path.join(BASE_DIR, 'yolo_crops')

class Config:
    # DB 연결 URI 구성
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql://{os.getenv('POSTGRESQL_USERNAME')}:{os.getenv('POSTGRESQL_PASSWORD')}"
        f"@{os.getenv('POSTGRESQL_URL')}:{os.getenv('POSTGRESQL_PORT')}/{os.getenv('POSTGRESQL_DATABASE')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
