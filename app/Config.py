import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_VIDEOS = os.path.join(BASE_DIR, 'extracted_videos')
TRACKING_RESULTS = os.path.join(BASE_DIR, 'yolo_crops')

class Config:
    # DB 연결 URI 구성
    print(os.getenv('POSTGRESQL_USERNAME'))  # myuser
    print(os.getenv('POSTGRESQL_PASSWORD'))  # mypass
    print(os.getenv('POSTGRESQL_URL'))  # localhost
    print(os.getenv('POSTGRESQL_PORT'))  # 5432
    print(os.getenv('POSTGRESQL_DATABASE'))
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql://{os.getenv('POSTGRESQL_USERNAME')}:{os.getenv('POSTGRESQL_PASSWORD')}"
        f"@{os.getenv('POSTGRESQL_URL')}:{os.getenv('POSTGRESQL_PORT')}/{os.getenv('POSTGRESQL_DATABASE')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
