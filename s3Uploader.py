from dotenv import load_dotenv
import boto3
import os

class s3Uploader:
    def __init__(self):
        load_dotenv(dotenv_path="env/aws.env")
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_DEFAULT_REGION")
        self.s3_bucket = os.getenv("S3_BUCKET")

        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )
    
    def upload_file(self, video_uuid, camera_id, detect_time):
        try:
            local_file_path = f"filepath/{video_uuid}.mp4"
            s3_key = f"{str(camera_id)}/{str(detect_time)}/{str(video_uuid)}"

            self.s3.upload_file(local_file_path, self.s3_bucket, s3_key)
            return True
        except Exception as e:
            return False
