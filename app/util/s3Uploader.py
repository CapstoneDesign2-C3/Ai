from app.Config import ORIGINAL_VIDEOS
import boto3
import os

class S3Uploader:
    def __init__(self, access_key, secret_key, region, bucket_name):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        self.s3_bucket = bucket_name
    
    def upload_file(self, video_uuid, camera_id, detect_time):
        print("S3Uploader upload_file start")
        try:
            local_file_path = f"{ORIGINAL_VIDEOS}/{video_uuid}.mp4"
            s3_key = f"{str(camera_id)}/{str(detect_time)}/{str(video_uuid)}"

            self.s3.upload_file(local_file_path, self.s3_bucket, s3_key)
            return True
        except Exception as e:
            return False
