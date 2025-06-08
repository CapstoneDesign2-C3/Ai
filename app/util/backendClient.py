import os
import httpx

class BackendClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def post_summary(self, camera_id, summary, video_url, start_time, thumbnail_url, status):
        path = f"{self.base_url}/api/v1/video"
        data = {
            'cameraId': camera_id,
            'summary': summary,
            'videoUrl': video_url,
            'startTime': start_time,
            'thumbnailUrl': thumbnail_url,
            'status': status
        }
        response = await self.client.post(url=path, json=data)
        return response

    async def post_feature(self, re_id, feature, start_frame, end_frame, video_url, camera_id, status):
        path = f"{self.base_url}/api/v1/detected-object"
        data = {
            'reId': re_id,
            'feature': feature,
            'startFrame': start_frame,
            'endFrame': end_frame,
            'videoUrl': video_url,
            'cameraId': camera_id,
            'status': status
        }
        response = await self.client.post(url=path, json=data)
        return response

    async def close(self):
        await self.client.aclose()
