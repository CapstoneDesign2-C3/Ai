# hls-api/app.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
import subprocess, shlex, os

## example code for user
'''
GET /api/objects/{object_id}
Authorization: Bearer <JWT>

// 응답 예시
{
  "camera": "ProdNVR",
  "channel": "3",
  "timestamp": "2025-07-18T10:02:15Z"
}

재생 요청 (Playback 메타 호출)

POST /api/playback
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "camera": "ProdNVR",
  "channel": "3",
  "start": "2025-07-18T10:02:00Z",
  "end":   "2025-07-18T10:05:00Z"
}

    // 응답 예시
    {
      "playlistUrl": "https://proxy.example.com/hls/playback/ProdNVR/3.m3u8?start=...&end=..."
    }
'''
app = FastAPI()

def verify_token(request: Request):
    token = request.headers.get("Authorization", "")
    if token != "Bearer SECRET_TOKEN":
        raise HTTPException(401, "Unauthorized")

@app.get("/hls/live/{camera}/{channel}.m3u8", dependencies=[Depends(verify_token)])
async def live_hls(camera: str, channel: str):
    rtsp_url = f"rtsp://localhost:8554/{camera}/{channel}"
    # ffmpeg HLS on-the-fly
    cmd = (
      f"ffmpeg -i {rtsp_url} -c copy -f hls "
      "-hls_time 2 -hls_list_size 3 -hls_flags delete_segments "
      "pipe:1"
    )
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
    return StreamingResponse(proc.stdout, media_type="application/vnd.apple.mpegurl")

@app.get("/hls/playback/{camera}/{channel}.m3u8", dependencies=[Depends(verify_token)])
async def playback_hls(camera: str, channel: str, start: str, end: str):
    # start,end e.g. "2025-07-18T10:00:00Z"
    rtsp_url = (
      f"rtsp://localhost:8554/{camera}/{channel}"
      f"?starttime={start}&endtime={end}&subtype=0"
    )
    cmd = (
      f"ffmpeg -i {rtsp_url} -c copy -f hls "
      "-hls_time 2 -hls_list_size 0 "    # 전체 기간
      "pipe:1"
    )
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
    return StreamingResponse(proc.stdout, media_type="application/vnd.apple.mpegurl")
