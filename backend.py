import requests
import os

url=os.getenv('BACKEND_URL')

def post_summary(cameraId, summary, videoUrl, startTime, thumbnailUrl, status):
  path = url + '/api/v1/video'
  data = {
    'cameraId':cameraId,
    'summary':summary,
    'videoUrl':videoUrl,
    'startTime':startTime,
    'thumbnailUrl':thumbnailUrl,
    'status':status
  }
  
  response = requests.post(url=path, json=data)
  return response

def post_feature(reId, feature, startFrame, endFrame, videoUrl, cameraId, status):
  path = url + '/api/v1/detected-object'
  data = {
    'reId':reId, 
    'feature':feature, 
    'startFrame':startFrame, 
    'endFrame':endFrame, 
    'videoUrl':videoUrl, 
    'cameraId':cameraId, 
    'status':status
  }

  response = requests.post(url=path, json=data)
  return response