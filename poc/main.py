from nvr_util import nvr_client, exceptions
from kafka_util import producers
from tracking_module import detector

def test_connect_nvr():
    client = nvr_client.NVRClient()
    channel = client.NVRChannelList[0]
    channel.connect()
    channel.receive()

def connect_nvr():
    client = nvr_client.NVRClient()
    for channel in client.NVRChannelList:
        realtime_channel_worker(channel=channel, camera_id=channel.camera_id)
        
def realtime_channel_worker(channel, camera_id):
    frame_producer = producers.FrameProducer(cameraID=camera_id)
    while True:
        try:
            # 프레임 가져오기
            ret, frame = channel.cap.read()
            if not ret:
                raise exceptions.NVRRecieveError("프레임 읽기 실패")
            # send to detector with kafka
            frame_producer.send_message(cameraID=camera_id, frame=frame)
        except:
            print("Error: read frame (from channel %d)", camera_id)

if __name__ == '__main__':
    test_connect_nvr()