from app.camera.camera import Camera

def get_camera_by_id(camera_id):
    return Camera.query.get(camera_id)