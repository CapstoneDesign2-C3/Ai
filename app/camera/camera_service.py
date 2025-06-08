from app.camera.camera_repository import get_camera_by_id

def find_camera_by_id(camera_id):
    camera = get_camera_by_id(camera_id)
    if not camera:
        return {"error" : "Not found"}, 404
    return {
        "id": camera.id,
        "angle": camera.angle
    }