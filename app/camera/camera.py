from app.extension import db

class Camera(db.Model):
    __table__name = 'camera'

    id = db.Column('camera_id', db.Integer, primary_key=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    angle = db.Column(db.String)
    status = db.Column(db.String)  # Enum은 문자열로 저장됨

    # Address는 @Embedded였으므로 그냥 평탄화된 필드들로 받아야 함
    address1 = db.Column(db.String)
    address2 = db.Column(db.String)