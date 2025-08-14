import psycopg2
from datetime import datetime

class PostgreSQL:
    def __init__(self, host, dbname, user, password, port, *, autocommit=False):
        self.db = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port
        )
        self.db.autocommit = autocommit
        print('DB connnect success')


    def addNewDetectedObject(self, uuid, crop_img, feature="", code_name="사람"):
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT code_id FROM event_codes WHERE code_name = %s",
                (code_name,)
            )
            result = cursor.fetchone()
            if result is None:
                raise ValueError(f"코드 이름({code_name})이 event_codes에 없습니다.")
            code_id = result[0]

            cursor.execute(
                """
                INSERT INTO detected_object (uuid, crop_img, feature, code_id)
                VALUES (%s, %s, %s, %s)
                """,
                (uuid, psycopg2.Binary(crop_img), feature, code_id)
            )
        self.db.commit()
        print(f'uuid : {uuid} is insert')   # for debugging


    def addNewDetection(self, uuid, appeared_time, exit_time=None):
        """
        appeared_time: datetime
        exit_time: Optional[datetime] (미정이면 None)
        """
        try:
            with self.db.cursor() as cursor:
                # 트랜잭션 시작
                cursor.execute("BEGIN;")

                cursor.execute(
                    "SELECT id FROM detected_object WHERE uuid = %s FOR SHARE",
                    (uuid,)
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"uuid({uuid})가 detected_object에 없습니다.")
                detected_object_id = row[0]

                utc_appeared_time = datetime.utcfromtimestamp(appeared_time / 1000.0)
                cursor.execute(
                    """
                    INSERT INTO detection (detected_object_id, appeared_time, exit_time)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (detected_object_id, utc_appeared_time, exit_time)
                )
                det_id = cursor.fetchone()[0]
                print(f'uuid : {uuid} is updated: add new detection')   # for debugging
                cursor.execute("COMMIT;")
                return det_id
        except Exception:
            with self.db.cursor() as c2:
                c2.execute("ROLLBACK;")
            raise
        
    # 퇴장 시간 업데이트를 위한 메서드
    def updateDetectionExitTime(self, detection_id, exit_time):
        """
        exit_time: epoch ms(int) 또는 datetime
        """
        if isinstance(exit_time, (int, float)):
            exit_dt = datetime.utcfromtimestamp(exit_time / 1000.0)
        elif isinstance(exit_time, datetime):
            exit_dt = exit_time
        else:
            raise TypeError("exit_time must be epoch ms or datetime")

        with self.db.cursor() as cursor:
            cursor.execute(
                "UPDATE detection SET exit_time = GREATEST(COALESCE(exit_time, %s), %s) WHERE id = %s;",
                (exit_dt, detection_id)
            )
        self.db.commit()
        print(f'updateDetectionExitTime : det id = {detection_id}')   # for debugging
    
    def getCameraInfo(self):
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT camera_id, camera_ip, camera_port, stream_path, rtsp_id, rtsp_password FROM camera_info"
            )
            rows = cursor.fetchall()

        camera_list = []
        for row in rows:
            camera_list.append({
                "camera_id": row[0],
                "camera_ip": row[1],
                "camera_port": row[2],
                "stream_path": row[3],
                "rtsp_id": row[4],
                "rtsp_password": row[5],
            })
        return camera_list