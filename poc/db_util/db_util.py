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


    def addNewDetection(self, video_id, crop_img, uuid, appeared_time, exit_time=None):
        with self.db.cursor() as cursor:
            cursor.execute("BEGIN;")
            cursor.execute("SELECT id FROM detected_object WHERE uuid = %s FOR SHARE", (uuid,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"uuid({uuid})가 detected_object에 없습니다.")
            detected_object_id = row[0]

            utc_appeared = datetime.utcfromtimestamp(appeared_time/1000.0)

            # 같은 객체의 '열린' 세션이 있으면 잡아서 재사용 (동시에 들어와도 FOR UPDATE로 직렬화)
            cursor.execute("""
                SELECT id FROM detection
                WHERE detected_object_id = %s AND exit_time IS NULL
                FOR UPDATE
            """, (detected_object_id,))
            open_row = cursor.fetchone()
            if open_row:
                det_id = open_row[0]
                cursor.execute("""
                    UPDATE detection
                    SET appeared_time = LEAST(appeared_time, %s)
                    WHERE id = %s
                """, (utc_appeared, det_id))
            else:
                cursor.execute("""
                    INSERT INTO detection (detected_object_id, appeared_time, exit_time, video_id, crop_img)
                    VALUES (%s, %s, %s, %d, %s)
                    RETURNING id
                """, (detected_object_id, utc_appeared, exit_time, video_id, psycopg2.Binary(crop_img)))
                det_id = cursor.fetchone()[0]

            cursor.execute("COMMIT;")
            return det_id

        
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
                """
                UPDATE detection
                SET exit_time = GREATEST(COALESCE(exit_time, %s), %s)
                WHERE id = %s
                """,
                (exit_dt, exit_dt, detection_id)
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Detection not found or not updated (id={detection_id})")
        self.db.commit()
        print(f'updateDetectionExitTime : det id = {detection_id}')

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