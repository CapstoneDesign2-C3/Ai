import psycopg2

class PostgreSQL:
    def __init__(self, host, dbname, user, password, port):
        self.db = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port
        )
        print(f'DB connnect success')

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
                "INSERT INTO detected_object (uuid, crop_img, feature, code_id) VALUES (%s, %s, %s, %s)",
                (uuid, psycopg2.Binary(crop_img), feature, code_id)
            )
            self.db.commit()

    def addNewDetection(self, uuid, appeared_time, exit_time):
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM detected_object WHERE uuid = %s",
                (uuid,)
            )
            result = cursor.fetchone()
            if result is None:
                raise ValueError(f"uuid({uuid})가 detected_object에 없습니다.")
            detected_object_id = result[0]

            cursor.execute(
                """
                INSERT INTO detection (detected_object_id, appeared_time, exit_time)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (detected_object_id, appeared_time, exit_time)
            )
            det_id = cursor.fetchone()[0]
            self.db.commit()
            return det_id
        
    # 퇴장 시간 업데이트를 위한 메서드
    def updateDetectionExitTime(self, detection_id, exit_time):
        with self.db.cursor() as cursor:
            cursor.execute(
                "UPDATE detection SET exit_time = %s WHERE id = %s",
                (exit_time, detection_id)
            )
            self.db.commit()

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