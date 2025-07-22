import psycopg2

class PostgreSQL:
    def __init__(self, host, dbname, user, password, port):
        self.db = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port
        )

    def addNewDetectedObject(self, uuid, crop_img_url, feature, code_name):
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
                "INSERT INTO detected_object (uuid, crop_img_url, feature, code_id) VALUES (%s, %s, %s, %s)",
                (uuid, crop_img_url, feature, code_id)
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
                "INSERT INTO detection (detected_object_id, appeared_time, exit_time) VALUES (%s, %s, %s)",
                (detected_object_id, appeared_time, exit_time)
            )
            self.db.commit()