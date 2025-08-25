import psycopg2
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


class PostgreSQL:
    def __init__(self, host, dbname, user, password, port, *, autocommit=False):
        self.db = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port
        )
        self.db.autocommit = autocommit
        print("DB connnect success")

    def close(self):
        try:
            self.db.close()
        except Exception:
            # 연결이 이미 닫혔거나 비정상일 수 있으므로 조용히 무시
            pass

    def addNewDetectedObject(self, uuid, crop_img, feature="", code_name="사람"):
        """
        detected_object 테이블에 신규 객체 레코드를 추가.
        예외 발생 시 롤백.
        """
        try:
            with self.db.cursor() as cursor:
                cursor.execute(
                    "SELECT code_id FROM event_codes WHERE code_name = %s",
                    (code_name,),
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
                    (uuid, psycopg2.Binary(crop_img), feature, code_id),
                )

            self.db.commit()
            print(f"uuid : {uuid} is insert")  # for debugging

        except Exception as e:
            self.db.rollback()
            raise e

    def addNewDetection(self, crop_img, uuid, appeared_time, exit_time=None):
        """
        detection 세션을 생성하거나, 열린 세션이 있으면 appeared_time을 보정.
        예외 발생 시 롤백.
        appeared_time: epoch ms(int)
        """
        try:
            with self.db.cursor() as cursor:
                # 명시적 트랜잭션 시작
                cursor.execute("BEGIN;")

                # 대상 객체 잠금 공유
                cursor.execute(
                    "SELECT id FROM detected_object WHERE uuid = %s FOR SHARE",
                    (uuid,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"uuid({uuid})가 detected_object에 없습니다.")
                detected_object_id = row[0]

                # KST 로 변환 후 DB에는 naive(datetime)로 저장 (timestamp without time zone 가정)
                appeared = (
                    datetime.fromtimestamp(appeared_time / 1000.0, tz=timezone.utc)
                    .astimezone(KST)
                    .replace(tzinfo=None)
                )

                # 같은 객체의 열린 세션이 있으면 재사용 (직렬화)
                cursor.execute(
                    """
                    SELECT id FROM detection
                    WHERE detected_object_id = %s AND exit_time IS NULL
                    FOR UPDATE
                    """,
                    (detected_object_id,),
                )
                open_row = cursor.fetchone()

                if open_row:
                    det_id = open_row[0]
                    cursor.execute(
                        """
                        UPDATE detection
                        SET appeared_time = LEAST(appeared_time, %s)
                        WHERE id = %s
                        """,
                        (appeared, det_id),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO detection (detected_object_id, appeared_time, exit_time, crop_img)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            detected_object_id,
                            appeared,
                            exit_time,
                            psycopg2.Binary(crop_img),
                        ),
                    )
                    det_id = cursor.fetchone()[0]

                cursor.execute("COMMIT;")
                return det_id

        except Exception as e:
            # 명시적 트랜잭션을 사용했으므로 연결 기준 롤백
            self.db.rollback()
            raise e

    def updateDetectionExitTime(self, detection_id, exit_time, camera_id):
        """
        퇴장 시간과 영상 ID를 업데이트.
        예외 발생 시 롤백.
        exit_time: epoch ms(int/float) 또는 datetime
        """
        try:
            if isinstance(exit_time, (int, float)):
                exit_dt = (
                    datetime.fromtimestamp(exit_time / 1000.0, tz=timezone.utc)
                    .astimezone(KST)
                    .replace(tzinfo=None)
                )
            elif isinstance(exit_time, datetime):
                exit_dt = exit_time
            else:
                raise TypeError("exit_time must be epoch ms or datetime")

            with self.db.cursor() as cursor:
                cursor.execute("BEGIN;")

                # 영상 레코드가 반드시 있어야 한다면 새로 생성
                cursor.execute(
                    """
                    INSERT INTO video (camera_id)
                    VALUES (%s)
                    RETURNING id
                    """,
                    (camera_id,),
                )
                video_id = cursor.fetchone()[0]

                cursor.execute(
                    """
                    UPDATE detection
                    SET exit_time = GREATEST(COALESCE(exit_time, %s), %s),
                        video_id  = %s
                    WHERE id = %s
                    """,
                    (exit_dt, exit_dt, video_id, detection_id),
                )

                if cursor.rowcount != 1:
                    raise ValueError(
                        f"Detection not found or not updated (id={detection_id})"
                    )

                cursor.execute("COMMIT;")

            self.db.commit()
            print(f"updateDetectionExitTime : det id = {detection_id}")

        except Exception as e:
            self.db.rollback()
            raise e

    def getCameraInfo(self):
        """
        카메라 정보 조회. 읽기만 수행하므로 별도 롤백 불필요.
        """
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT camera_id, camera_ip, camera_port, stream_path, rtsp_id, rtsp_password FROM camera_info"
            )
            rows = cursor.fetchall()

        camera_list = []
        for row in rows:
            camera_list.append(
                {
                    "camera_id": row[0],
                    "camera_ip": row[1],
                    "camera_port": row[2],
                    "stream_path": row[3],
                    "rtsp_id": row[4],
                    "rtsp_password": row[5],
                }
            )
        return camera_list
