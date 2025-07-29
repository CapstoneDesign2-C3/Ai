from confluent_kafka.admin import AdminClient

# 1) Kafka 브로커 주소 설정
admin = AdminClient({'bootstrap.servers': 'localhost:9092'})

# 2) 토픽 메타데이터 요청
md = admin.list_topics(timeout=10)    # 10초까지 대기

# 3) 토픽 이름 출력
print("Topics:")
for t in md.topics.values():
    print(f" - {t.topic}")
