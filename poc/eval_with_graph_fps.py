import subprocess, json
import matplotlib.pyplot as plt
from pathlib import Path

# 스윕할 conf 값들
CONF_LIST = [0.3]

# 실행 설정
JSON_DIR = "/home/hiperwall/Ai/poc/labels"
VIDEO_DIR = "/home/hiperwall/Ai/poc/videos"
OUT_ROOT_BASE = "./eval_time_out"   # 폴더만 의미상 구분, 실제 저장은 안함
TRACKER = "ocsort"
NPROC = 4

xs = []
infer_fps_means = []
track_fps_means = []
reid_fps_means  = []

for conf in CONF_LIST:
    out_root = Path(f"{OUT_ROOT_BASE}_{conf}")

    print(f"\n=== Running eval.py (time-only) with conf={conf} ===")
    # eval.py가 summary.json을 쓰지 않는다면, 표준출력만 보고 평균 낼 수 없다.
    # 그래서 여기서는 eval.py를 실행한 다음, 각 시퀀스의 summary.json을 읽는다.
    # 만약 eval.py에서 summary.json 쓰기를 주석 처리했다면, 그 줄을 살려라.
    subprocess.run([
        "python", "eval_fps.py",
        "--json-dir", JSON_DIR,
        "--video-dir", VIDEO_DIR,
        "--out-root", str(out_root),
        "--conf", str(conf),
        "--tracker", TRACKER,
        "--nproc", str(NPROC)
    ], check=False)

    # 요약 읽기: 각 시퀀스의 summary.json을 모아 평균 FPS
    summaries = list(out_root.glob("*/summary.json"))
    infer_fps, track_fps, reid_fps = [], [], []

    for sm in summaries:
        try:
            data = json.loads(sm.read_text(encoding="utf-8"))
            if data.get("error"):
                continue
            if "infer_fps" in data and "track_fps" in data and "reid_fps" in data:
                infer_fps.append(float(data["infer_fps"]))
                track_fps.append(float(data["track_fps"]))
                reid_fps.append(float(data["reid_fps"]))
        except Exception:
            continue

    def mean_or_zero(lst):
        return sum(lst) / max(len(lst), 1)

    xs.append(conf)
    infer_fps_means.append(mean_or_zero(infer_fps))
    track_fps_means.append(mean_or_zero(track_fps))
    reid_fps_means.append(mean_or_zero(reid_fps))

    print(f"[RESULT] conf={conf:.2f} -> "
          f"infer {infer_fps_means[-1]:.2f} fps, "
          f"track {track_fps_means[-1]:.2f} fps, "
          f"reid {reid_fps_means[-1]:.2f} fps")

# 그래프 출력
plt.figure()
plt.plot(xs, infer_fps_means, marker="o", label="infer FPS (mean)")
plt.plot(xs, track_fps_means, marker="s", label="track FPS (mean)")
plt.plot(xs, reid_fps_means,  marker="^", label="reid FPS (mean)")
plt.xlabel("Detection confidence threshold (--conf)")
plt.ylabel("FPS (higher is better)")
plt.title("infer / track / reid FPS vs confidence threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 저장하고 싶으면 주석 해제
# plt.savefig("fps_vs_conf.png")
