import subprocess, json
import matplotlib.pyplot as plt
from pathlib import Path

# 실험할 conf 값들 (요청한 범위)
CONF_LIST = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# eval.py 실행 설정
JSON_DIR = "/home/hiperwall/Ai/poc/labels"
VIDEO_DIR = "/home/hiperwall/Ai/poc/videos"
OUT_ROOT_BASE = "./eval_out_sweep"
IOU = 0.3
TRACKER = "ocsort"
NPROC = 4

xs, ys_prec, ys_rec = [], [], []

for conf in CONF_LIST:
    out_root = Path(f"{OUT_ROOT_BASE}_{conf}")

    print(f"\n=== Running eval.py with conf={conf} ===")
    subprocess.run([
        "python", "eval.py",
        "--json-dir", JSON_DIR,
        "--video-dir", VIDEO_DIR,
        "--out-root", str(out_root),
        "--iou", str(IOU),
        "--conf", str(conf),
        "--tracker", TRACKER,
        "--nproc", str(NPROC)
    ], check=False)

    # eval.py가 남긴 summary.json 파일들 모아서 평균 계산
    summaries = list(out_root.glob("*/summary.json"))
    precisions, recalls = [], []
    for sm in summaries:
        try:
            data = json.loads(sm.read_text(encoding="utf-8"))
            if "precision" in data and "recall" in data and data.get("error") in (None, "", "null"):
                precisions.append(float(data["precision"]))
                recalls.append(float(data["recall"]))
        except Exception:
            continue

    mean_p = sum(precisions) / max(len(precisions), 1)
    mean_r = sum(recalls) / max(len(recalls), 1)

    xs.append(conf)
    ys_prec.append(mean_p)
    ys_rec.append(mean_r)

    print(f"[RESULT] conf={conf:.2f} -> mean Precision={mean_p:.4f}, mean Recall={mean_r:.4f}")

# 그래프 출력
plt.figure()
plt.plot(xs, ys_prec, marker="o", label="Precision (mean)")
plt.plot(xs, ys_rec, marker="s", label="Recall (mean)")
plt.xlabel("Detection confidence threshold (--conf)")
plt.ylabel("Mean over sequences")
plt.title("Precision/Recall vs confidence threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 파일 저장은 네가 원하면 풀어라. 기본은 주석.
plt.savefig("pr_vs_conf.png")
