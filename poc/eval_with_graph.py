# eval_with_graph.py
import subprocess, json
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# 실험할 conf 값들
CONF_LIST = [0.05, 0.1]

# eval.py 실행 설정
JSON_DIR = "/home/hiperwall/Ai/poc/labels_test"
VIDEO_DIR = "/home/hiperwall/Ai/poc/videos_test"
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
        "--nproc", str(NPROC),
    ], check=True)

    # eval.py 실행 후 metrics.json 읽기
    sum_files = glob.glob(str(out_root / "*" / "summary.json"))
    for sf in sum_files:
        with open(sf, "r") as f:
            summary = json.load(f)
        prec, rec = summary.get("precision", 0), summary.get("recall", 0)
        xs.append(conf)
        ys_prec.append(prec)
        ys_rec.append(rec)

# --- 결과 그래프 ---
plt.figure(figsize=(8, 6))
plt.plot(xs, ys_prec, marker="o", label="Precision")
plt.plot(xs, ys_rec, marker="s", label="Recall")
plt.xlabel("Confidence Threshold")
plt.ylabel("Score")
plt.title(f"Precision / Recall vs Confidence (Tracker={TRACKER}, IoU={IOU})")
plt.grid(True)
plt.legend()
plt.savefig("precision_recall_vs_conf.png")
plt.show()
