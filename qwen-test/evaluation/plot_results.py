
import json
import os
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

files = glob.glob(os.path.join(RESULTS_DIR, "run_*.json"))

if not files:
    print("No evaluation result files found.")
    raise SystemExit(1)

latest_file = max(files, key=os.path.getmtime)

with open(latest_file, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data["summary"]["normalized"]

metrics = ["precision", "recall", "f1"]
labels = ["Precision", "Recall", "F1"]

node_scores = [
    summary["node"]["macro"][m] for m in metrics
]
edge_scores = [
    summary["edge"]["macro"][m] for m in metrics
]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(
    [i - width / 2 for i in x],
    node_scores,
    width,
    label="Nodes"
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    edge_scores,
    width,
    label="Edges"
)

ax.set_title("Text2Arch Evaluation Results")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metrics")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.bar_label(bars1, fmt="%.3f", padding=3)
ax.bar_label(bars2, fmt="%.3f", padding=3)

plt.tight_layout()

output_path = os.path.join(
    RESULTS_DIR, "latest_score_bar_graph.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print("Graph generated successfully!")
print(f"Source results: {latest_file}")
print(f"Graph saved at: {output_path}")
