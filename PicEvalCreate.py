# -*- coding: utf-8 -*-
# กราฟเดียวรวมทุก metric โดยแต่ละหัวข้อมี 3 แท่ง (สามโมเดล)
# ไม่แสดงตัวเลขบนแท่ง และบันทึกเป็น nsfw_model_comparison_grouped.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.patches import Patch

csv_text = """Model,Best Validation Accuracy,Test Accuracy,Total Training Time (seconds),Test Loss,Test Precision (Weighted Avg),Test Recall (Weighted Avg),Test F1-Score (Weighted Avg)
ResNet18,0.87906446092413,0.878062678062678,4638.350125789642,0.4596177329365005,0.8771540949151673,0.878062678062678,0.8768169999823309
EfficientNetB0,0.9047347404449515,0.8888888888888888,25647.188742160797,0.45823431456530533,0.8883745182060779,0.8888888888888888,0.8883951473753504
MobileNetV3_Small,0.865373645179692,0.8541310541310542,3437.9968404769897,0.4983946426981195,0.8545937308908266,0.8541310541310542,0.8525061567119945
"""

# โหลดข้อมูล
df = pd.read_csv(StringIO(csv_text)).set_index("Model")

# ลำดับโมเดล + สีประจำโมเดล (คนละสีชัดเจน)
models = ["EfficientNetB0", "ResNet18", "MobileNetV3_Small"]
colors = {
    "EfficientNetB0": "#4e79a7",
    "ResNet18": "#f28e2b",
    "MobileNetV3_Small": "#59a14f",
}

# metric ฝั่งซ้าย (สเกล 0-1) + metric ฝั่งขวา (เวลา)
left_metrics = [
    "Best Validation Accuracy",
    "Test Accuracy",
    "Test Precision (Weighted Avg)",
    "Test Recall (Weighted Avg)",
    "Test F1-Score (Weighted Avg)",
    "Test Loss",
]
right_metric = "Total Training Time (seconds)"  # จะแปลงเป็นชั่วโมง

all_metrics = left_metrics + [right_metric]
x = np.arange(len(all_metrics))
width = 0.22

fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
ax2 = ax.twinx()

# วาดแท่งของ metrics ฝั่งซ้าย (0~1)
for i, m in enumerate(models):
    vals = []
    for metric in all_metrics:
        vals.append(float(df.loc[m, metric]) if metric in left_metrics else np.nan)
    vals = np.array(vals, dtype=float)
    ax.bar(x + (i - 1) * width, vals, width, color=colors[m], edgecolor="black", linewidth=0.7)

# วาดแท่งเวลาเทรน (ชั่วโมง) บนแกนขวา
right_x = x[-1]
for i, m in enumerate(models):
    hours = float(df.loc[m, right_metric]) / 3600.0
    ax2.bar(right_x + (i - 1) * width, hours, width, color=colors[m], edgecolor="black", linewidth=0.7)

# ป้ายชื่อแกน X
label_map = {
    "Best Validation Accuracy": "Best Val Acc",
    "Test Accuracy": "Test Acc",
    "Test Precision (Weighted Avg)": "Precision (w)",
    "Test Recall (Weighted Avg)": "Recall (w)",
    "Test F1-Score (Weighted Avg)": "F1 (w)",
    "Test Loss": "Loss",
    "Total Training Time (seconds)": "Train Time (h)",
}
plt.xticks(x, [label_map.get(m, m) for m in all_metrics], rotation=0, fontsize=10)

# ชื่อแกนและกริด
ax.set_ylabel("Score / Loss (0–1)", fontsize=11)
ax2.set_ylabel("Training Time (hours)", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# ขอบเขตแกนซ้าย
left_vals = [float(df.loc[m, metric]) for m in models for metric in left_metrics]
ax.set_ylim(max(0.0, min(left_vals) - 0.05), min(1.05, max(left_vals) + 0.05))

# ขอบเขตแกนขวา
right_vals = [float(df.loc[m, right_metric]) / 3600.0 for m in models]
ax2.set_ylim(0, max(right_vals) * 1.2)

# legend
legend_handles = [Patch(facecolor=colors[m], edgecolor="black", label=m) for m in models]
ax.legend(handles=legend_handles, title="Model", loc="upper left")

plt.title("NSFW Model Comparison – Grouped Metrics", fontsize=13)
plt.tight_layout()
plt.savefig("nsfw_model_comparison_grouped.png", bbox_inches="tight")
print("Saved to nsfw_model_comparison_grouped.png")
