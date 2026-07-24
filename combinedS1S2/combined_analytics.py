import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Load CSVs
# ==========================================================

stage2 = pd.read_csv(r"D:\Hackathon\ProcesseDt\jnnce-hackathon\stage-2\preprocessing_metrics.csv")
stage1 = pd.read_csv(r"D:\Hackathon\ProcesseDt\jnnce-hackathon\stage_1 reboot\brats2020_stage1_image_properties.csv")

# ==========================================================
# Output Folder
# ==========================================================

OUT_DIR = r"report-combined\analytics_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# Rename Stage 1 columns
# ==========================================================

stage1 = stage1.rename(columns={
    "Deviation":"Std",
    "Sharpness_LaplacianVar":"Sharpness",
    "Edge_Strength_Sobel":"EdgeStrength",
    "Noise_Level_Sigma":"NoiseLevel",
    "modality":"Modality"
})

metrics = [
    "Mean",
    "Std",
    "Contrast",
    "Sharpness",
    "EdgeStrength",
    "NoiseLevel"
]

stage1_avg = stage1.groupby("Modality").mean(numeric_only=True)
stage2_avg = stage2.groupby("Modality").mean(numeric_only=True)

# ==========================================================
# Comparison Graphs
# ==========================================================

for metric in metrics:

    comparison = pd.DataFrame({
        "Stage 1": stage1_avg[metric],
        "Stage 2": stage2_avg[metric]
    })

    plt.figure(figsize=(8,5))

    comparison.plot(kind="bar")

    plt.title(metric + " : Stage 1 vs Stage 2")

    plt.ylabel(metric)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            metric + "_comparison.png"
        )
    )

    plt.close()

# ==========================================================
# Combined Summary
# ==========================================================

summary = pd.concat(
    [
        stage1_avg[metrics].add_prefix("Stage1_"),
        stage2_avg[metrics].add_prefix("Stage2_")
    ],
    axis=1
)

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "combined_summary.csv"
    )
)

print("\nCombined analytics completed.")
print("Saved to:", OUT_DIR)