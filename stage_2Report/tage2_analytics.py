import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Load CSV
# ==========================================================

CSV_PATH = r"D:\Hackathon\ProcesseDt\jnnce-hackathon\stage-2\preprocessing_metrics.csv"

df = pd.read_csv(CSV_PATH)

# ==========================================================
# Output Folder
# ==========================================================

OUT_DIR = r"combinedReport_s1_s2"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# 1 Modality Distribution
# ==========================================================

plt.figure(figsize=(8,5))

df["Modality"].value_counts().sort_index().plot(kind="bar")

plt.title("MRI Modality Distribution")
plt.xlabel("Modality")
plt.ylabel("Number of MRI Volumes")

plt.tight_layout()

plt.savefig(os.path.join(
    OUT_DIR,
    "modality_distribution.png"
))
plt.close()

# ==========================================================
# 2 Property Analysis
# ==========================================================

properties = {
    "Mean":"Mean Intensity",
    "Std":"Standard Deviation",
    "Contrast":"Contrast",
    "Sharpness":"Sharpness",
    "EdgeStrength":"Edge Strength",
    "NoiseLevel":"Noise Level"
}

group = df.groupby("Modality").mean(numeric_only=True)

for col,title in properties.items():

    plt.figure(figsize=(8,5))

    group[col].plot(kind="bar")

    plt.title(title + " by MRI Modality")
    plt.ylabel(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"{col}_bar.png"
        )
    )

    plt.close()

# ==========================================================
# 3 Boxplots
# ==========================================================

for col,title in properties.items():

    plt.figure(figsize=(8,5))

    data = [
        df[df.Modality==m][col]
        for m in sorted(df.Modality.unique())
    ]

    plt.boxplot(
        data,
        tick_labels=sorted(df.Modality.unique())
    )

    plt.title(title + " Distribution")
    plt.xlabel("MRI Modality")
    plt.ylabel(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"{col}_boxplot.png"
        )
    )

    plt.close()

# ==========================================================
# 4 Summary Statistics
# ==========================================================

summary = group[list(properties.keys())]

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "preprocessing_summary.csv"
    )
)

print("\nStage 2 analytics completed.")
print("Saved to:", OUT_DIR)