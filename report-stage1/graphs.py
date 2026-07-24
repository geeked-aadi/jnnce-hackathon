import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Load CSV
# ==========================================================

CSV_PATH = r"stage_1 reboot\brats2020_stage1_image_properties.csv"

df = pd.read_csv(CSV_PATH)

# ==========================================================
# Output folder
# ==========================================================

OUT_DIR = r"report-stage1\analytics_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# Parse Shape
# ==========================================================

def parse_shape(s):
    try:
        return ast.literal_eval(s)
    except:
        return None

df["shape_tuple"] = df["shape"].apply(parse_shape)

df["Width"] = df["shape_tuple"].apply(lambda x: x[0] if x else None)
df["Height"] = df["shape_tuple"].apply(lambda x: x[1] if x else None)
df["Slices"] = df["shape_tuple"].apply(lambda x: x[2] if x else None)

# ==========================================================
# 1 Resolution Table
# ==========================================================

resolution_table = df[
    ["patient_id","modality","Width","Height","Slices"]
]

resolution_table.to_csv(
    os.path.join(OUT_DIR,"resolution_table.csv"),
    index=False
)

print("Resolution table saved.")

# ==========================================================
# 2 MRI Modality Distribution
# ==========================================================

plt.figure(figsize=(8,5))

df["modality"].value_counts().sort_index().plot(kind="bar")

plt.title("MRI Modality Distribution")
plt.xlabel("Modality")
plt.ylabel("Number of MRI Scans")

plt.tight_layout()

plt.savefig(os.path.join(
    OUT_DIR,
    "modality_distribution.png"
))
plt.close()

# ==========================================================
# 3 Resolution Distribution
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(df["Width"], bins=15)

plt.title("Resolution Width Distribution")
plt.xlabel("Width")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(os.path.join(
    OUT_DIR,
    "resolution_distribution.png"
))
plt.close()

# ==========================================================
# 4 Slice Distribution
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(df["Slices"], bins=20)

plt.title("Slice Distribution")
plt.xlabel("Number of Slices")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(os.path.join(
    OUT_DIR,
    "slice_distribution.png"
))
plt.close()

# ==========================================================
# 5 Image Property Analysis
# ==========================================================

properties = {
    "Mean":"Mean Intensity",
    "Deviation":"Std Deviation",
    "Contrast":"Contrast",
    "Sharpness_LaplacianVar":"Sharpness",
    "Edge_Strength_Sobel":"Edge Strength",
    "Noise_Level_Sigma":"Noise",
    "Complexity_Entropy":"Entropy"
}

group = df.groupby("modality").mean(numeric_only=True)

for col,title in properties.items():

    plt.figure(figsize=(8,5))

    group[col].plot(kind="bar")

    plt.title(title+" by MRI Modality")
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
# 6 Mean Intensity Boxplot
# ==========================================================

plt.figure(figsize=(8,5))

data = [
    df[df.modality==m]["Mean"]
    for m in sorted(df.modality.unique())
]

plt.boxplot(
    data,
    tick_labels=sorted(df.modality.unique())
)

plt.title("Mean Intensity Distribution")
plt.xlabel("MRI Modality")
plt.ylabel("Mean Intensity")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "mean_intensity_boxplot.png"
    )
)

plt.close()

# ==========================================================
# 7 Summary Statistics Table
# ==========================================================

summary = group[
    [
        "Mean",
        "Deviation",
        "Contrast",
        "Sharpness_LaplacianVar",
        "Edge_Strength_Sobel",
        "Noise_Level_Sigma",
        "Complexity_Entropy"
    ]
]

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "image_property_summary.csv"
    )
)

# ==========================================================
# 8 Resolution Statistics
# ==========================================================

resolution_stats = df[
    ["Width","Height","Slices"]
].describe()

resolution_stats.to_csv(
    os.path.join(
        OUT_DIR,
        "resolution_statistics.csv"
    )
)

print("\nAll analytics completed.")
print("Graphs saved inside:", OUT_DIR)