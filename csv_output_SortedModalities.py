import pandas as pd

df = pd.read_csv(r"D:\Hackathon\ProcesseDt\jnnce-hackathon\stage_1 reboot\brats2020_stage1_image_properties.csv")

t1 = df[df["modality"] == "T1"]
t2 = df[df["modality"] == "T2"]
t1ce = df[df["modality"] == "T1CE"]
flair = df[df["modality"] == "FLAIR"]

t1.to_csv("T1.csv", index=False)
t2.to_csv("T2.csv", index=False)
t1ce.to_csv("T1CE.csv", index=False)
flair.to_csv("FLAIR.csv", index=False)