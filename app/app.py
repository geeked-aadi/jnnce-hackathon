from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from pathlib import Path
import os
import tempfile
import numpy as np
import nibabel as nib
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    EnsureTyped,
)

from monai.inferers import sliding_window_inference

# ============================================================
# FastAPI
# ============================================================

app = FastAPI()

templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Load Model
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.ts"

print("Model Path:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nmodel.ts not found!\n\nExpected:\n{MODEL_PATH}\n"
    )

model = torch.jit.load(str(MODEL_PATH), map_location=device)
model.eval()

print("Model Loaded Successfully!")

# ============================================================
# Home
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ============================================================
# Predict
# ============================================================

@app.post("/predict")
async def predict(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
):

    tmp_dir = tempfile.mkdtemp()

    saved_files = {}

    for file in [flair, t1, t1ce, t2]:

        path = os.path.join(tmp_dir, file.filename)

        with open(path, "wb") as f:
            f.write(await file.read())

        name = file.filename.lower()

        print("Uploaded filename:", file.filename)

        if "flair" in name:
            saved_files["flair"] = path
        elif "t1ce" in name:
            saved_files["t1ce"] = path
        elif "t1" in name:
            saved_files["t1"] = path
        elif "t2" in name:
            saved_files["t2"] = path

    print("Saved files mapping:", saved_files)

    if len(saved_files) != 4:
        return {
            "error": "Upload all four MRI files (FLAIR, T1, T1CE, T2)."
        }

    sample = {
        "image": [
            saved_files["t1ce"],
            saved_files["t1"],
            saved_files["t2"],
            saved_files["flair"],
        ]
    }

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
        EnsureTyped(keys=["image"]),
    ])

    data = transforms(sample)

    image = data["image"]

    with torch.no_grad():

        x = image.unsqueeze(0).to(device)

        pred = sliding_window_inference(
            inputs=x,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    pred = torch.sigmoid(pred)

    tc = (pred[0, 0] > 0.5).cpu().numpy()

    tumor_voxels = int(tc.sum())

    return {
        "Prediction": "Completed",
        "Tumor Voxels": tumor_voxels
    }