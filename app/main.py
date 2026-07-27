from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

import os
import shutil

from predictor import predict

app = FastAPI()

templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {}
    )


@app.post("/predict")
async def inference(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
):

    files = {
        "flair": flair,
        "t1": t1,
        "t1ce": t1ce,
        "t2": t2,
    }

    paths = {}

    for key, file in files.items():

        path = os.path.join("uploads", file.filename)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        paths[key] = path

    result = predict(
        paths["flair"],
        paths["t1"],
        paths["t1ce"],
        paths["t2"],
    )

    return JSONResponse({
        "preview_base64": result["preview_base64"],
        "overlay_base64": result["overlay_base64"],
        "label_stats": result["label_stats"],
        "viewer_html": result["viewer_html"],
        "download_url": "/download",
    })


@app.get("/download")
async def download():
    return FileResponse(
        "outputs/prediction.nii.gz",
        filename="prediction.nii.gz",
        media_type="application/octet-stream",
    )