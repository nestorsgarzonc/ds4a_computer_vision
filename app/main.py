from motion_heatmap import heatmap_video
from fastapi import FastAPI, File, UploadFile
import shutil

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "DS4A"}


@app.post("/uploadHeatmapVideo/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    heatmap_video(file.filename, 'diff-overlay.jpg', frames_sec=2, thresh=4, maxValue=2)
    return {"filename": file.filename}
