from motion_heatmap import heatmap_video
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "DS4A"}


@app.post("/uploadHeatmapVideo/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result_path=f'{file.filename}.jpg'
    heatmap_video(file.filename, result_path, frames_sec=1, thresh=4, maxValue=2)
    
    os.remove(file.filename)
    return FileResponse(result_path)
