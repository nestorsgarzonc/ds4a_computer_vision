import shutil
from fastapi import FastAPI, File, UploadFile
from motion_heatmap import heatmap_video

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadHeatmapVideo/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    heatmap_video(file.filename, 'diff-overlay.jpg', frames_sec=2, thresh=4, maxValue=2)
    return {"filename": file.filename}
