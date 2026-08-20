from fastapi import FastAPI, UploadFile, File
import os

from backend.video_processor import extract_frames, get_video_info


app = FastAPI()


VIDEO_FOLDER = "videos"
FRAME_FOLDER = "frames"


os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "ZERO Video Evidence API is running"
    }


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):

    video_path = os.path.join(VIDEO_FOLDER, file.filename)

    with open(video_path, "wb") as video:
        video.write(await file.read())

    video_name = os.path.splitext(file.filename)[0]

    output_folder = os.path.join(FRAME_FOLDER, video_name)

    total_frames = extract_frames(
        video_path,
        output_folder
    )

    fps, total, width, height, duration = get_video_info(
        video_path
    )

    return {
        "filename": file.filename,
        "message": "Video uploaded and processed successfully",
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{width} x {height}",
        "duration_seconds": round(duration, 2),
        "frames_folder": output_folder
    }