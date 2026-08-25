from fastapi import FastAPI, UploadFile, File
import os
import shutil

from backend.video_processor import (
    extract_frames,
    get_video_info
)

from backend.spatial_analysis import extract_features

from backend.ela_analysis import (
    process_all_frames as process_ela
)

from backend.dct_analysis import (
    process_all_frames as process_dct
)

from backend.optical_flow import (
    process_frames as process_optical_flow
)

from backend.feature_fusion import (
    create_fused_features
)

from backend.anomaly_detection import (
    detect_anomalies
)

from backend.suspicious_regions import (
    analyze_suspicious_regions
)

from backend.forensic_result import (
    create_forensic_result
)


app = FastAPI(
    title="ZERO Video Evidence API",
    description="AI-powered video evidence authentication system",
    version="1.0.0"
)


VIDEO_FOLDER = "videos"
FRAME_FOLDER = "frames"
ANALYSIS_FOLDER = "analysis"


os.makedirs(
    VIDEO_FOLDER,
    exist_ok=True
)

os.makedirs(
    FRAME_FOLDER,
    exist_ok=True
)

os.makedirs(
    ANALYSIS_FOLDER,
    exist_ok=True
)


@app.get("/")
def home():

    return {
        "message":
            "ZERO Video Evidence API is running",
        "status":
            "online"
    }


@app.get("/health")
def health_check():

    return {
        "status":
            "healthy"
    }


@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...)
):

    video_path = os.path.join(
        VIDEO_FOLDER,
        file.filename
    )

    with open(
        video_path,
        "wb"
    ) as video:

        shutil.copyfileobj(
            file.file,
            video
        )

    video_name = os.path.splitext(
        file.filename
    )[0]

    frames_folder = os.path.join(
        FRAME_FOLDER,
        video_name
    )

    total_frames = extract_frames(
        video_path,
        frames_folder
    )

    fps, total, width, height, duration = (
        get_video_info(
            video_path
        )
    )

    return {

        "filename":
            file.filename,

        "message":
            "Video uploaded and processed successfully",

        "total_frames":
            total_frames,

        "fps":
            fps,

        "resolution":
            f"{width} x {height}",

        "duration_seconds":
            round(
                duration,
                2
            ),

        "frames_folder":
            frames_folder
    }


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...)
):

    # ---------------------------------------------
    # 1. Save video
    # ---------------------------------------------

    video_path = os.path.join(
        VIDEO_FOLDER,
        file.filename
    )

    with open(
        video_path,
        "wb"
    ) as video:

        shutil.copyfileobj(
            file.file,
            video
        )

    video_name = os.path.splitext(
        file.filename
    )[0]

    # ---------------------------------------------
    # 2. Create video-specific folders
    # ---------------------------------------------

    frames_folder = os.path.join(
        FRAME_FOLDER,
        video_name
    )

    analysis_folder = os.path.join(
        ANALYSIS_FOLDER,
        video_name
    )

    os.makedirs(
        analysis_folder,
        exist_ok=True
    )

    # ---------------------------------------------
    # 3. Extract frames
    # ---------------------------------------------

    print()
    print(
        "Extracting frames..."
    )

    total_frames = extract_frames(
        video_path,
        frames_folder
    )

    # ---------------------------------------------
    # 4. Get video information
    # ---------------------------------------------

    fps, total, width, height, duration = (
        get_video_info(
            video_path
        )
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"FPS: {fps}"
    )

    # ---------------------------------------------
    # 5. MobileNetV2 spatial features
    # ---------------------------------------------

    print()
    print(
        "Extracting spatial features..."
    )

    spatial_file = os.path.join(
        analysis_folder,
        "spatial_features.npy"
    )

    spatial_features = extract_features(
        frames_folder
    )

    import numpy as np

    np.save(
        spatial_file,
        spatial_features
    )

    print(
        f"Spatial features saved to: "
        f"{spatial_file}"
    )

    # ---------------------------------------------
    # 6. ELA
    # ---------------------------------------------

    print()
    print(
        "Running ELA analysis..."
    )

    ela_scores, ela_names = process_ela(
        frames_folder,
        analysis_folder
    )

    # ---------------------------------------------
    # 7. DCT
    # ---------------------------------------------

    print()
    print(
        "Running DCT analysis..."
    )

    dct_scores, dct_names = process_dct(
        frames_folder,
        analysis_folder
    )

    # ---------------------------------------------
    # 8. Optical Flow
    # ---------------------------------------------

    print()
    print(
        "Running Optical Flow analysis..."
    )

    flow_scores, flow_names = (
        process_optical_flow(
            frames_folder,
            analysis_folder
        )
    )

    # ---------------------------------------------
    # 9. Feature Fusion
    # ---------------------------------------------

    print()
    print(
        "Running feature fusion..."
    )

    fused_features = create_fused_features(
        spatial_file,
        analysis_folder
    )

    # ---------------------------------------------
    # 10. Anomaly Detection
    # ---------------------------------------------

    print()
    print(
        "Running anomaly detection..."
    )

    anomaly_scores, suspicious_frames = (
        detect_anomalies(
            video_path,
            analysis_folder
        )
    )

    # ---------------------------------------------
    # 11. Suspicious Regions
    # ---------------------------------------------

    print()
    print(
        "Finding suspicious regions..."
    )

    suspicious_regions = (
        analyze_suspicious_regions(
            video_path,
            analysis_folder
        )
    )

    # ---------------------------------------------
    # 12. Final forensic result
    # ---------------------------------------------

    print()
    print(
        "Creating forensic result..."
    )

    result = create_forensic_result(
        video_path,
        analysis_folder
    )

    # ---------------------------------------------
    # 13. Return result
    # ---------------------------------------------

    return {

        "message":
            "ZERO forensic analysis completed",

        "filename":
            file.filename,

        "video": {

            "fps":
                fps,

            "total_frames":
                total_frames,

            "resolution":
                f"{width} x {height}",

            "duration_seconds":
                round(
                    duration,
                    2
                )
        },

        "feature_shapes": {

            "spatial":
                list(
                    spatial_features.shape
                ),

            "fused":
                list(
                    fused_features.shape
                )
        },

        "analysis": {

            "ela_frames":
                len(ela_scores),

            "dct_frames":
                len(dct_scores),

            "optical_flow_transitions":
                len(flow_scores),

            "suspicious_frames":
                len(suspicious_frames),

            "suspicious_regions":
                len(suspicious_regions)
        },

        "forensic_result":
            result
    }