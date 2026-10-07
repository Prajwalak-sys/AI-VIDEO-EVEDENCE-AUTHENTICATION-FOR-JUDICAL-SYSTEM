from fastapi import FastAPI, UploadFile, File
import os
import shutil
import numpy as np

from backend.video_processor import (
    extract_frames,
    get_video_info
)

from backend.spatial_analysis import (
    load_model,
    process_all_frames
)

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


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ZERO Video Evidence API",
    description="AI-powered video evidence authentication system",
    version="1.0.0"
)


# ============================================================
# FOLDERS
# ============================================================

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


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "ZERO Video Evidence API is running",
        "status": "online"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# UPLOAD VIDEO
# ============================================================

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...)
):

    # Save video
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

    # Video name without extension
    video_name = os.path.splitext(
        file.filename
    )[0]

    # Frame folder
    frames_folder = os.path.join(
        FRAME_FOLDER,
        video_name
    )

    # Extract frames
    total_frames = extract_frames(
        video_path,
        frames_folder
    )

    # Get video information
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


# ============================================================
# COMPLETE FORENSIC ANALYSIS
# ============================================================

@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...)
):

    # ========================================================
    # 1. SAVE VIDEO
    # ========================================================

    print()
    print("=" * 60)
    print("ZERO FORENSIC ANALYSIS STARTED")
    print("=" * 60)

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

    print()
    print(
        f"Video saved: {video_path}"
    )


    # ========================================================
    # 2. CREATE VIDEO-SPECIFIC FOLDERS
    # ========================================================

    video_name = os.path.splitext(
        file.filename
    )[0]

    frames_folder = os.path.join(
        FRAME_FOLDER,
        video_name
    )

    analysis_folder = os.path.join(
        ANALYSIS_FOLDER,
        video_name
    )

    os.makedirs(
        frames_folder,
        exist_ok=True
    )

    os.makedirs(
        analysis_folder,
        exist_ok=True
    )

    print(
        f"Frames folder: {frames_folder}"
    )

    print(
        f"Analysis folder: {analysis_folder}"
    )


    # ========================================================
    # 3. EXTRACT VIDEO FRAMES
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 1: FRAME EXTRACTION")
    print("=" * 60)

    total_frames = extract_frames(
        video_path,
        frames_folder
    )

    print(
        f"Total frames extracted: {total_frames}"
    )


    # ========================================================
    # 4. VIDEO INFORMATION
    # ========================================================

    fps, total, width, height, duration = (
        get_video_info(
            video_path
        )
    )

    print()
    print("Video Information")
    print(
        f"FPS: {fps}"
    )
    print(
        f"Total frames: {total}"
    )
    print(
        f"Resolution: {width} x {height}"
    )
    print(
        f"Duration: {duration:.2f} seconds"
    )


    # ========================================================
    # 5. SPATIAL ANALYSIS - MOBILENETV2
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 2: SPATIAL ANALYSIS - MOBILENETV2")
    print("=" * 60)

    print(
        "Loading MobileNetV2..."
    )

    model = load_model()

    print(
        "MobileNetV2 loaded successfully!"
    )

    print(
        "Extracting spatial features..."
    )

    spatial_features, spatial_frame_names = (
        process_all_frames(
            model,
            frames_folder
        )
    )

    spatial_file = os.path.join(
        analysis_folder,
        "spatial_features.npy"
    )

    spatial_names_file = os.path.join(
        analysis_folder,
        "spatial_frame_names.npy"
    )

    np.save(
        spatial_file,
        spatial_features
    )

    np.save(
        spatial_names_file,
        np.array(
            spatial_frame_names
        )
    )

    print()
    print(
        "Spatial feature extraction completed!"
    )

    print(
        f"Spatial feature shape: "
        f"{spatial_features.shape}"
    )

    print(
        f"Spatial features saved to: "
        f"{spatial_file}"
    )


    # ========================================================
    # 6. ELA ANALYSIS
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 3: ELA ANALYSIS")
    print("=" * 60)

    ela_scores, ela_names = process_ela(
        frames_folder,
        analysis_folder
    )

    print()
    print(
        f"ELA frames analyzed: "
        f"{len(ela_scores)}"
    )


    # ========================================================
    # 7. DCT ANALYSIS
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 4: DCT ANALYSIS")
    print("=" * 60)

    dct_scores, dct_names = process_dct(
        frames_folder,
        analysis_folder
    )

    print()
    print(
        f"DCT frames analyzed: "
        f"{len(dct_scores)}"
    )


    # ========================================================
    # 8. OPTICAL FLOW
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 5: OPTICAL FLOW ANALYSIS")
    print("=" * 60)

    flow_scores, flow_names = (
        process_optical_flow(
            frames_folder,
            analysis_folder
        )
    )

    print()
    print(
        f"Optical Flow transitions analyzed: "
        f"{len(flow_scores)}"
    )


    # ========================================================
    # 9. FEATURE FUSION
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 6: FEATURE FUSION")
    print("=" * 60)

    fused_features = create_fused_features(
        spatial_file,
        analysis_folder
    )

    print()
    print(
        f"Fused feature shape: "
        f"{fused_features.shape}"
    )


    # ========================================================
    # 10. ANOMALY DETECTION
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 7: ANOMALY DETECTION")
    print("=" * 60)

    anomaly_scores, suspicious_frames = (
        detect_anomalies(
            video_path,
            analysis_folder
        )
    )

    print()
    print(
        f"Suspicious frames: "
        f"{len(suspicious_frames)}"
    )


    # ========================================================
    # 11. SUSPICIOUS REGIONS
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 8: SUSPICIOUS REGION DETECTION")
    print("=" * 60)

    suspicious_regions = (
        analyze_suspicious_regions(
            video_path,
            analysis_folder
        )
    )

    print()
    print(
        f"Suspicious regions: "
        f"{len(suspicious_regions)}"
    )


    # ========================================================
    # 12. FINAL FORENSIC RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 9: FINAL FORENSIC RESULT")
    print("=" * 60)

    result = create_forensic_result(
        video_path,
        analysis_folder
    )


    # ========================================================
    # 13. FINAL RESPONSE
    # ========================================================

    print()
    print("=" * 60)
    print("ZERO FORENSIC ANALYSIS COMPLETED")
    print("=" * 60)

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