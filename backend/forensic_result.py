import os
import cv2
import numpy as np


def get_video_info(video_path):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    video.release()

    return {
        "fps": float(fps),
        "total_frames": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": float(duration)
    }


def load_suspicious_regions(
    analysis_folder
):

    regions_file = os.path.join(
        analysis_folder,
        "suspicious_regions.npy"
    )

    if not os.path.exists(
        regions_file
    ):

        return []

    data = np.load(
        regions_file,
        allow_pickle=True
    )

    return data.tolist()


def calculate_anomaly_summary(
    analysis_folder
):

    anomaly_file = os.path.join(
        analysis_folder,
        "anomaly_scores.npy"
    )

    if not os.path.exists(
        anomaly_file
    ):

        return {
            "average_score": 0.0,
            "maximum_score": 0.0
        }

    scores = np.load(
        anomaly_file
    )

    if len(scores) == 0:

        return {
            "average_score": 0.0,
            "maximum_score": 0.0
        }

    return {
        "average_score": float(
            np.mean(scores)
        ),
        "maximum_score": float(
            np.max(scores)
        )
    }


def create_forensic_result(
    video_path,
    analysis_folder
):

    video_info = get_video_info(
        video_path
    )

    suspicious_regions = (
        load_suspicious_regions(
            analysis_folder
        )
    )

    anomaly_summary = (
        calculate_anomaly_summary(
            analysis_folder
        )
    )

    if len(suspicious_regions) > 0:

        status = "potentially_suspicious"

    else:

        status = "no_significant_anomaly_detected"

    result = {

        "status": status,

        "video": {

            "fps":
                video_info["fps"],

            "total_frames":
                video_info[
                    "total_frames"
                ],

            "resolution":
                f"{video_info['width']} x "
                f"{video_info['height']}",

            "duration_seconds":
                round(
                    video_info[
                        "duration_seconds"
                    ],
                    2
                )
        },

        "analysis": {

            "ela": True,

            "dct": True,

            "optical_flow": True,

            "spatial_features": True,

            "temporal_analysis": True

        },

        "anomaly_summary": {

            "average_score":
                anomaly_summary[
                    "average_score"
                ],

            "maximum_score":
                anomaly_summary[
                    "maximum_score"
                ]

        },

        "suspicious_regions":
            suspicious_regions
    }

    return result