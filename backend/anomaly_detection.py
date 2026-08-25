import os
import cv2
import numpy as np


def get_video_fps(video_path):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    video.release()

    if fps <= 0:
        raise ValueError(
            "Could not determine video FPS."
        )

    return fps


def load_scores(analysis_folder):

    ela_file = os.path.join(
        analysis_folder,
        "ela_scores.npy"
    )

    dct_file = os.path.join(
        analysis_folder,
        "dct_scores.npy"
    )

    flow_file = os.path.join(
        analysis_folder,
        "optical_flow_scores.npy"
    )

    required_files = [
        ela_file,
        dct_file,
        flow_file
    ]

    for file in required_files:

        if not os.path.exists(file):

            raise FileNotFoundError(
                f"Missing file: {file}"
            )

    ela = np.load(
        ela_file
    )

    dct = np.load(
        dct_file
    )

    flow = np.load(
        flow_file
    )

    return ela, dct, flow


def calculate_z_scores(values):

    mean = np.mean(values)

    std = np.std(values)

    if std == 0:

        return np.zeros_like(
            values,
            dtype=np.float32
        )

    return (
        (values - mean) / std
    )


def detect_anomalies(
    video_path,
    analysis_folder
):

    ela, dct, flow = load_scores(
        analysis_folder
    )

    fps = get_video_fps(
        video_path
    )

    print(
        f"Video FPS: {fps:.4f}"
    )

    num_frames = min(
        len(ela),
        len(dct)
    )

    if num_frames == 0:

        raise ValueError(
            "No forensic scores available."
        )

    ela = ela[
        :num_frames
    ]

    dct = dct[
        :num_frames
    ]

    # Optical Flow describes transitions,
    # so it normally has one fewer value.
    if len(flow) >= num_frames:

        flow_aligned = flow[
            :num_frames
        ]

    elif len(flow) > 0:

        flow_aligned = np.pad(
            flow,
            (
                0,
                num_frames - len(flow)
            ),
            mode="edge"
        )

    else:

        flow_aligned = np.zeros(
            num_frames,
            dtype=np.float32
        )

    ela_z = calculate_z_scores(
        ela
    )

    dct_z = calculate_z_scores(
        dct
    )

    flow_z = calculate_z_scores(
        flow_aligned
    )

    combined_score = (
        0.4 * ela_z
        + 0.3 * dct_z
        + 0.3 * flow_z
    )

    threshold = 2.0

    suspicious_indices = np.where(
        combined_score >= threshold
    )[0]

    print()
    print(
        "Anomaly detection completed!"
    )

    print(
        f"Total frames analyzed: "
        f"{num_frames}"
    )

    print(
        f"Suspicious frames found: "
        f"{len(suspicious_indices)}"
    )

    print()
    print(
        "Suspicious frames:"
    )

    for index in suspicious_indices:

        timestamp = index / fps

        print(
            f"Frame {index:05d} | "
            f"Time: {timestamp:.2f}s | "
            f"Score: "
            f"{combined_score[index]:.3f}"
        )

    anomaly_file = os.path.join(
        analysis_folder,
        "anomaly_scores.npy"
    )

    suspicious_file = os.path.join(
        analysis_folder,
        "suspicious_frames.npy"
    )

    np.save(
        anomaly_file,
        combined_score
    )

    np.save(
        suspicious_file,
        suspicious_indices
    )

    print()
    print(
        f"Anomaly scores saved to: "
        f"{anomaly_file}"
    )

    print(
        f"Suspicious frames saved to: "
        f"{suspicious_file}"
    )

    return (
        combined_score,
        suspicious_indices
    )