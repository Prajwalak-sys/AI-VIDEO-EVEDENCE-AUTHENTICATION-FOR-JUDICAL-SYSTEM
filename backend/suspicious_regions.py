import os
import cv2
import numpy as np


def get_video_fps(video_path):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)

    video.release()

    if fps <= 0:
        raise ValueError(
            "Could not determine video FPS."
        )

    return fps


def group_consecutive_frames(
    frame_indices,
    max_gap=2
):

    if len(frame_indices) == 0:
        return []

    frame_indices = sorted(
        frame_indices
    )

    groups = []

    current_group = [
        frame_indices[0]
    ]

    for frame in frame_indices[1:]:

        previous_frame = current_group[-1]

        if frame - previous_frame <= max_gap:

            current_group.append(frame)

        else:

            groups.append(current_group)

            current_group = [frame]

    groups.append(current_group)

    return groups


def analyze_suspicious_regions(
    video_path,
    analysis_folder
):

    suspicious_file = os.path.join(
        analysis_folder,
        "suspicious_frames.npy"
    )

    if not os.path.exists(
        suspicious_file
    ):

        raise FileNotFoundError(
            f"Missing file: {suspicious_file}"
        )

    suspicious_frames = np.load(
        suspicious_file
    )

    print(
        f"Suspicious frames loaded: "
        f"{len(suspicious_frames)}"
    )

    if len(suspicious_frames) == 0:

        print(
            "No suspicious frames detected."
        )

        regions = []

        output_file = os.path.join(
            analysis_folder,
            "suspicious_regions.npy"
        )

        np.save(
            output_file,
            np.array(
                regions,
                dtype=object
            )
        )

        return regions

    fps = get_video_fps(
        video_path
    )

    groups = group_consecutive_frames(
        suspicious_frames
    )

    regions = []

    print()
    print(
        "Suspicious regions:"
    )

    for number, group in enumerate(
        groups,
        start=1
    ):

        start_frame = group[0]

        end_frame = group[-1]

        start_time = start_frame / fps

        end_time = end_frame / fps

        duration = end_time - start_time

        region = {

            "region": number,

            "start_frame":
                int(start_frame),

            "end_frame":
                int(end_frame),

            "start_time_seconds":
                float(start_time),

            "end_time_seconds":
                float(end_time),

            "duration_seconds":
                float(duration),

            "frame_count":
                len(group)
        }

        regions.append(region)

        print(
            f"Region {number}: "
            f"{start_time:.2f}s → "
            f"{end_time:.2f}s "
            f"({len(group)} frames)"
        )

    output_file = os.path.join(
        analysis_folder,
        "suspicious_regions.npy"
    )

    np.save(
        output_file,
        np.array(
            regions,
            dtype=object
        )
    )

    print()
    print(
        f"Total suspicious regions: "
        f"{len(regions)}"
    )

    print(
        f"Regions saved to: "
        f"{output_file}"
    )

    return regions