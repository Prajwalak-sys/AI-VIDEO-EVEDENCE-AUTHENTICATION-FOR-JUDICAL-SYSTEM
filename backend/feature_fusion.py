import os
import numpy as np


def normalize_scores(scores):

    mean = np.mean(scores)
    std = np.std(scores)

    if std == 0:
        return np.zeros_like(
            scores,
            dtype=np.float32
        )

    return (
        (scores - mean) / std
    ).astype(np.float32)


def create_fused_features(
    spatial_file,
    analysis_folder
):

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
        spatial_file,
        ela_file,
        dct_file,
        flow_file
    ]

    for file in required_files:

        if not os.path.exists(file):

            raise FileNotFoundError(
                f"Missing file: {file}"
            )

    spatial = np.load(
        spatial_file
    )

    ela = np.load(
        ela_file
    )

    dct = np.load(
        dct_file
    )

    optical_flow = np.load(
        flow_file
    )

    print(
        f"Spatial features: {spatial.shape}"
    )

    print(
        f"ELA scores: {ela.shape}"
    )

    print(
        f"DCT scores: {dct.shape}"
    )

    print(
        f"Optical Flow scores: "
        f"{optical_flow.shape}"
    )

    num_frames = min(
        len(spatial),
        len(ela),
        len(dct)
    )

    if num_frames == 0:

        raise ValueError(
            "No common frames found."
        )

    spatial = spatial[
        :num_frames
    ]

    ela = ela[
        :num_frames
    ]

    dct = dct[
        :num_frames
    ]

    # Optical Flow has one fewer value
    # because it represents frame transitions.
    if len(optical_flow) >= num_frames:

        flow = optical_flow[
            :num_frames
        ]

    elif len(optical_flow) > 0:

        flow = np.pad(
            optical_flow,
            (
                0,
                num_frames -
                len(optical_flow)
            ),
            mode="edge"
        )

    else:

        flow = np.zeros(
            num_frames,
            dtype=np.float32
        )

    ela = normalize_scores(
        ela
    )

    dct = normalize_scores(
        dct
    )

    flow = normalize_scores(
        flow
    )

    ela = ela.reshape(
        num_frames,
        1
    )

    dct = dct.reshape(
        num_frames,
        1
    )

    flow = flow.reshape(
        num_frames,
        1
    )

    fused_features = np.concatenate(
        [
            spatial,
            ela,
            dct,
            flow
        ],
        axis=1
    )

    output_file = os.path.join(
        analysis_folder,
        "fused_features.npy"
    )

    np.save(
        output_file,
        fused_features
    )

    print()
    print(
        "Feature fusion completed!"
    )

    print(
        f"Fused feature shape: "
        f"{fused_features.shape}"
    )

    print(
        f"Saved to: {output_file}"
    )

    return fused_features