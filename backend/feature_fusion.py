import numpy as np
import os


SPATIAL_FILE = "spatial_features.npy"
ELA_FILE = "ela_scores.npy"
DCT_FILE = "dct_scores.npy"
FLOW_FILE = "optical_flow_scores.npy"


def load_features():

    required_files = [
        SPATIAL_FILE,
        ELA_FILE,
        DCT_FILE,
        FLOW_FILE
    ]

    for file in required_files:

        if not os.path.exists(file):

            print(
                f"Missing file: {file}"
            )

            return None

    spatial = np.load(
        SPATIAL_FILE
    )

    ela = np.load(
        ELA_FILE
    )

    dct = np.load(
        DCT_FILE
    )

    optical_flow = np.load(
        FLOW_FILE
    )

    return (
        spatial,
        ela,
        dct,
        optical_flow
    )


def normalize_scores(scores):

    mean = np.mean(scores)
    std = np.std(scores)

    if std == 0:

        return np.zeros_like(
            scores,
            dtype=np.float32
        )

    normalized = (
        scores - mean
    ) / std

    return normalized.astype(
        np.float32
    )


def create_fused_features():

    data = load_features()

    if data is None:

        return

    spatial, ela, dct, optical_flow = data

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

    # Find the number of common frames
    num_frames = min(
        len(spatial),
        len(ela),
        len(dct)
    )

    if num_frames == 0:

        print(
            "No common frames found."
        )

        return

    # Keep only matching frames
    spatial = spatial[
        :num_frames
    ]

    ela = ela[
        :num_frames
    ]

    dct = dct[
        :num_frames
    ]

    # Optical flow has one fewer value
    # because it describes frame-to-frame motion.
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

    # Normalize forensic scores
    ela = normalize_scores(ela)

    dct = normalize_scores(dct)

    flow = normalize_scores(flow)

    # Convert scores into column vectors
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

    # Combine all features
    fused_features = np.concatenate(
        [
            spatial,
            ela,
            dct,
            flow
        ],
        axis=1
    )

    print()
    print(
        "Feature fusion completed!"
    )

    print(
        f"Fused feature shape: "
        f"{fused_features.shape}"
    )

    np.save(
        "fused_features.npy",
        fused_features
    )

    print(
        "Saved to fused_features.npy"
    )


if __name__ == "__main__":

    create_fused_features()