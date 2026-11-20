from __future__ import annotations

import numpy as np


NUM_HANDS = 2
NUM_KEYPOINTS = 21
COORDINATE_DIMS = 2
LEFT_WRIST_INDEX = 0
RIGHT_WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
EPSILON = 1e-6


def make_hand_mask(keypoints: np.ndarray) -> np.ndarray:
    return np.any(keypoints != 0.0, axis=(2, 3))


def stabilize_hand_slots(
    keypoints: np.ndarray,
    swap_ratio: float,
) -> tuple[np.ndarray, int]:
    stabilized = keypoints.copy()
    hand_mask = make_hand_mask(stabilized)
    swap_count = 0
    for frame_index in range(1, len(stabilized)):
        previous_valid = hand_mask[frame_index - 1]
        current_valid = hand_mask[frame_index]
        if not previous_valid.all() or not current_valid.all():
            continue
        previous = stabilized[frame_index - 1]
        current = stabilized[frame_index]
        same_cost = (
            np.linalg.norm(previous[0] - current[0], axis=-1).mean()
            + np.linalg.norm(previous[1] - current[1], axis=-1).mean()
        )
        swapped_cost = (
            np.linalg.norm(previous[0] - current[1], axis=-1).mean()
            + np.linalg.norm(previous[1] - current[0], axis=-1).mean()
        )
        if swapped_cost < same_cost * swap_ratio:
            stabilized[frame_index] = current[[1, 0]]
            swap_count += 1
    return stabilized, swap_count


def trim_empty_edges(
    keypoints: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    hand_mask = make_hand_mask(keypoints)
    frame_mask = hand_mask.any(axis=1)
    if not frame_mask.any():
        raise ValueError("No hand was detected in the entire sequence.")
    first_valid = int(np.argmax(frame_mask))
    last_valid = len(frame_mask) - 1 - int(np.argmax(frame_mask[::-1]))
    trimmed = keypoints[first_valid : last_valid + 1].copy()
    return trimmed, first_valid, len(keypoints) - last_valid - 1


def interpolate_short_gaps(
    keypoints: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    interpolated = keypoints.copy()
    original_mask = make_hand_mask(interpolated)
    interpolation_mask = np.zeros_like(original_mask, dtype=bool)
    interpolation_count = 0
    for hand_index in range(NUM_HANDS):
        valid = original_mask[:, hand_index]
        frame_index = 0
        while frame_index < len(interpolated):
            if valid[frame_index]:
                frame_index += 1
                continue
            gap_start = frame_index
            while frame_index < len(interpolated) and not valid[frame_index]:
                frame_index += 1
            gap_end = frame_index
            gap_length = gap_end - gap_start
            has_left_boundary = gap_start > 0 and valid[gap_start - 1]
            has_right_boundary = gap_end < len(interpolated) and valid[gap_end]
            if (
                gap_length <= max_gap
                and has_left_boundary
                and has_right_boundary
            ):
                left_points = interpolated[gap_start - 1, hand_index]
                right_points = interpolated[gap_end, hand_index]
                for offset in range(gap_length):
                    ratio = (offset + 1) / (gap_length + 1)
                    interpolated[gap_start + offset, hand_index] = (
                        left_points * (1.0 - ratio) + right_points * ratio
                    )
                    interpolation_mask[gap_start + offset, hand_index] = True
                    interpolation_count += 1
    return interpolated, interpolation_mask, interpolation_count


def palm_scale_for_hand(
    hand_points: np.ndarray,
    valid_frames: np.ndarray,
) -> float:
    if not valid_frames.any():
        return 0.0
    palm_lengths = np.linalg.norm(
        hand_points[:, MIDDLE_MCP_INDEX]
        - hand_points[:, LEFT_WRIST_INDEX],
        axis=-1,
    )
    palm_lengths = palm_lengths[
        valid_frames & (palm_lengths > EPSILON)
    ]
    if len(palm_lengths) == 0:
        return 0.0
    return float(np.median(palm_lengths))


def remove_isolated_outliers(
    keypoints: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    cleaned = keypoints.copy()
    hand_mask = make_hand_mask(cleaned)
    outlier_count = 0
    for hand_index in range(NUM_HANDS):
        palm_scale = palm_scale_for_hand(
            cleaned[:, hand_index],
            hand_mask[:, hand_index],
        )
        if palm_scale <= EPSILON:
            continue
        for frame_index in range(1, len(cleaned) - 1):
            if not hand_mask[
                frame_index - 1 : frame_index + 2,
                hand_index,
            ].all():
                continue
            previous = cleaned[frame_index - 1, hand_index]
            current = cleaned[frame_index, hand_index]
            following = cleaned[frame_index + 1, hand_index]
            previous_distance = (
                np.linalg.norm(current - previous, axis=-1).mean()
                / palm_scale
            )
            following_distance = (
                np.linalg.norm(following - current, axis=-1).mean()
                / palm_scale
            )
            bridge_distance = (
                np.linalg.norm(following - previous, axis=-1).mean()
                / palm_scale
            )
            isolated_jump = (
                previous_distance > threshold
                and following_distance > threshold
                and bridge_distance < threshold * 0.5
            )
            if isolated_jump:
                cleaned[frame_index, hand_index] = 0.0
                hand_mask[frame_index, hand_index] = False
                outlier_count += 1
    return cleaned, outlier_count


def compute_centers_and_scale(
    keypoints: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    hand_mask = make_hand_mask(keypoints)
    both_hands = hand_mask.all(axis=1)
    centers = np.zeros((len(keypoints), 2), dtype=np.float32)
    if both_hands.any():
        centers[both_hands] = (
            keypoints[both_hands, 0, LEFT_WRIST_INDEX]
            + keypoints[both_hands, 1, RIGHT_WRIST_INDEX]
        ) / 2.0
        frame_indices = np.arange(len(keypoints))
        valid_indices = frame_indices[both_hands]
        for coordinate_index in range(2):
            centers[:, coordinate_index] = np.interp(
                frame_indices,
                valid_indices,
                centers[both_hands, coordinate_index],
            )
    else:
        valid_wrists = []
        for hand_index in range(NUM_HANDS):
            valid = hand_mask[:, hand_index]
            if valid.any():
                valid_wrists.append(
                    keypoints[valid, hand_index, LEFT_WRIST_INDEX],
                )
        if not valid_wrists:
            raise ValueError("Cannot calculate a center without valid hands.")
        fixed_center = np.concatenate(valid_wrists, axis=0).mean(axis=0)
        centers[:] = fixed_center

    inter_wrist_scale = 0.0
    if both_hands.any():
        wrist_distances = np.linalg.norm(
            keypoints[both_hands, 0, LEFT_WRIST_INDEX]
            - keypoints[both_hands, 1, RIGHT_WRIST_INDEX],
            axis=-1,
        )
        wrist_distances = wrist_distances[wrist_distances > EPSILON]
        if len(wrist_distances):
            inter_wrist_scale = float(wrist_distances.mean())

    palm_scales = []
    for hand_index in range(NUM_HANDS):
        scale = palm_scale_for_hand(
            keypoints[:, hand_index],
            hand_mask[:, hand_index],
        )
        if scale > EPSILON:
            palm_scales.append(scale)
    palm_scale = float(np.median(palm_scales)) if palm_scales else 0.0
    if inter_wrist_scale > EPSILON:
        scale = max(inter_wrist_scale, palm_scale, EPSILON)
        scale_method = "mean_inter_wrist_with_palm_floor"
    elif palm_scale > EPSILON:
        scale = max(palm_scale, EPSILON)
        scale_method = "median_palm"
    else:
        raise ValueError("Cannot calculate a valid scale.")
    return centers, scale, scale_method


def normalize_positions(
    keypoints: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    centers, scale, scale_method = compute_centers_and_scale(keypoints)
    hand_mask = make_hand_mask(keypoints)
    normalized = np.zeros_like(keypoints, dtype=np.float32)
    for hand_index in range(NUM_HANDS):
        valid = hand_mask[:, hand_index]
        normalized[valid, hand_index] = (
            keypoints[valid, hand_index] - centers[valid, None, :]
        ) / scale
    return normalized, scale, scale_method


def add_velocity_and_acceleration(
    normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_mask = make_hand_mask(normalized)
    velocity = np.zeros_like(normalized, dtype=np.float32)
    acceleration = np.zeros_like(normalized, dtype=np.float32)
    for frame_index in range(1, len(normalized)):
        valid_pair = hand_mask[frame_index] & hand_mask[frame_index - 1]
        for hand_index in np.flatnonzero(valid_pair):
            velocity[frame_index, hand_index] = (
                normalized[frame_index, hand_index]
                - normalized[frame_index - 1, hand_index]
            )
    for frame_index in range(2, len(normalized)):
        valid_triplet = (
            hand_mask[frame_index]
            & hand_mask[frame_index - 1]
            & hand_mask[frame_index - 2]
        )
        for hand_index in np.flatnonzero(valid_triplet):
            acceleration[frame_index, hand_index] = (
                velocity[frame_index, hand_index]
                - velocity[frame_index - 1, hand_index]
            )
    features = np.concatenate(
        [normalized, velocity, acceleration],
        axis=-1,
    )
    return features, velocity, acceleration


def preprocess_sample(
    keypoints: np.ndarray,
    max_interpolation_gap: int = 3,
    swap_ratio: float = 0.65,
    outlier_threshold: float = 3.0,
) -> dict[str, object]:
    if keypoints.ndim != 4 or keypoints.shape[1:] != (2, 21, 2):
        raise ValueError(
            f"Expected keypoints (T,2,21,2), got {keypoints.shape}",
        )
    if len(keypoints) == 0:
        raise ValueError("Keypoint sequence must not be empty.")
    if not np.isfinite(keypoints).all():
        raise ValueError("Keypoint sequence contains NaN or Inf.")

    original_length = len(keypoints)
    observed_before_trim = make_hand_mask(keypoints)
    stabilized, swap_count = stabilize_hand_slots(
        keypoints,
        swap_ratio=swap_ratio,
    )
    trimmed, trimmed_start, trimmed_end = trim_empty_edges(stabilized)
    observed_hand_mask = make_hand_mask(trimmed)
    interpolated_once, first_interpolation_mask, first_interpolations = (
        interpolate_short_gaps(
            trimmed,
            max_gap=max_interpolation_gap,
        )
    )
    without_outliers, outlier_count = remove_isolated_outliers(
        interpolated_once,
        threshold=outlier_threshold,
    )
    cleaned, second_interpolation_mask, second_interpolations = (
        interpolate_short_gaps(
            without_outliers,
            max_gap=max_interpolation_gap,
        )
    )
    interpolation_mask = (
        first_interpolation_mask | second_interpolation_mask
    )
    hand_mask = make_hand_mask(cleaned)
    frame_mask = hand_mask.any(axis=1)
    normalized, scale, scale_method = normalize_positions(cleaned)
    features_4d, velocity_4d, acceleration_4d = (
        add_velocity_and_acceleration(normalized)
    )
    positions = normalized.reshape(len(normalized), 42, 2)
    velocity = velocity_4d.reshape(len(normalized), 42, 2)
    acceleration = acceleration_4d.reshape(len(normalized), 42, 2)
    features = features_4d.reshape(len(normalized), 42, 6)
    cleaned_keypoints = cleaned.reshape(len(cleaned), 42, 2)
    if not np.isfinite(features).all():
        raise ValueError("NaN or Inf appeared after preprocessing.")
    if features.shape != (len(cleaned), 42, 6):
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    return {
        "cleaned_keypoints": cleaned_keypoints.astype(np.float32),
        "positions": positions.astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "acceleration": acceleration.astype(np.float32),
        "features": features.astype(np.float32),
        "hand_mask": hand_mask.astype(bool),
        "frame_mask": frame_mask.astype(bool),
        "observed_hand_mask": observed_hand_mask.astype(bool),
        "interpolation_mask": interpolation_mask.astype(bool),
        "original_length": original_length,
        "processed_length": len(cleaned),
        "trimmed_start": trimmed_start,
        "trimmed_end": trimmed_end,
        "swap_count": swap_count,
        "outlier_count": outlier_count,
        "interpolation_count": (
            first_interpolations + second_interpolations
        ),
        "original_detected_hand_frames": int(
            observed_before_trim.sum(),
        ),
        "scale": scale,
        "scale_method": scale_method,
    }
