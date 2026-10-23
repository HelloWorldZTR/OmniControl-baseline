"""Fixed analytic trajectories in physical coordinates; no generated-pose fitting."""

import numpy as np

from .protocol import (
    ADAPTER,
    FIXED_TARGET_ADAPTER,
    FPS,
    MODEL_FRAMES,
    franken_to_humanml,
)


def smoothstep(values):
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def pulse(frames, plateau):
    start = ADAPTER["hold_frames"] - 1
    end = frames - ADAPTER["hold_frames"]
    rise_end, fall_start = [round(fraction * (frames - 1)) for fraction in plateau]
    if not start < rise_end <= fall_start < end:
        raise ValueError("Duration too short for fixed analytic pulse")
    indices = np.arange(frames, dtype=np.float64)
    return smoothstep((indices - start) / (rise_end - start)) * (
        1.0 - smoothstep((indices - fall_start) / (end - fall_start))
    )


def physical_hint(task, command, frames):
    """Return absolute HumanML XYZ [196,22,3] and explicit [196,22] mask."""
    name = task["name"]
    pose = np.asarray(task["reference"]["positions_f"], dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if not 10 <= frames <= MODEL_FRAMES:
        raise ValueError("Invalid real duration")
    if "hint_adapter" in task:
        if task["hint_adapter"] != FIXED_TARGET_ADAPTER or name not in (
            "strike",
            "kick",
        ):
            raise ValueError("Unsupported per-task hint adapter")
        window = FIXED_TARGET_ADAPTER["active_frames"]
        if frames < window:
            raise ValueError("Duration shorter than fixed target window")
        joint = 21 if name == "strike" else 11
        target = pose[17 if name == "strike" else 11].copy()
        target[0] += command[0]
        if name == "kick":
            target[2] = FIXED_TARGET_ADAPTER["kick_absolute_height_m"]
        start = (frames - window) // 2
        positions = np.zeros((MODEL_FRAMES, 22, 3), dtype=np.float32)
        mask = np.zeros((MODEL_FRAMES, 22), dtype=bool)
        positions[start : start + window, joint] = franken_to_humanml(target)
        mask[start : start + window, joint] = True
        return positions, mask
    height = pose[0, 2]
    times = np.arange(frames, dtype=np.float64) / FPS
    joint = 21 if name in ("reach", "strike") else 11 if name == "kick" else 0
    trajectory = np.repeat(pose[joint][None], frames, axis=0)
    if joint == 0:
        trajectory[:, :2] = 0
    if name in ("walk", "jog", "back_walk"):
        trajectory[:, 0] = command[0] * times * (-1 if name == "back_walk" else 1)
    elif name == "sidestep":
        trajectory[:, 1] = -command[0] * smoothstep(times / times[-1])
    elif name == "jump":
        apex_time = ((frames - 1) // 2) / FPS
        trajectory[:, 2] = height + np.maximum(
            0, command[0] - 0.5 * ADAPTER["gravity_m_s2"] * (times - apex_time) ** 2
        )
    elif name == "squat":
        trajectory[:, 2] = height - command[0] * pulse(frames, ADAPTER["squat_plateau"])
    elif name in ("strike", "kick"):
        profile = pulse(frames, ADAPTER["limb_plateau"])
        trajectory[:, 0] += command[0] * profile
        if name == "kick":
            trajectory[:, 2] += ADAPTER["kick_lift_m"] * profile
    elif name == "reach":
        hold = ADAPTER["hold_frames"]
        fraction = smoothstep(
            (np.arange(frames) - (hold - 1)) / (frames - 2 * hold + 1)
        )
        target = command.copy()
        target[2] += height
        trajectory = pose[21] + fraction[:, None] * (target - pose[21])
    else:
        raise ValueError("Unknown task: " + name)
    positions = np.zeros((MODEL_FRAMES, 22, 3), dtype=np.float32)
    mask = np.zeros((MODEL_FRAMES, 22), dtype=bool)
    positions[:frames, joint] = franken_to_humanml(trajectory)
    mask[:frames, joint] = True
    return positions, mask


def normalize_hint(positions, mask, mean, std):
    mean = np.asarray(mean, dtype=np.float32).reshape(22, 3)
    std = np.asarray(std, dtype=np.float32).reshape(22, 3)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid spatial normalization statistics")
    hint = np.where(mask[..., None], (positions - mean) / std, 0).astype(np.float32)
    # The original implementation uses sums, not an explicit mask. Fail rather
    # than perturb a legitimate target to work around a sentinel collision.
    if not np.array_equal(hint.sum(-1) != 0, mask):
        raise ValueError("Native joint-mask sentinel collision after normalization")
    if not np.array_equal(hint.reshape(MODEL_FRAMES, 66).sum(-1) != 0, mask.any(-1)):
        raise ValueError("Native sequence-mask sentinel collision")
    return hint.reshape(MODEL_FRAMES, 66)
