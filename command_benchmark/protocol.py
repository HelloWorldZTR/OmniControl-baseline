"""Versioned protocol, canonical coordinates, provenance and atomic artifacts.

This module and the generation adapter remain compatible with Python 3.7.
Only export/scoring import the independently installed frankenmotion repository.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

TASK_NAMES = (
    "walk",
    "back_walk",
    "sidestep",
    "jog",
    "reach",
    "jump",
    "squat",
    "strike",
    "kick",
)
SEEDS = (1928, 1938, 1948)
FPS = 20
MODEL_FRAMES = 196
RELEASE = "motion13_reach_v2_20261021"
WORKSPACE_SHA256 = "8df745f08b937ecff8e9abb85ad6e79af3a544c0bd9a5b23613d12d86b0ad17d"
REPOSITORY = Path(__file__).resolve().parents[1]
ADAPTER = {
    "version": "analytic_single_joint_v1",
    "fps": FPS,
    "model_frames": MODEL_FRAMES,
    "hold_frames": 5,
    "squat_plateau": [0.4, 0.6],
    "limb_plateau": [0.45, 0.55],
    "kick_lift_m": 0.35,
    "gravity_m_s2": 9.81,
    "coordinate_mapping": "F=(H.z,H.x,H.y); F: forward,left,up",
    "reference_selection": "scalar: nearest real-train median, ID tie-break; reach: workspace reference; first frame FK",
}

FIXED_TARGET_ADAPTER = {
    "version": "limb_fixed_target_v2",
    "fps": FPS,
    "model_frames": MODEL_FRAMES,
    "active_frames": 20,
    "window": "centered half-open [(T-20)//2, (T-20)//2+20)",
    "strike_target": "reference right shoulder + (command, 0, 0) in F",
    "kick_target": "(reference right foot X + command, reference right foot Y, 1.0) in F",
    "kick_absolute_height_m": 1.0,
    "uncontrolled_frames": "zero hint and false mask; no approach/retraction path",
    "root_control": False,
    "coordinate_mapping": "F=(H.z,H.x,H.y); F: forward,left,up",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name, dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_npz(path, **arrays):
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def git_revision(root):
    try:
        top_level = (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if Path(top_level).resolve() != Path(root).resolve():
            # A copied source snapshot may live inside another repository.
            # Never attribute that parent repository's commit to the snapshot.
            return None
        return (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def source_hashes(root, relative_paths):
    return {
        str(path): file_sha256(Path(root) / path) for path in sorted(relative_paths)
    }


def verify_sources(root, hashes):
    for relative, expected in hashes.items():
        if file_sha256(Path(root) / relative) != expected:
            raise ValueError("Source changed since protocol export: " + relative)


def task_by_name(protocol, name):
    return next(task for task in protocol["tasks"] if task["name"] == name)


def sample_id(task_name, command_index, prompt_index, seed):
    return "{}__c{:04d}__p{}__s{}".format(task_name, command_index, prompt_index, seed)


def samples(protocol, selected_tasks=None, maximum_commands=None):
    """Stable identities independent of batching, filtering and resume order."""
    for task in protocol["tasks"]:
        if selected_tasks is not None and task["name"] not in selected_tasks:
            continue
        count = len(task["commands"])
        indices = np.arange(count)
        if maximum_commands is not None:
            if maximum_commands < 1:
                raise ValueError("maximum_commands must be positive")
            indices = (
                np.linspace(0, count - 1, min(count, maximum_commands))
                .round()
                .astype(int)
            )
        for prompt_index, prompt in enumerate(task["prompts"]):
            for seed in protocol["seeds"]:
                for command_index in indices.tolist():
                    yield dict(
                        sample_id=sample_id(
                            task["name"], command_index, prompt_index, seed
                        ),
                        task=task["name"],
                        command_index=command_index,
                        command=task["commands"][command_index],
                        frames=task["lengths"][command_index],
                        prompt_index=prompt_index,
                        prompt=prompt,
                        seed=seed,
                    )


def load_protocol(path, require_references=True):
    protocol = read_json(path)
    expected = protocol.pop("protocol_id", None)
    if expected != identity(protocol):
        raise ValueError("Protocol content hash mismatch")
    protocol["protocol_id"] = expected
    if protocol["schema_version"] != 1 or protocol["adapter"] not in (
        ADAPTER,
        FIXED_TARGET_ADAPTER,
    ):
        raise ValueError("Unsupported protocol or analytic adapter")
    revised = protocol["adapter"] == FIXED_TARGET_ADAPTER
    names = ("strike", "kick") if revised else TASK_NAMES
    if tuple(task["name"] for task in protocol["tasks"]) != names:
        raise ValueError("Protocol tasks do not match the versioned adapter")
    if revised and any(
        task.get("hint_adapter") != FIXED_TARGET_ADAPTER for task in protocol["tasks"]
    ):
        raise ValueError("Missing or inconsistent per-task fixed target adapter")
    if revised and protocol.get("expected_samples") != 540:
        raise ValueError("Revised limb protocol must contain 540 samples")
    if tuple(protocol["seeds"]) != SEEDS or protocol["fps"] != FPS:
        raise ValueError("Protocol seed/FPS mismatch")
    if require_references and not protocol["references_ready"]:
        raise ValueError(
            "Plan-only protocol: export again with --skeleton and --amass-root"
        )
    for task in protocol["tasks"]:
        expected_count = 2360 if task["name"] == "reach" else 30
        commands = np.asarray(task["commands"], dtype=np.float32)
        if commands.shape != (expected_count, 3) or not np.isfinite(commands).all():
            raise ValueError("Invalid command grid: " + task["name"])
        if (
            len(task["lengths"]) != expected_count
            or len(task["prompts"]) != 3
            or len(set(task["prompts"])) != 3
        ):
            raise ValueError("Invalid lengths/prompts")
        if any(
            length != task["definition"]["evaluation_frames"]
            for length in task["lengths"]
        ):
            raise ValueError("Unexpected task duration")
        if require_references:
            pose = np.asarray(task["reference"]["positions_f"], dtype=np.float32)
            if pose.shape != (22, 3) or not np.isfinite(pose).all():
                raise ValueError("Reference must be 22 finite joint positions")
    return protocol


def humanml_to_franken(positions):
    return np.asarray(positions)[..., [2, 0, 1]].copy()


def franken_to_humanml(positions):
    return np.asarray(positions)[..., [1, 2, 0]].copy()


def canonicalize(positions):
    """Align initial hips to +X without changing scale or height; [T,J,3]."""
    positions = np.asarray(positions, dtype=np.float32).copy()
    lateral = positions[0, 1, :2] - positions[0, 2, :2]
    if np.linalg.norm(lateral) < 1e-8:
        raise ValueError("Degenerate initial hips; cannot define heading")
    heading = np.arctan2(lateral[1], lateral[0]) - np.pi / 2
    rotation = np.array(
        [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]],
        dtype=np.float32,
    )
    positions[..., :2] -= positions[0, 0, :2].copy()
    positions[..., :2] = positions[..., :2] @ rotation.T
    return positions


def archive_path(output, sample):
    return Path(output) / "samples" / sample["task"] / (sample["sample_id"] + ".npz")


def read_sample(path, sample, run_id):
    """Reject stale, truncated or misidentified files instead of silently skipping."""
    with np.load(path, allow_pickle=False) as archive:
        if (
            str(archive["run_id"]) != run_id
            or str(archive["sample_id"]) != sample["sample_id"]
        ):
            raise ValueError("Sample/run identity mismatch: " + str(path))
        if int(archive["length"]) != sample["frames"] or int(archive["fps"]) != FPS:
            raise ValueError("Sample timing mismatch: " + str(path))
        if not np.array_equal(
            archive["requested"], np.asarray(sample["command"], dtype=np.float32)
        ):
            raise ValueError("Sample command mismatch: " + str(path))
        arrays = {name: archive[name].copy() for name in archive.files}
    positions = arrays["positions_h"]
    if positions.shape != (MODEL_FRAMES, 22, 3) or not np.isfinite(positions).all():
        raise ValueError("Invalid generated joint positions: " + str(path))
    if arrays["hint_h"].shape != positions.shape or arrays["hint_mask"].shape != (
        MODEL_FRAMES,
        22,
    ):
        raise ValueError("Invalid saved hint: " + str(path))
    return arrays
