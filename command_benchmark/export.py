"""Export upstream plans and first-frame FK references without loading a model."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from .protocol import (
    ADAPTER,
    FPS,
    RELEASE,
    SEEDS,
    TASK_NAMES,
    WORKSPACE_SHA256,
    REPOSITORY,
    canonicalize,
    file_sha256,
    git_revision,
    identity,
    read_json,
    source_hashes,
    write_json,
)


def connect_frankenmotion(root):
    root = Path(root).resolve()
    if not (root / "shared_motion/tasks/registry.py").is_file():
        raise FileNotFoundError("Not a frankenmotion checkout: " + str(root))
    sys.path.insert(0, str(root))
    return root


def upstream_source_paths(root):
    paths = [
        path.relative_to(root) for path in (root / "shared_motion/tasks").glob("*.py")
    ]
    paths += [
        Path(relative)
        for relative in (
            "shared_motion/training/benchmark.py",
            "shared_motion/utils/geometry.py",
            "config/benchmark/dense.yaml",
            "config/evaluate.yaml",
            "config/task/parameters/reach.yaml",
            "config/data_processing/task_rules/reach.yaml",
            "data_processing/tasks/reach/rules.py",
            "data_processing/tasks/sidestep/rules.py",
            "data_processing/pipeline/sources.py",
            "data_processing/pipeline/transforms.py",
            "data_processing/common/slerp.py",
            "src/tools/geometry.py",
            "src/tools/smplrifke_feats.py",
        )
    ]
    return paths


def real_training_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    return [
        row
        for row in rows
        if row["split"] == "train" and row["source"]["dataset"] == "amass"
    ]


def choose_reference(name, rows, workspace):
    if name == "reach":
        selected_id = workspace["reference"]["episode_id"]
        return next(row for row in rows if row["id"] == selected_id)
    values = np.asarray(
        [row["expected_quantity"] for row in rows], dtype=float
    ).reshape(-1)
    median = float(np.median(values))
    return min(
        rows,
        key=lambda row: (
            abs(float(np.asarray(row["expected_quantity"]).reshape(-1)[0]) - median),
            row["id"],
        ),
    )


def reference_pose(row, skeleton, amass_roots):
    """Reconstruct the selected crop, but export/use only its first FK pose."""
    import torch
    from data_processing.pipeline.sources import read_raw, resolve_amass_source
    from data_processing.pipeline.transforms import TransformContext, apply_transforms

    roots = {
        "amass": str(amass_roots[0]),
        "amass_roots": [str(path) for path in amass_roots[1:]],
    }
    raw_path = resolve_amass_source(roots, row["source"]["path"])
    if file_sha256(raw_path) != row["source"]["sha256"]:
        raise ValueError("Raw reference source hash mismatch: " + str(raw_path))
    sample = apply_transforms(
        read_raw(raw_path), row["transforms"], TransformContext(skeleton)
    )
    first_frame = np.asarray(sample["motion"][:1], dtype=np.float32).copy()
    with torch.no_grad():
        positions = (
            skeleton(torch.from_numpy(first_frame)[None])[0, :, :22].cpu().numpy()
        )
    return canonicalize(positions)[0].tolist()


def build_protocol(frankenmotion_root, skeleton_path=None, amass_roots=()):
    root = connect_frankenmotion(frankenmotion_root)
    from omegaconf import OmegaConf
    from shared_motion.tasks import get_task
    from shared_motion.tasks.evaluation import references_from_manifest
    from shared_motion.tasks.reach_evaluation import load_workspace

    episode_root = root / "data/annotations" / RELEASE / "episodes"
    settings = OmegaConf.load(root / "config/task/parameters/reach.yaml")
    settings.workspace = str(root / settings.workspace)
    settings.maximum_points = None
    if settings.workspace_sha256 != WORKSPACE_SHA256:
        raise ValueError("Reach workspace version differs from the approved protocol")
    workspace = load_workspace(settings.workspace, WORKSPACE_SHA256)
    grid = OmegaConf.load(root / "config/benchmark/dense.yaml")
    evaluate_config = OmegaConf.load(root / "config/evaluate.yaml")
    if (grid.bins, grid.points_per_bin, grid.design_seed) != (10, 3, 20261020):
        raise ValueError(
            "Upstream dense grid changed; review protocol before exporting"
        )
    if tuple(evaluate_config.seeds) != SEEDS:
        raise ValueError("Upstream benchmark seeds changed")
    skeleton = None
    if skeleton_path is not None:
        from shared_motion.utils.geometry import Skeleton

        if not amass_roots:
            raise ValueError("--skeleton requires at least one --amass-root")
        if file_sha256(skeleton_path) != workspace["skeleton_sha256"]:
            raise ValueError(
                "Reference skeleton differs from frozen workspace skeleton"
            )
        skeleton = Skeleton(skeleton_path)
    protocol = dict(
        schema_version=1,
        fps=FPS,
        seeds=list(SEEDS),
        adapter=copy.deepcopy(ADAPTER),
        release=RELEASE,
        references_ready=skeleton is not None,
        source=dict(
            frankenmotion_revision=git_revision(root),
            files=source_hashes(root, upstream_source_paths(root)),
            exporter_files=source_hashes(
                REPOSITORY,
                [
                    Path("command_benchmark/export.py"),
                    Path("command_benchmark/protocol.py"),
                ],
            ),
            manifests={},
            workspace_sha256=WORKSPACE_SHA256,
            skeleton_sha256=workspace["skeleton_sha256"],
        ),
        tasks=[],
    )
    for name in TASK_NAMES:
        task = get_task(name)
        references = references_from_manifest(episode_root, name)
        manifest = episode_root / (name + ".jsonl")
        if name != "reach":
            old_manifest = (
                root / "data/annotations/motion13_v1_20261020/episodes" / manifest.name
            )
            if file_sha256(old_manifest) != file_sha256(manifest):
                raise ValueError(
                    "Non-reach annotations changed between v1 and v2: " + name
                )
        protocol["source"]["manifests"][name] = file_sha256(manifest)
        plan = task.evaluation_plans(
            "benchmark",
            evaluate_config.points,
            settings if name == "reach" else None,
            references,
            command_grid=grid,
        )[0]
        expected = 2360 if name == "reach" else 30
        if len(plan.commands) != expected:
            raise ValueError("Unexpected command count for " + name)
        selected = choose_reference(name, real_training_rows(manifest), workspace)
        reference = dict(
            episode_id=selected["id"],
            source=selected["source"],
            transforms=selected["transforms"],
            source_split="train",
            expected_quantity=selected["expected_quantity"],
            pose_frame_in_crop=0,
            positions_f=(
                reference_pose(selected, skeleton, amass_roots)
                if skeleton is not None
                else None
            ),
        )
        protocol["tasks"].append(
            dict(
                name=name,
                definition=task.definition(),
                prompts=list(task.prompts),
                commands=plan.commands.float().tolist(),
                lengths=plan.lengths,
                metadata=plan.metadata,
                reference=reference,
            )
        )
    protocol["expected_samples"] = sum(
        len(task["commands"]) * 9 for task in protocol["tasks"]
    )
    # JSON round-trip removes tuples from upstream definitions before identity.
    protocol = json.loads(json.dumps(protocol))
    protocol["protocol_id"] = identity(protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frankenmotion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path)
    parser.add_argument("--amass-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Export real commands/metadata only; generation will reject missing FK references",
    )
    args = parser.parse_args()
    if args.plan_only and (args.skeleton or args.amass_root):
        parser.error("--plan-only cannot be combined with reference assets")
    if not args.plan_only and (not args.skeleton or not args.amass_root):
        parser.error(
            "Provide --skeleton and --amass-root, or explicitly use --plan-only"
        )
    protocol = build_protocol(args.frankenmotion_root, args.skeleton, args.amass_root)
    if args.output.exists() and read_json(args.output) != protocol:
        raise ValueError(
            "Refusing to replace a different protocol; use a new output path"
        )
    write_json(args.output, protocol)
    print(
        "Exported {} samples; references_ready={}: {}".format(
            protocol["expected_samples"], protocol["references_ready"], args.output
        )
    )
