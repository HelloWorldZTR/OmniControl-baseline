"""Native OmniControl generation with paired random streams and resumable samples."""

import argparse
import hashlib
from itertools import groupby
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .hints import normalize_hint, physical_hint
from .protocol import (
    FPS,
    MODEL_FRAMES,
    REPOSITORY,
    TASK_NAMES,
    archive_path,
    file_sha256,
    git_revision,
    identity,
    load_protocol,
    read_json,
    read_sample,
    samples,
    source_hashes,
    task_by_name,
    write_json,
    write_npz,
)


def model_arguments(checkpoint):
    from utils.parser_util import (
        add_data_options,
        add_diffusion_options,
        add_model_options,
    )

    parser = argparse.ArgumentParser()
    add_data_options(parser)
    add_diffusion_options(parser)
    add_model_options(parser)
    arguments = parser.parse_args([])
    arguments.cond_mode = "both_text_spatial"
    path = checkpoint.parent / "args.json"
    saved = read_json(path) if path.exists() else {}
    for name in vars(arguments):
        if name in saved:
            setattr(arguments, name, saved[name])
    if arguments.dataset != "humanml" or arguments.cond_mode != "both_text_spatial":
        raise ValueError("This benchmark requires a HumanML text+spatial checkpoint")
    if (
        arguments.cond_mask_prob <= 0
        or arguments.unconstrained
        or arguments.diffusion_steps != 1000
    ):
        raise ValueError(
            "Checkpoint incompatible with fixed CFG 2.5 / 1000-step protocol"
        )
    return arguments


def required_assets(checkpoint):
    return {
        "checkpoint": checkpoint,
        "motion_mean": REPOSITORY / "dataset/HumanML3D/Mean.npy",
        "motion_std": REPOSITORY / "dataset/HumanML3D/Std.npy",
        "spatial_mean": REPOSITORY / "dataset/humanml_spatial_norm/Mean_raw.npy",
        "spatial_std": REPOSITORY / "dataset/humanml_spatial_norm/Std_raw.npy",
        "smpl": REPOSITORY / "body_models/smpl/SMPL_NEUTRAL.pkl",
        "joint_regressor": REPOSITORY / "body_models/smpl/J_regressor_extra.npy",
    }


def paired_sample(model, diffusion, hints, records, device):
    """Each call restarts a common noise stream, independent of chunk length."""
    import torch
    from utils.fixseed import fixseed

    if len({(row["task"], row["prompt_index"], row["seed"]) for row in records}) != 1:
        raise ValueError("A sampling batch must have one task, prompt and seed")
    lengths = torch.tensor([row["frames"] for row in records], device=device)
    condition = {
        "text": [row["prompt"] for row in records],
        "hint": torch.as_tensor(hints, device=device, dtype=torch.float32),
        "lengths": lengths,
        "mask": (torch.arange(MODEL_FRAMES, device=device)[None] < lengths[:, None])[
            :, None, None
        ],
        "scale": torch.full((len(records),), 2.5, device=device),
    }
    fixseed(records[0]["seed"])
    with torch.no_grad():
        return diffusion.p_sample_loop(
            model,
            (len(records), 263, 1, MODEL_FRAMES),
            clip_denoised=False,
            model_kwargs={"y": condition},
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=True,
        )


def token_audit(protocol):
    import clip
    from clip.clip import _tokenizer

    return {
        task["name"]: [
            dict(
                prompt=prompt,
                token_count=len(_tokenizer.encode(prompt)),
                truncated=len(_tokenizer.encode(prompt)) > 20,
                context_22_ids=clip.tokenize(
                    [prompt], context_length=22, truncate=True
                )[0].tolist(),
            )
            for prompt in task["prompts"]
        ]
        for task in protocol["tasks"]
    }


def clip_provenance(model):
    """Fingerprint the actual frozen text weights, including checkpoint overrides."""
    import clip

    digest = hashlib.sha256()
    for name, tensor in sorted(model.clip_model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    root = Path(clip.__file__).resolve().parent
    return dict(
        state_sha256=digest.hexdigest(),
        source_files=source_hashes(
            root, [path.relative_to(root) for path in root.glob("*.py")]
        ),
    )


def prepare_run(output, run, protocol):
    run = dict(run)
    run["run_id"] = identity(run)
    path = output / "run.json"
    if path.exists():
        if read_json(path) != run:
            raise ValueError("Run identity changed; use a separate output directory")
        if (
            load_protocol(output / "protocol.json")["protocol_id"]
            != protocol["protocol_id"]
        ):
            raise ValueError("Run protocol changed")
    else:
        if output.exists() and any(output.iterdir()):
            # Recover a crash between the two metadata writes, but never adopt
            # unowned motion files from an earlier run.
            if {path.name for path in output.iterdir()} != {
                "protocol.json"
            } or read_json(output / "protocol.json") != protocol:
                raise ValueError(
                    "Output is nonempty without run.json; use a new directory"
                )
        write_json(output / "protocol.json", protocol)
        write_json(path, run)
    return run["run_id"]


def generate(
    protocol,
    output,
    model,
    diffusion,
    mean,
    std,
    device,
    run_id,
    selected_tasks=None,
    maximum_commands=None,
    batch_size=8,
):
    """Atomic per-sample commits allow resume even after a partially saved batch."""
    import torch
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generated = 0
    skipped = 0
    planned = list(samples(protocol, selected_tasks, maximum_commands))
    for _, group in groupby(
        planned, key=lambda row: (row["task"], row["prompt_index"], row["seed"])
    ):
        pending = []
        for row in group:
            path = archive_path(output, row)
            if path.exists():
                read_sample(path, row, run_id)
                skipped += 1
            else:
                pending.append(row)
        for begin in range(0, len(pending), batch_size):
            records = pending[begin : begin + batch_size]
            controls = [
                physical_hint(
                    task_by_name(protocol, row["task"]), row["command"], row["frames"]
                )
                for row in records
            ]
            hints = np.stack(
                [
                    normalize_hint(positions, mask, mean, std)
                    for positions, mask in controls
                ]
            )
            batch_started = time.monotonic()
            features = paired_sample(model, diffusion, hints, records, device)
            with torch.no_grad():
                # Exactly the statistics used inside native spatial guidance.
                vectors = features[:, :263, 0].permute(0, 2, 1)
                vectors = vectors * diffusion.std.to(device) + diffusion.mean.to(device)
                positions = recover_from_ric(vectors, 22).detach().cpu().numpy()
            for row, joints, (hint, mask) in zip(records, positions, controls):
                if (
                    joints.shape != (MODEL_FRAMES, 22, 3)
                    or not np.isfinite(joints).all()
                ):
                    raise ValueError(
                        "Nonfinite or malformed output: " + row["sample_id"]
                    )
                write_npz(
                    archive_path(output, row),
                    positions_h=joints.astype(np.float32),
                    hint_h=hint,
                    hint_mask=mask,
                    sample_id=np.array(row["sample_id"]),
                    run_id=np.array(run_id),
                    length=np.array(row["frames"]),
                    fps=np.array(FPS),
                    requested=np.asarray(row["command"], dtype=np.float32),
                    prompt_index=np.array(row["prompt_index"]),
                    seed=np.array(row["seed"]),
                )
                generated += 1
            print(
                "{}: generated={}, reused={}, selected={}, batch_seconds={:.2f}".format(
                    records[-1]["sample_id"],
                    generated,
                    skipped,
                    len(planned),
                    time.monotonic() - batch_started,
                ),
                flush=True,
            )
    return dict(generated=generated, reused=skipped, selected=len(planned))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES)
    parser.add_argument(
        "--maximum-commands",
        type=int,
        help="Evenly indexed subset for smoke checks, never a full benchmark",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Report missing assets without importing the model or downloading weights",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol, require_references=not args.preflight)
    checkpoint = args.checkpoint.resolve()
    assets = required_assets(checkpoint)
    missing = {name: str(path) for name, path in assets.items() if not path.is_file()}
    if args.preflight:
        import json

        print(
            json.dumps(
                dict(
                    references_ready=protocol["references_ready"],
                    missing=missing,
                    expected_samples=protocol["expected_samples"],
                ),
                indent=2,
            )
        )
        return
    if missing:
        raise FileNotFoundError("Missing assets: " + str(missing))
    if args.output is None:
        parser.error("--output is required for generation")
    if args.batch_size < 1 or (
        args.maximum_commands is not None and args.maximum_commands < 1
    ):
        parser.error("Batch size and maximum commands must be positive")
    output = args.output.resolve()
    # Upstream diffusion/SMPL paths are repository-relative.
    os.chdir(str(REPOSITORY))
    import torch
    from model.cfg_sampler import ClassifierFreeSampleModel
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu for checks or cuda for the native benchmark")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a CUDA host for the full benchmark")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_args = model_arguments(checkpoint)
    model, diffusion = create_model_and_diffusion(
        model_args, SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    unexpected = set(state) - set(model.state_dict())
    if unexpected:
        raise ValueError("Unexpected checkpoint keys: " + str(sorted(unexpected)))
    load_model_wo_clip(model, state)
    audit = token_audit(protocol)
    clip_identity = clip_provenance(model)
    model = ClassifierFreeSampleModel(model).to(device).eval()
    code_paths = [
        Path("command_benchmark") / (name + ".py")
        for name in ("generate", "hints", "protocol")
    ]
    for directory in ("model", "diffusion", "utils"):
        code_paths += [
            path.relative_to(REPOSITORY)
            for path in (REPOSITORY / directory).glob("*.py")
        ]
    code_paths += [
        Path("data_loaders/humanml/scripts/motion_process.py"),
        Path("data_loaders/humanml/common/quaternion.py"),
    ]
    run = dict(
        schema_version=1,
        protocol_id=protocol["protocol_id"],
        model_arguments=vars(model_args),
        checkpoint_sha256=file_sha256(checkpoint),
        assets={name: file_sha256(path) for name, path in assets.items()},
        args_json_sha256=(
            file_sha256(checkpoint.parent / "args.json")
            if (checkpoint.parent / "args.json").exists()
            else None
        ),
        sampler="native_ddpm",
        steps=1000,
        guidance_param=2.5,
        const_noise=True,
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        device=str(device),
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        omnicontrol_revision=git_revision(REPOSITORY),
        files=source_hashes(REPOSITORY, code_paths),
        token_audit=audit,
        clip=clip_identity,
        limitations=[
            "196-frame unmasked attention; only real prefix scored",
            "native spatial gradients disabled at frame zero",
            "right foot hint (11), right ankle kick metric (8)",
            "absolute hints versus relative command measurements",
        ],
    )
    run_id = prepare_run(output, run, protocol)
    result = generate(
        protocol,
        output,
        model,
        diffusion,
        np.load(assets["spatial_mean"]),
        np.load(assets["spatial_std"]),
        device,
        run_id,
        args.tasks,
        args.maximum_commands,
        args.batch_size,
    )
    print(result)
