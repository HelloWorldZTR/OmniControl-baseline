"""Score generated joints with upstream task measurements and reach diagnostics."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from .export import connect_frankenmotion
from .hints import physical_hint
from .protocol import (
    REPOSITORY,
    archive_path,
    atomic_bytes,
    canonicalize,
    humanml_to_franken,
    identity,
    load_protocol,
    read_json,
    read_sample,
    samples,
    source_hashes,
    task_by_name,
    verify_sources,
    write_json,
    write_npz,
)


def measure_sample(task, positions_h, frames):
    """Crop before alignment and measurement; padded frames never enter scores."""
    import torch
    from shared_motion.tasks import get_task

    positions = canonicalize(humanml_to_franken(positions_h[:frames]))
    tensor = torch.from_numpy(positions)[None]
    definition = get_task(task["name"])
    with torch.no_grad():
        measured = definition.measure_positions(tensor).reshape(-1).cpu().numpy()
        quality = (
            {}
            if definition.diagnostics is None
            else {
                name: float(value[0])
                for name, value in definition.diagnostics(tensor).items()
            }
        )
    return measured, quality


def summarize(task, entries, seeds):
    """Aggregate a balanced command × prompt × seed panel, never average first."""
    command_indices = sorted({entry["command_index"] for entry in entries})
    curve_keys = [(prompt_index, seed) for prompt_index in range(3) for seed in seeds]
    by_identity = {
        (entry["prompt_index"], entry["seed"], entry["command_index"]): entry
        for entry in entries
    }
    if len(by_identity) != len(entries):
        raise ValueError("Duplicate measurements")
    if len(entries) != len(command_indices) * len(curve_keys):
        raise ValueError(
            "Incomplete prompt/seed panel for {}; finish generation before scoring".format(
                task["name"]
            )
        )
    ordered = [
        [by_identity[(prompt_index, seed, index)] for index in command_indices]
        for prompt_index, seed in curve_keys
    ]
    dimensions = len(task["definition"]["bounds"])
    requested = np.asarray(task["commands"], dtype=np.float32)[
        command_indices, :dimensions
    ]
    measured = np.asarray(
        [[entry["measured"] for entry in curve] for curve in ordered], dtype=np.float32
    )
    spans = np.asarray(
        [upper - lower for lower, upper in task["definition"]["bounds"]],
        dtype=np.float32,
    )
    errors = np.abs(measured - requested[None])
    result = dict(
        task=task["name"],
        unit=task["definition"]["unit"],
        dimensions=dimensions,
        command_indices=command_indices,
        commands=len(command_indices),
        samples=len(entries),
        frames_per_command=[task["lengths"][index] for index in command_indices],
        requested=requested.tolist(),
        measured=measured.mean(0).tolist(),
        measured_std=measured.std(0, ddof=0).tolist(),
        mae=errors.mean((0, 1)).tolist(),
        normalized_mae=float((errors / spans).mean()),
        per_command_mae=errors.mean(0).tolist(),
        curves=[],
        hint_mean_error_m=float(
            np.mean([entry["hint_mean_error_m"] for entry in entries])
        ),
        skating_ratio=float(np.mean([entry["skating_ratio"] for entry in entries])),
    )
    for curve_index, (prompt_index, seed) in enumerate(curve_keys):
        result["curves"].append(
            dict(
                prompt_index=prompt_index,
                prompt=task["prompts"][prompt_index],
                seed=seed,
                measured=measured[curve_index].tolist(),
                mae=errors[curve_index].mean(0).tolist(),
                normalized_mae=float((errors[curve_index] / spans).mean()),
            )
        )
    quality_names = sorted(entries[0]["quality"])
    if quality_names:
        result["quality"] = {
            name: float(np.mean([entry["quality"][name] for entry in entries]))
            for name in quality_names
        }
        for curve, curve_entries in zip(result["curves"], ordered):
            curve["quality"] = {
                name: float(
                    np.mean([entry["quality"][name] for entry in curve_entries])
                )
                for name in quality_names
            }
    metadata = task["metadata"]
    if metadata.get("position_tolerances_m"):
        difference = measured - requested[None]
        distances = np.linalg.norm(difference, axis=-1)
        quality = np.asarray(
            [[entry["quality"]["valid"] for entry in curve] for curve in ordered],
            dtype=bool,
        )
        groups = np.asarray(metadata["evaluation_groups"])[command_indices]
        nearest = np.asarray(metadata["nearest_real_train_m"])[command_indices]
        subsets = {"all": np.ones(len(command_indices), dtype=bool)}
        subsets.update({name: groups == name for name in sorted(set(groups))})
        subsets.update(real_supported=nearest <= 0.05, real_gap=nearest > 0.05)
        result["position_metrics"] = {}
        for name, selected in subsets.items():
            if not selected.any():
                continue
            residual = difference[:, selected]
            metrics = dict(
                commands=int(selected.sum()),
                samples=int(distances[:, selected].size),
                rmse_m=float(np.sqrt(np.mean(distances[:, selected] ** 2))),
                bias_rms_m=float(
                    np.sqrt(np.mean(np.sum(residual.mean(0) ** 2, axis=-1)))
                ),
                std_rms_m=float(
                    np.sqrt(np.mean(np.sum(residual.var(0, ddof=0), axis=-1)))
                ),
                quality_pass_rate=float(quality[:, selected].mean()),
            )
            for tolerance in metadata["position_tolerances_m"]:
                success = distances[:, selected] <= tolerance
                metrics["success_{:g}m".format(tolerance)] = float(success.mean())
                metrics["joint_success_{:g}m".format(tolerance)] = float(
                    (success & quality[:, selected]).mean()
                )
            result["position_metrics"][name] = metrics
        result["evaluation_groups"] = groups.tolist()
        result["nearest_real_train_m"] = nearest.tolist()
    return result, measured


def score_run(protocol, run, generated_root, allow_partial=False):
    import torch
    from data_loaders.humanml.utils.metrics import calculate_skating_ratio

    records = list(samples(protocol))
    present = [
        record for record in records if archive_path(generated_root, record).is_file()
    ]
    if not allow_partial and len(present) != len(records):
        raise ValueError(
            "Missing {} of {} samples; partial checks require --allow-partial".format(
                len(records) - len(present), len(records)
            )
        )
    if not present:
        raise ValueError("No generated samples found")
    entries = []
    for index, record in enumerate(present):
        task = task_by_name(protocol, record["task"])
        arrays = read_sample(
            archive_path(generated_root, record), record, run["run_id"]
        )
        expected_hint, expected_mask = physical_hint(
            task, record["command"], record["frames"]
        )
        if not np.array_equal(arrays["hint_mask"], expected_mask) or not np.array_equal(
            arrays["hint_h"], expected_hint
        ):
            raise ValueError(
                "Stored control differs from frozen adapter: " + record["sample_id"]
            )
        measured, quality = measure_sample(
            task, arrays["positions_h"], record["frames"]
        )
        joints = arrays["positions_h"][: record["frames"]]
        mask = expected_mask[: record["frames"]]
        hint_error = np.linalg.norm(
            joints - expected_hint[: record["frames"]], axis=-1
        )[mask]
        skating, _ = calculate_skating_ratio(
            torch.from_numpy(joints.transpose(1, 2, 0))[None]
        )
        entries.append(
            dict(
                record,
                measured=measured.tolist(),
                quality=quality,
                hint_mean_error_m=float(hint_error.mean()),
                skating_ratio=float(skating[0]),
            )
        )
        if (index + 1) % 500 == 0:
            print("Scored {}/{}".format(index + 1, len(present)), flush=True)
    results = []
    measured_panels = {}
    for task in protocol["tasks"]:
        selected = [entry for entry in entries if entry["task"] == task["name"]]
        if selected:
            result, panel = summarize(task, selected, protocol["seeds"])
            results.append(result)
            measured_panels[task["name"]] = panel
    complete = len(present) == len(records)
    result = dict(
        protocol_id=protocol["protocol_id"],
        run_id=run["run_id"],
        complete=complete,
        expected_samples=len(records),
        scored_samples=len(entries),
        # A smoke subset must never accidentally enter the official comparison table.
        macro_normalized_mae=(
            float(np.mean([task["normalized_mae"] for task in results]))
            if complete
            else None
        ),
        status="full_benchmark" if complete else "partial_smoke_only_not_comparable",
        tasks=results,
    )
    return result, entries, measured_panels


def write_csv(path, fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())


def save_reports(output, protocol, run, result, entries, panels, generated_root):
    from .reports import response_plot, motion_gallery

    result["scoring_files"] = source_hashes(
        REPOSITORY,
        [
            Path("command_benchmark") / (name + ".py")
            for name in ("score", "reports", "protocol")
        ],
    )
    write_json(output / "protocol.json", protocol)
    write_json(output / "run.json", run)
    write_json(output / "results.json", result)
    write_csv(
        output / "summary.csv",
        [
            "task",
            "samples",
            "unit",
            "mae",
            "normalized_mae",
            "hint_mean_error_m",
            "skating_ratio",
        ],
        [
            {
                name: task[name]
                for name in (
                    "task",
                    "samples",
                    "unit",
                    "mae",
                    "normalized_mae",
                    "hint_mean_error_m",
                    "skating_ratio",
                )
            }
            for task in result["tasks"]
        ],
    )
    rows = []
    for entry in entries:
        row = {
            name: entry[name]
            for name in (
                "sample_id",
                "task",
                "command_index",
                "prompt_index",
                "seed",
                "frames",
                "hint_mean_error_m",
                "skating_ratio",
            )
        }
        for component in range(3):
            row["requested_" + str(component)] = entry["command"][component]
            row["measured_" + str(component)] = (
                entry["measured"][component]
                if component < len(entry["measured"])
                else ""
            )
        row["quality"] = json.dumps(entry["quality"], sort_keys=True)
        rows.append(row)
    write_csv(output / "samples.csv", list(rows[0]), rows)
    galleries = []
    for task_result in result["tasks"]:
        task = task_by_name(protocol, task_result["task"])
        name = task["name"]
        write_json(output / (name + ".json"), task_result)
        write_npz(
            output / (name + ".npz"),
            measured=panels[name],
            requested=np.asarray(task_result["requested"], dtype=np.float32),
            command_indices=np.asarray(task_result["command_indices"]),
            prompt_indices=np.repeat(np.arange(3), len(protocol["seeds"])),
            seeds=np.tile(protocol["seeds"], 3),
        )
        response_plot(output / (name + "_response.png"), task, task_result)
        motion_gallery(
            output / (name + "_motions.html"),
            protocol,
            run,
            task,
            task_result,
            generated_root,
        )
        galleries.append(
            "- [{} motions]({}_motions.html) · [response]({}_response.png)".format(
                name, name, name
            )
        )
    text = (
        "# OmniControl command benchmark\n\nStatus: **{}**. Samples: {}/{}.\n\n".format(
            result["status"], result["scored_samples"], result["expected_samples"]
        )
    )
    if result["complete"]:
        text += "Task-macro normalized MAE: {:.6f}.\n\n".format(
            result["macro_normalized_mae"]
        )
    text += "Command error, spatial hint error and skating are separate metrics. "
    text += "Scalar metrics alone do not certify action semantics. Reach certificates apply to reference endpoints only.\n\n"
    text += "Native DDPM: 1000 steps; text CFG: 2.5; 196 model frames, real prefixes scored; seeds: 1928/1938/1948. "
    text += "Gallery curves use fixed low/median/high command choices (reach: shoulder radius); no best-seed selection.\n\n"
    text += "\n".join(galleries) + "\n\nProtocol: `{}`\n\nRun: `{}`\n".format(
        protocol["protocol_id"], run["run_id"]
    )
    atomic_bytes(output / "README.md", text.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frankenmotion-root", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.generated / "protocol.json")
    run = read_json(args.generated / "run.json")
    unsigned = dict(run)
    run_id = unsigned.pop("run_id")
    if identity(unsigned) != run_id or run["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Run provenance mismatch")
    verify_sources(args.frankenmotion_root, protocol["source"]["files"])
    verify_sources(REPOSITORY, run["files"])
    connect_frankenmotion(args.frankenmotion_root)
    result, entries, panels = score_run(
        protocol, run, args.generated, args.allow_partial
    )
    save_reports(args.output, protocol, run, result, entries, panels, args.generated)
    print(
        "{}: {} samples -> {}".format(
            result["status"], result["scored_samples"], args.output
        )
    )
