#!/usr/bin/env python3
"""Remeasure existing limb motions with versioned upstream scoring sources.

Generation commands, hints, run identity and protocol remain immutable. The new
measurement definitions and normalization bounds belong to a separate scoring
revision, never a replacement generation protocol.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_benchmark.export import connect_frankenmotion
from command_benchmark.protocol import (
    REPOSITORY,
    archive_path,
    file_sha256,
    humanml_to_franken,
    identity,
    load_protocol,
    read_json,
    read_sample,
    samples,
    source_hashes,
    verify_sources,
    write_json,
)
from command_benchmark.score import (
    measure_sample,
    save_reports,
    score_run,
    summarize,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frankenmotion-root", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--previous-reports", type=Path, required=True)
    parser.add_argument("--original-nine-reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new directory for each scoring revision")
    protocol = load_protocol(args.generated / "protocol.json")
    run = read_json(args.generated / "run.json")
    unsigned = dict(run)
    run_id = unsigned.pop("run_id")
    assert identity(unsigned) == run_id
    assert run["protocol_id"] == protocol["protocol_id"]
    assert {task["name"] for task in protocol["tasks"]} == {"strike", "kick"}
    verify_sources(REPOSITORY, run["files"])
    previous = read_json(args.previous_reports / "results.json")
    original = read_json(args.original_nine_reports / "results.json")
    assert previous["complete"] and previous["scored_samples"] == 540
    assert previous["run_id"] == run_id and original["complete"]

    # Freeze a small importable upstream package; never modify its original files.
    upstream = args.frankenmotion_root.resolve()
    snapshot = args.output / "scoring_source"
    source_files = list((upstream / "shared_motion/tasks").glob("*.py"))
    source_files += list((upstream / "config/task/parameters").glob("*.yaml"))
    source_files += [
        upstream / relative
        for relative in [
            "shared_motion/__init__.py",
            "shared_motion/utils/__init__.py",
            "shared_motion/utils/command_frame.py",
            "shared_motion/utils/file.py",
            "shared_motion/training/benchmark.py",
        ]
    ]
    for source in source_files:
        destination = snapshot / source.relative_to(upstream)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    connect_frankenmotion(snapshot)
    from shared_motion.tasks import get_task

    evaluated_tasks = copy.deepcopy(protocol["tasks"])
    for task in evaluated_tasks:
        task["definition"] = json.loads(json.dumps(get_task(task["name"]).definition()))
        assert task["definition"]["evaluation_frames"] == task["lengths"][0]
        assert task["definition"]["unit"] == "m"
    evaluation = dict(
        version="initial_root_frame_rescore_20261023",
        generation_protocol_id=protocol["protocol_id"],
        generation_run_id=run_id,
        source_files=source_hashes(
            snapshot, [source.relative_to(upstream) for source in source_files]
        ),
        definitions={task["name"]: task["definition"] for task in evaluated_tasks},
        previous_definitions={
            task["name"]: task["definition"] for task in protocol["tasks"]
        },
        command_sampling="Original frozen commands, prompts, seeds and lengths retained; no new grid or motion generation.",
        normalization="Current upstream declared bounds; previous spans provided separately for comparison.",
        unchanged_tasks="Other seven task scores and ranges reused from the original nine-task report, including original reach policy.",
        previous_results_sha256=file_sha256(args.previous_reports / "results.json"),
        original_nine_results_sha256=file_sha256(
            args.original_nine_reports / "results.json"
        ),
    )
    evaluation["evaluation_id"] = identity(evaluation)
    write_json(args.output / "evaluation.json", evaluation)

    # This validates every archive identity and stored hint against generation;
    # the imported upstream functions supply the NEW measurements.
    result, entries, _ = score_run(protocol, run, args.generated)
    assert result["complete"] and len(entries) == 540
    panels = {}
    result["tasks"] = []
    for task in evaluated_tasks:
        selected = [entry for entry in entries if entry["task"] == task["name"]]
        summary, panel = summarize(task, selected, protocol["seeds"])
        old_task = next(
            item for item in previous["tasks"] if item["task"] == task["name"]
        )
        old_definition = next(
            item["definition"]
            for item in protocol["tasks"]
            if item["name"] == task["name"]
        )
        old_span = old_definition["bounds"][0][1] - old_definition["bounds"][0][0]
        summary.update(
            evaluation_definition=task["definition"],
            evaluation_id=evaluation["evaluation_id"],
            previous_metric_mae=old_task["mae"],
            previous_metric_normalized_mae=old_task["normalized_mae"],
            new_metric_normalized_by_previous_span=float(
                np.mean(summary["mae"]) / old_span
            ),
        )
        result["tasks"].append(summary)
        panels[task["name"]] = panel
    result.update(
        status="complete_rescore_existing_commands",
        evaluation_id=evaluation["evaluation_id"],
        evaluation=evaluation,
        macro_normalized_mae=float(
            np.mean([task["normalized_mae"] for task in result["tasks"]])
        ),
        macro_mae_m=float(np.mean([task["mae"][0] for task in result["tasks"]])),
    )
    save_reports(
        args.output / "reports", protocol, run, result, entries, panels, args.generated
    )
    write_json(args.output / "reports/evaluation.json", evaluation)
    readme = args.output / "reports/README.md"
    readme.write_text(
        readme.read_text()
        + "\nScoring revision: `"
        + evaluation["evaluation_id"]
        + "`.\n\n"
        "protocol.json and run.json describe the unchanged generation. evaluation.json records the updated measurement functions and bounds. "
        "This rescores the existing narrow command grid, not the new training-label grid.\n"
    )

    replacement = {task["task"]: task for task in result["tasks"]}
    combined_tasks = []
    for task in original["tasks"]:
        selected = copy.deepcopy(replacement.get(task["task"], task))
        source = result if task["task"] in replacement else original
        selected.update(
            source_run_id=source["run_id"], source_protocol_id=source["protocol_id"]
        )
        combined_tasks.append(selected)
    macro = float(np.mean([task["normalized_mae"] for task in combined_tasks]))
    constant_span_macro = float(
        np.mean(
            [
                task.get(
                    "new_metric_normalized_by_previous_span", task["normalized_mae"]
                )
                for task in combined_tasks
            ]
        )
    )
    combined = dict(
        status="combined_seven_original_tasks_and_two_rescored_tasks",
        complete=True,
        expected_samples=23400,
        scored_samples=23400,
        rescored_samples=540,
        regenerated_samples=0,
        reused_other_task_samples=22860,
        macro_normalized_mae=macro,
        macro_normalized_mae_with_previous_spans=constant_span_macro,
        task_equal_mae_by_unit={
            unit: float(
                np.mean(
                    [
                        np.mean(task["mae"])
                        for task in combined_tasks
                        if task["unit"] == unit
                    ]
                )
            )
            for unit in ["m", "m/s"]
        },
        note="No mixed-unit physical MAE: use task-equal normalized MAE for the nine-task total.",
        evaluation_id=evaluation["evaluation_id"],
        tasks=combined_tasks,
    )
    write_json(args.output / "combined_results.json", combined)
    columns = ["task", "samples", "unit", "mae", "normalized_mae", "source_run_id"]
    write_csv(
        args.output / "combined_summary.csv",
        columns,
        [{key: task[key] for key in columns} for task in combined_tasks],
    )

    # Independent invocation in unaligned coordinates verifies wrapper frame handling.
    measured_lookup = {entry["sample_id"]: entry["measured"] for entry in entries}
    archive_hashes = {}
    for record in samples(protocol):
        path = archive_path(args.generated, record)
        archive_hashes[record["sample_id"]] = file_sha256(path)
        arrays = read_sample(path, record, run_id)
        positions = torch.from_numpy(
            humanml_to_franken(arrays["positions_h"][: record["frames"]])
        )[None]
        with torch.no_grad():
            direct = get_task(record["task"]).measure_positions(positions).numpy()
        np.testing.assert_allclose(
            direct, measured_lookup[record["sample_id"]], atol=2e-5
        )
    for task in evaluated_tasks:
        record = next(
            record for record in samples(protocol) if record["task"] == task["name"]
        )
        arrays = read_sample(archive_path(args.generated, record), record, run_id)
        modified = arrays["positions_h"].copy()
        modified[record["frames"] :] = 10000
        actual, _ = measure_sample(task, modified, record["frames"])
        np.testing.assert_allclose(
            actual, measured_lookup[record["sample_id"]], atol=1e-6
        )
    verify_sources(upstream, evaluation["source_files"])
    write_json(
        args.output / "audit.json",
        dict(
            unique_samples=len(measured_lookup),
            archive_sha256=archive_hashes,
            direct_upstream_measurement_matches=True,
            padding_invariance=True,
            unchanged_generation_identity=True,
        ),
    )
    print(
        json.dumps(
            {
                "tasks": [
                    {
                        key: task[key]
                        for key in [
                            "task",
                            "mae",
                            "normalized_mae",
                            "new_metric_normalized_by_previous_span",
                        ]
                    }
                    for task in result["tasks"]
                ],
                "nine_task_normalized_mae": macro,
                "nine_task_normalized_mae_previous_spans": constant_span_macro,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
