#!/usr/bin/env python3
"""Compare the complete limb rerun and combine task summaries with seven frozen tasks."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = read(args.original / "results.json")
    new = read(args.revised / "results.json")
    assert old["complete"] and old["scored_samples"] == old["expected_samples"] == 23400
    assert new["complete"] and new["scored_samples"] == new["expected_samples"] == 540
    replacement = {task["task"]: task for task in new["tasks"]}
    assert set(replacement) == {"strike", "kick"}
    parent_protocol = read(args.original / "protocol.json")
    revised_protocol = read(args.revised / "protocol.json")
    assert (
        revised_protocol["revision"]["parent_protocol_id"]
        == parent_protocol["protocol_id"]
    )
    for task in revised_protocol["tasks"]:
        parent = next(
            item for item in parent_protocol["tasks"] if item["name"] == task["name"]
        )
        assert all(task[key] == value for key, value in parent.items())
    for task in new["tasks"]:
        parent = next(item for item in old["tasks"] if item["task"] == task["task"])
        assert task["samples"] == parent["samples"] == 270
        np.testing.assert_array_equal(task["requested"], parent["requested"])
        assert [(curve["prompt"], curve["seed"]) for curve in task["curves"]] == [
            (curve["prompt"], curve["seed"]) for curve in parent["curves"]
        ]
    tasks = []
    for task in old["tasks"]:
        source = new if task["task"] in replacement else old
        selected = copy.deepcopy(replacement.get(task["task"], task))
        selected["source_run_id"] = source["run_id"]
        selected["source_protocol_id"] = source["protocol_id"]
        tasks.append(selected)
    combined = dict(
        status="combined_task_summaries_from_two_runs",
        complete=True,
        expected_samples=23400,
        scored_samples=sum(task["samples"] for task in tasks),
        reused_samples=22860,
        regenerated_samples=540,
        macro_normalized_mae=float(np.mean([task["normalized_mae"] for task in tasks])),
        original_macro_normalized_mae=old["macro_normalized_mae"],
        tasks=tasks,
        sources={
            str(directory.resolve() / "results.json"): hashlib.sha256(
                (directory / "results.json").read_bytes()
            ).hexdigest()
            for directory in [args.original, args.revised]
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "combined_results.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    with (args.output / "combined_summary.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["task", "samples", "mae", "normalized_mae", "source_run_id"],
        )
        writer.writeheader()
        writer.writerows(
            {key: task[key] for key in writer.fieldnames} for task in tasks
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for axis, name in zip(axes, ["strike", "kick"]):
        for results, label, color in [
            (old, "Original trajectory", "#9c6a52"),
            (new, "Fixed target v2", "#245db5"),
        ]:
            task = next(item for item in results["tasks"] if item["task"] == name)
            requested = np.asarray(task["requested"])[:, 0]
            mean = np.asarray(task["measured"])[:, 0]
            std = np.asarray(task["measured_std"])[:, 0]
            axis.plot(
                requested,
                mean,
                color=color,
                label="{} (MAE {:.3f} m)".format(label, task["mae"][0]),
            )
            axis.fill_between(
                requested, mean - std, mean + std, color=color, alpha=0.15
            )
        axis.plot(requested, requested, "k--", linewidth=1, label="Ideal response")
        axis.set(
            title=name.title(), xlabel="Command (m)", ylabel="Measured excursion (m)"
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    figure.suptitle(
        "Same commands, prompts and seeds | 270 samples per task\nBands: ±1 population std; unchanged relative-excursion scoring"
    )
    for extension in ["png", "svg"]:
        figure.savefig(args.output / ("limb_revision_comparison." + extension), dpi=180)
    print("Combined nine-task nMAE:", combined["macro_normalized_mae"])


if __name__ == "__main__":
    main()
