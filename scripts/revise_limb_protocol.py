#!/usr/bin/env python3
"""Derive an immutable two-task fixed-target rerun from the original protocol."""

import argparse
import copy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_benchmark.protocol import (
    ADAPTER,
    FIXED_TARGET_ADAPTER,
    file_sha256,
    identity,
    load_protocol,
    write_json,
)
from command_benchmark.hints import physical_hint


def derive(parent, source_path):
    if parent["adapter"] != ADAPTER:
        raise ValueError("Revision must derive from the original analytic protocol")
    revised = copy.deepcopy(parent)
    revised.pop("protocol_id")
    revised["adapter"] = copy.deepcopy(FIXED_TARGET_ADAPTER)
    revised["tasks"] = [
        task for task in revised["tasks"] if task["name"] in ("strike", "kick")
    ]
    revised["expected_samples"] = 540
    revised["revision"] = {
        "parent_protocol_id": parent["protocol_id"],
        "parent_protocol_sha256": file_sha256(source_path),
        "tasks_replaced": ["strike", "kick"],
        "unchanged": [
            "commands",
            "prompts",
            "seeds",
            "lengths",
            "measurement",
            "reference poses",
        ],
        "limitations": [
            "Fixed absolute target is a proxy for relative excursion; initial wrist/ankle positions are not enforced.",
            "No root or shoulder constraints; body compensation remains possible.",
            "Static reference reachability is reported, never used to clip or exclude commands.",
            "Strike uses shoulder-plane zero extension, not the already extended reference wrist.",
        ],
    }
    geometry = {}
    for task in revised["tasks"]:
        task["hint_adapter"] = copy.deepcopy(FIXED_TARGET_ADAPTER)
        pose = np.asarray(task["reference"]["positions_f"])
        chain = [17, 19, 21] if task["name"] == "strike" else [2, 5, 8, 11]
        chain_length = sum(
            np.linalg.norm(pose[end] - pose[start])
            for start, end in zip(chain[:-1], chain[1:])
        )
        distances = []
        for command, frames in zip(task["commands"], task["lengths"]):
            hint, mask = physical_hint(task, command, frames)
            target = hint[mask][0][[2, 0, 1]]
            distances.append(float(np.linalg.norm(target - pose[chain[0]])))
        geometry[task["name"]] = {
            "reference_chain_length_m": float(chain_length),
            "target_distance_to_static_chain_origin_m": distances,
            "exceeds_static_chain_length": [
                bool(value > chain_length) for value in distances
            ],
            "note": "Necessary distance bound only; not a pose/whole-motion feasibility certificate.",
        }
    revised["revision"]["static_geometry_audit"] = geometry
    revised["protocol_id"] = identity(revised)
    return revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not replace a frozen protocol: " + str(args.output))
    revised = derive(load_protocol(args.parent), args.parent)
    write_json(args.output, revised)
    load_protocol(args.output)
    print("Created", revised["protocol_id"], "with 540 samples")


if __name__ == "__main__":
    main()
