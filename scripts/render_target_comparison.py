#!/usr/bin/env python3
"""Render paired raw kick samples with synchronized frames and identical axes."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.samples:
        with np.load(path, allow_pickle=False) as archive:
            length = int(archive["length"])
            record = dict(
                sample_id=str(archive["sample_id"]),
                run_id=str(archive["run_id"]),
                command=float(archive["requested"][0]),
                fps=int(archive["fps"]),
                length=length,
                seed=int(archive["seed"]),
                prompt_index=int(archive["prompt_index"]),
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                positions=archive["positions_h"][:length][..., [2, 0, 1]].copy(),
                hints=archive["hint_h"][:length, 11][:, [2, 0, 1]].copy(),
                active=archive["hint_mask"][:length, 11].copy(),
            )
        assert record["sample_id"].startswith("kick__")
        target = np.unique(record["hints"][record["active"]], axis=0)
        assert target.shape == (1, 3), "Expected a constant fixed-target hint"
        record["target"] = target[0]
        records.append(record)
    for field in ["fps", "length", "seed", "prompt_index", "run_id"]:
        assert len({record[field] for record in records}) == 1, field
    args.output.mkdir(parents=True, exist_ok=True)
    points = np.concatenate(
        [
            np.concatenate([r["positions"].reshape(-1, 3), r["target"][None]])
            for r in records
        ]
    )
    lower = np.minimum(points.min(0), [0, 0, 0]) - 0.2
    upper = points.max(0) + 0.2
    span = max(2.25, float((upper - lower).max()))
    center = (upper + lower) / 2
    limits = [(float(value - span / 2), float(value + span / 2)) for value in center]
    chains = [
        ([0, 2, 5, 8, 11], "#d34943"),
        ([0, 1, 4, 7, 10], "#3b6eac"),
        ([0, 3, 6, 9, 12, 15], "#47596d"),
        ([9, 14, 17, 19, 21], "#76889b"),
        ([9, 13, 16, 18, 20], "#76889b"),
    ]
    figure, axes = plt.subplots(
        2,
        len(records),
        figsize=(5 * len(records), 9),
        squeeze=False,
        layout="constrained",
    )
    figure.suptitle(
        "OmniControl kick | fixed-target v2 | prompt {} | seed {}\n"
        "Same camera / scale / time; right leg red; target orange when active, gray otherwise".format(
            records[0]["prompt_index"], records[0]["seed"]
        ),
        fontsize=14,
    )
    artists = []
    labels = []
    for column, record in enumerate(records):
        for row, horizontal in enumerate([0, 1]):
            axis = axes[row, column]
            axis.set(
                xlim=limits[horizontal],
                ylim=limits[2],
                xlabel="Forward X (m)" if horizontal == 0 else "Left Y (m)",
                ylabel="Up Z (m)",
            )
            axis.set_title(
                "Command {:.3f} m | {}\nTarget F = ({:.3f}, {:.3f}, {:.3f}) m".format(
                    record["command"],
                    "side view" if horizontal == 0 else "front view",
                    *record["target"]
                ),
                fontsize=10,
            )
            axis.set_aspect("equal")
            axis.axhline(0, color="#8996a4", linewidth=1)
            axis.axvline(0, color="#c3cbd5", linewidth=0.7, linestyle=":")
            axis.grid(alpha=0.15)
            lines = [
                (
                    axis.plot(
                        [], [], color=color, linewidth=3, marker="o", markersize=3
                    )[0],
                    chain,
                )
                for chain, color in chains
            ]
            (marker,) = axis.plot(
                [record["target"][horizontal]],
                [record["target"][2]],
                marker="x",
                markersize=11,
                markeredgewidth=2,
                linestyle="",
                color="#aaa",
            )
            (root,) = axis.plot(
                [], [], marker="o", markersize=6, color="#288c7a", linestyle=""
            )
            artists.append((record, horizontal, lines, marker, root))
            if row == 0:
                labels.append(
                    (
                        axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9),
                        record,
                    )
                )

    def update(frame):
        for record, horizontal, lines, marker, root in artists:
            pose = record["positions"][frame]
            for line, chain in lines:
                line.set_data(pose[chain, horizontal], pose[chain, 2])
            marker.set_color("#e18514" if record["active"][frame] else "#aaa")
            root.set_data([pose[0, horizontal]], [pose[0, 2]])
        for label, record in labels:
            label.set_text(
                "t = {:.2f} s | hint {}".format(
                    frame / record["fps"], "ON" if record["active"][frame] else "OFF"
                )
            )
        return []

    animation = FuncAnimation(
        figure,
        update,
        frames=records[0]["length"],
        interval=1000 / records[0]["fps"],
        blit=False,
    )
    animation.save(
        args.output / "kick_targets_comparison.mp4",
        writer=FFMpegWriter(
            fps=records[0]["fps"],
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "19"],
        ),
        dpi=120,
    )
    animation.save(
        args.output / "kick_targets_comparison.gif",
        writer=PillowWriter(fps=records[0]["fps"]),
        dpi=80,
    )
    update(30)
    figure.savefig(args.output / "kick_targets_at_1_5s.png", dpi=150)
    plt.close(figure)
    metadata = {
        "camera_limits_f": limits,
        "motion_postprocessing": "none; coordinate permutation only",
        "samples": [
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in record.items()
                if key not in ["positions", "hints", "active"]
            }
            for record in records
        ],
    }
    (args.output / "render_sources.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Rendered paired comparison to", args.output.resolve())


if __name__ == "__main__":
    main()
