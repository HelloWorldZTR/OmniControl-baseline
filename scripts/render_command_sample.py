#!/usr/bin/env python3
"""Render original generated joints with a fixed camera and the actual hint mask."""

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
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.sample) as archive:
        length = int(archive["length"])
        fps = int(archive["fps"])
        sample_id = str(archive["sample_id"])
        run_id = str(archive["run_id"])
        command = float(archive["requested"][0])
        positions = archive["positions_h"][:length][..., [2, 0, 1]].copy()
        mask = archive["hint_mask"][:length]
        joints = np.flatnonzero(mask.any(axis=0))
        if len(joints) != 1:
            raise ValueError("Renderer expects a single controlled joint")
        joint = int(joints[0])
        targets = archive["hint_h"][:length, joint][:, [2, 0, 1]].copy()
        active = mask[:, joint].copy()
    task = sample_id.split("__")[0]
    chains = [
        ([0, 2, 5, 8, 11], "#d34943" if task == "kick" else "#3b6eac"),
        ([0, 1, 4, 7, 10], "#3b6eac"),
        ([0, 3, 6, 9, 12, 15], "#47596d"),
        ([9, 14, 17, 19, 21], "#d34943" if task == "strike" else "#76889b"),
        ([9, 13, 16, 18, 20], "#76889b"),
    ]
    points = np.concatenate([positions.reshape(-1, 3), targets[active]], axis=0)
    lower = np.minimum(points.min(axis=0), [0, 0, 0]) - 0.15
    upper = points.max(axis=0) + 0.15
    span = max(2.2, float((upper - lower).max()))
    center = (upper + lower) / 2
    limits = [(float(value - span / 2), float(value + span / 2)) for value in center]
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    figure.suptitle(
        "{} | command {:.3f} m\nRaw generated joints; active limb red; orange target only during hint window".format(
            sample_id, command
        ),
        fontsize=12,
    )
    artists = []
    for axis, horizontal in zip(axes, [0, 1]):
        axis.set(
            xlim=limits[horizontal],
            ylim=limits[2],
            xlabel="Forward X (m)" if horizontal == 0 else "Left Y (m)",
            ylabel="Up Z (m)",
            title="Side view" if horizontal == 0 else "Front view",
        )
        axis.set_aspect("equal")
        axis.axhline(0, color="#8996a4", linewidth=1)
        axis.grid(alpha=0.15)
        lines = [
            (
                axis.plot([], [], color=color, linewidth=3, marker="o", markersize=3)[
                    0
                ],
                chain,
            )
            for chain, color in chains
        ]
        (marker,) = axis.plot(
            [],
            [],
            marker="x",
            color="#e18514",
            markersize=11,
            markeredgewidth=2,
            linestyle="",
        )
        artists.append((horizontal, lines, marker))
    label = axes[0].text(0.02, 0.95, "", transform=axes[0].transAxes)

    def update(frame):
        for horizontal, lines, marker in artists:
            for line, chain in lines:
                line.set_data(
                    positions[frame, chain, horizontal], positions[frame, chain, 2]
                )
            marker.set_data(
                [targets[frame, horizontal]] if active[frame] else [],
                [targets[frame, 2]] if active[frame] else [],
            )
        label.set_text(
            "t = {:.2f} s | hint {}".format(
                frame / fps, "ON" if active[frame] else "OFF"
            )
        )
        return []

    animation = FuncAnimation(
        figure, update, frames=length, interval=1000 / fps, blit=False
    )
    animation.save(
        args.output / (sample_id + ".mp4"),
        writer=FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]
        ),
        dpi=120,
    )
    animation.save(
        args.output / (sample_id + ".gif"), writer=PillowWriter(fps=fps), dpi=80
    )
    plt.close(figure)
    figure, axes = plt.subplots(1, 6, figsize=(16, 4), layout="constrained")
    for axis, frame in zip(axes, np.linspace(0, length - 1, 6).round().astype(int)):
        for chain, color in chains:
            axis.plot(
                positions[frame, chain, 0],
                positions[frame, chain, 2],
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=2,
            )
        if active[frame]:
            axis.plot(
                targets[frame, 0],
                targets[frame, 2],
                marker="x",
                color="#e18514",
                markersize=9,
            )
        axis.set(
            xlim=limits[0],
            ylim=limits[2],
            xlabel="Forward X (m)",
            title="{:.2f} s | {}".format(frame / fps, "ON" if active[frame] else "OFF"),
        )
        axis.axhline(0, color="#8996a4", linewidth=0.8)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    figure.suptitle(
        "{} | command {:.3f} m | fixed camera; unmodified generated joints".format(
            sample_id, command
        )
    )
    figure.savefig(args.output / (sample_id + "_frames.png"), dpi=180)
    metadata = dict(
        sample_id=sample_id,
        run_id=run_id,
        command=command,
        fps=fps,
        length=length,
        input_sha256=hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        hint_frames=np.flatnonzero(active).tolist(),
        joint=joint,
    )
    (args.output / (sample_id + ".json")).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Rendered", sample_id)


if __name__ == "__main__":
    main()
