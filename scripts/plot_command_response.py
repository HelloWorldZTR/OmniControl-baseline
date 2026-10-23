"""Plot completed benchmark measurements without changing the frozen reports."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

SCALARS = ["walk", "back_walk", "sidestep", "jog", "jump", "squat", "strike", "kick"]
BLUE = "#2465ae"
RED = "#b64b47"


def load_task(reports, name):
    result = json.loads((reports / (name + ".json")).read_text())
    with np.load(reports / (name + ".npz")) as archive:
        requested = archive["requested"].copy()
        measured = archive["measured"].copy()
    assert measured.shape == (9, len(requested), requested.shape[1])
    assert np.isfinite(measured).all() and np.isfinite(requested).all()
    np.testing.assert_allclose(measured.mean(axis=0), result["measured"], atol=1e-6)
    np.testing.assert_allclose(measured.std(axis=0), result["measured_std"], atol=1e-6)
    return result, requested, measured


def panel(axis, data, name, component=0, revised=False, initial_frame=False):
    result, requested, measured = data
    x = requested[:, component]
    y = measured[:, :, component]
    mean, std = y.mean(axis=0), y.std(axis=0)
    if name == "reach":
        axis.scatter(
            np.tile(x, 9),
            y.ravel(),
            s=2,
            color="#8392a3",
            alpha=0.055,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            x, mean, s=5, color=BLUE, alpha=0.45, linewidths=0, rasterized=True
        )
        title = "Reach / " + "XYZ"[component]
        metric = "MAE {:.3f} m".format(result["mae"][component])
    else:
        order = np.argsort(x)
        x, y, mean, std = x[order], y[:, order], mean[order], std[order]
        for curve in y:
            axis.plot(x, curve, color=BLUE, alpha=0.22, linewidth=0.8)
        axis.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.15)
        axis.plot(x, mean, color=BLUE, linewidth=2.2)
        title = name.replace("_", " ").title()
        if initial_frame:
            title += " (initial-frame score)"
        elif revised:
            title += " (fixed target v2)"
        metric = "MAE {:.3f} {}  |  nMAE {:.3f}".format(
            float(np.asarray(result["mae"]).mean()),
            result["unit"],
            result["normalized_mae"],
        )
    # Shared X/Y limits preserve the geometry of the ideal response; include all samples.
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    margin = max((high - low) * 0.06, 0.01)
    limits = (low - margin, high + margin)
    axis.plot(limits, limits, "--", color=RED, linewidth=1.2)
    axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="Command ({})".format(result["unit"]),
        ylabel="Measured ({})".format(result["unit"]),
    )
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=20)
    axis.text(0, 1.025, metric, transform=axis.transAxes, fontsize=8.5, color="#536275")
    axis.grid(alpha=0.16)
    axis.set_axisbelow(True)
    axis.set_aspect("equal", adjustable="box")
    axis.spines[["top", "right"]].set_visible(False)


def legend(axis):
    axis.axis("off")
    handles = [
        Line2D(
            [],
            [],
            color=RED,
            linestyle="--",
            label="Ideal response: measured = command",
        ),
        Line2D([], [], color=BLUE, linewidth=2.2, label="Mean of 3 prompts × 3 seeds"),
        Patch(color=BLUE, alpha=0.15, label="Scalar band: ±1 population std"),
        Line2D(
            [], [], color=BLUE, alpha=0.3, label="Thin lines: 9 individual responses"
        ),
        Line2D(
            [],
            [],
            marker=".",
            linestyle="",
            color="#8392a3",
            label="Reach gray dots: individual samples",
        ),
        Line2D(
            [],
            [],
            marker=".",
            linestyle="",
            color=BLUE,
            label="Reach blue dots: per-target mean",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelspacing=1.3,
        borderaxespad=0,
    )
    axis.text(
        0,
        0.23,
        "23,400 scored motions\n30 commands per scalar task\n2,360 three-dimensional reach targets\nSeeds: 1928, 1938, 1948",
        transform=axis.transAxes,
        fontsize=10,
        linespacing=1.6,
        color="#536275",
    )
    axis.text(
        0,
        0.00,
        "Reach panels project the same 3D workspace.\nThey are not independent single-axis sweeps.\nPhysical command accuracy does not establish\naction semantics or motion quality.",
        transform=axis.transAxes,
        fontsize=9,
        linespacing=1.4,
        color="#536275",
    )


def save(figure, output, name):
    for extension in ["png", "svg"]:
        figure.savefig(output / (name + "." + extension), dpi=200, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument(
        "--limb-reports",
        type=Path,
        help="Replace strike/kick with a completed fixed-target rerun",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.reports / "results.json").read_text())
    assert (
        summary["complete"]
        and summary["scored_samples"] == summary["expected_samples"] == 23400
    )
    folders = {name: args.reports for name in SCALARS + ["reach"]}
    run_sources = {"original": summary}
    initial_frame = False
    if args.limb_reports:
        revised_summary = json.loads((args.limb_reports / "results.json").read_text())
        revised_protocol = json.loads((args.limb_reports / "protocol.json").read_text())
        assert (
            revised_summary["complete"]
            and revised_summary["scored_samples"]
            == revised_summary["expected_samples"]
            == 540
        )
        assert {task["task"] for task in revised_summary["tasks"]} == {"strike", "kick"}
        assert (
            revised_protocol["revision"]["parent_protocol_id"] == summary["protocol_id"]
        )
        folders.update(strike=args.limb_reports, kick=args.limb_reports)
        run_sources["fixed_target_v2"] = revised_summary
        initial_frame = (
            revised_summary.get("evaluation", {}).get("version")
            == "initial_root_frame_rescore_20261023"
        )
        for name in ("strike", "kick"):
            old_task, old_commands, _ = load_task(args.reports, name)
            new_task, new_commands, _ = load_task(args.limb_reports, name)
            np.testing.assert_array_equal(old_commands, new_commands)
            assert [
                (curve["prompt"], curve["seed"]) for curve in old_task["curves"]
            ] == [(curve["prompt"], curve["seed"]) for curve in new_task["curves"]]
    data = {name: load_task(folder, name) for name, folder in folders.items()}
    macro = float(np.mean([values[0]["normalized_mae"] for values in data.values()]))
    provenance_note = (
        "\nStrike / kick: fixed-target v2 rerun; other seven tasks: original run"
        if args.limb_reports
        else ""
    )
    if initial_frame:
        provenance_note = "\nStrike / kick: new initial-frame scoring on existing motions; other seven tasks: original scores"
    assert all(len(data[name][1]) == 30 for name in SCALARS)
    assert len(data["reach"][1]) == 2360
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#374558",
            "text.color": "#233247",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(4, 3, figsize=(14, 18), layout="constrained")
    fig.suptitle(
        "OmniControl | Command response\n23,400 motions · task-equal normalized MAE {:.4f}".format(
            macro
        )
        + provenance_note,
        fontsize=15,
    )
    for axis, name in zip(axes.flat, SCALARS):
        panel(
            axis,
            data[name],
            name,
            revised=bool(args.limb_reports and name in ("strike", "kick")),
            initial_frame=initial_frame and name in ("strike", "kick"),
        )
    legend(axes[2, 2])
    for component in range(3):
        panel(axes[3, component], data["reach"], "reach", component)
    save(fig, args.output, "command_response_overview")

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), layout="constrained")
    fig.suptitle(
        "OmniControl | Scalar command responses\nBlue: mean ±1 std; thin lines: 9 prompt/seed combinations; dashed: ideal response"
        + provenance_note,
        fontsize=14,
    )
    for axis, name in zip(axes.flat, SCALARS):
        panel(
            axis,
            data[name],
            name,
            revised=bool(args.limb_reports and name in ("strike", "kick")),
            initial_frame=initial_frame and name in ("strike", "kick"),
        )
    save(fig, args.output, "scalar_command_responses")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), layout="constrained")
    fig.suptitle(
        "OmniControl | Reach workspace projections\nBlue: per-target mean; gray: all 21,240 samples; dashed: ideal response. Other coordinates vary jointly.",
        fontsize=13,
    )
    for component, axis in enumerate(axes):
        panel(axis, data["reach"], "reach", component)
    save(fig, args.output, "reach_command_responses")
    sources = [args.reports / "results.json"] + [
        folders[n] / (n + ext) for n in SCALARS + ["reach"] for ext in [".json", ".npz"]
    ]
    if args.limb_reports:
        sources += [
            args.limb_reports / "results.json",
            args.limb_reports / "protocol.json",
        ]
        if initial_frame:
            sources.append(args.limb_reports / "evaluation.json")
    manifest = {
        "source_runs": {
            name: {
                key: source[key]
                for key in ["protocol_id", "run_id", "evaluation_id"]
                if key in source
            }
            for name, source in run_sources.items()
        },
        "task_sources": {
            name: str(folder.resolve()) for name, folder in folders.items()
        },
        "macro_normalized_mae": macro,
        "scored_samples": summary["scored_samples"],
        "std_ddof": 0,
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    (args.output / "plot_sources.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        "Saved overview, scalar and reach plots (PNG + SVG) to", args.output.resolve()
    )


if __name__ == "__main__":
    main()
