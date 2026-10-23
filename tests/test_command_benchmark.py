"""CPU protocol, measurement, native random-stream and resumability regressions.

Set FRANKENMOTION_ROOT to exercise the real frozen release and its metric code.
Synthetic poses/motions exist only in these tests, never in production export.
"""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from command_benchmark.export import build_protocol, reference_pose
from command_benchmark.generate import generate, paired_sample, prepare_run
from command_benchmark.hints import normalize_hint, physical_hint
from command_benchmark.protocol import (
    MODEL_FRAMES,
    FIXED_TARGET_ADAPTER,
    REPOSITORY,
    SEEDS,
    TASK_NAMES,
    archive_path,
    canonicalize,
    file_sha256,
    franken_to_humanml,
    humanml_to_franken,
    identity,
    load_protocol,
    read_sample,
    samples,
    task_by_name,
    write_json,
    write_npz,
)
from command_benchmark.score import measure_sample, save_reports, score_run, summarize


def reference_positions():
    positions = np.zeros((22, 3), dtype=np.float32)
    positions[:, 2] = 1.0
    positions[1, 1], positions[2, 1] = 0.1, -0.1
    positions[16] = [0, 0.18, 1.45]
    positions[17] = [0, -0.18, 1.45]
    positions[18] = [0, 0.2, 1.2]
    positions[19] = [0, -0.2, 1.2]
    positions[20] = [0.05, 0.2, 1.0]
    positions[21] = [0.05, -0.2, 1.0]
    positions[7] = [0, 0.1, 0.05]
    positions[8] = [0, -0.1, 0.05]
    positions[10] = [0.1, 0.1, 0.02]
    positions[11] = [0.1, -0.1, 0.02]
    return positions


def synthetic_motion(task, command, frames):
    """Ideal joints for measurement tests, with the planned controlled curve."""
    hint, mask = physical_hint(task, command, frames)
    pose = np.asarray(task["reference"]["positions_f"], dtype=np.float32)
    motion = np.repeat(pose[None], MODEL_FRAMES, axis=0)
    joint = int(np.flatnonzero(mask[0])[0])
    curve = humanml_to_franken(hint[:frames, joint])
    if joint == 0:
        motion[:frames] += (curve - pose[0])[:, None]
    else:
        motion[:frames, joint] = curve
        if task["name"] == "kick":
            motion[:frames, 8] += curve - pose[11]
    motion[frames:] = motion[frames - 1]
    return franken_to_humanml(motion), hint, mask


class AdapterTests(unittest.TestCase):
    def test_fixed_targets_only_in_center_window(self):
        for name, frames, joint, origin in [
            ("strike", 40, 21, 17),
            ("kick", 60, 11, 11),
        ]:
            pose = reference_positions()
            task = dict(
                name=name,
                reference=dict(positions_f=pose.tolist()),
                hint_adapter=copy.deepcopy(FIXED_TARGET_ADAPTER),
            )
            hint, mask = physical_hint(task, [0.4, 0, 0], frames)
            active_frames, active_joints = np.where(mask)
            np.testing.assert_array_equal(
                active_frames, np.arange((frames - 20) // 2, (frames + 20) // 2)
            )
            self.assertTrue((active_joints == joint).all())
            self.assertFalse(mask[:, 0].any())
            self.assertFalse(hint[~mask].any())
            expected = pose[origin].copy()
            expected[0] += 0.4
            if name == "kick":
                expected[2] = 1.0
            np.testing.assert_allclose(
                humanml_to_franken(hint[mask]), np.repeat(expected[None], 20, axis=0)
            )
            normalized = normalize_hint(hint, mask, np.full(66, 0.1), np.full(66, 0.7))
            np.testing.assert_array_equal(
                normalized.reshape(196, 22, 3).sum(-1) != 0, mask
            )

    def test_axis_roundtrip_and_heading_alignment(self):
        positions = reference_positions()[None].repeat(10, axis=0)
        positions[:, :, 0] += np.arange(10)[:, None] * 0.1
        np.testing.assert_array_equal(
            humanml_to_franken(franken_to_humanml(positions)), positions
        )
        heading = 0.63
        rotation = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        rotated = positions.copy()
        rotated[..., :2] = positions[..., :2] @ rotation.T + [4, -3]
        np.testing.assert_allclose(canonicalize(rotated), positions, atol=1e-6)

    def test_explicit_zero_mask_and_sentinel_collision(self):
        positions = np.zeros((MODEL_FRAMES, 22, 3), dtype=np.float32)
        mask = np.zeros((MODEL_FRAMES, 22), dtype=bool)
        mask[0, 0] = True
        mean = np.ones(66, dtype=np.float32)
        std = np.ones(66, dtype=np.float32)
        hint = normalize_hint(positions, mask, mean, std).reshape(MODEL_FRAMES, 22, 3)
        self.assertTrue((hint[0, 0] == -1).all())
        self.assertFalse(hint[~mask].any())
        np.testing.assert_allclose((hint + mean.reshape(22, 3))[mask], positions[mask])
        with self.assertRaisesRegex(ValueError, "sentinel"):
            normalize_hint(positions, mask, np.zeros(66), std)

    def test_normalized_error_before_mean_and_incomplete_panel(self):
        task = dict(
            name="walk",
            definition=dict(bounds=[[0.48, 1.88]], unit="m/s"),
            prompts=["one", "two", "three"],
            commands=[[1, 0, 0]],
            lengths=[60],
            metadata={},
        )
        entries = [
            dict(
                command_index=0,
                prompt_index=prompt_index,
                seed=seed,
                measured=[1 + error],
                quality={},
                hint_mean_error_m=0,
                skating_ratio=0,
            )
            for prompt_index in range(3)
            for seed, error in zip(SEEDS, [-0.3, 0, 0.3])
        ]
        result, _ = summarize(task, entries, SEEDS)
        self.assertAlmostEqual(result["measured"][0][0], 1)
        self.assertAlmostEqual(result["mae"][0], 0.2, places=6)
        self.assertAlmostEqual(result["normalized_mae"], 0.2 / 1.4, places=6)
        self.assertAlmostEqual(result["measured_std"][0][0], np.sqrt(0.06), places=6)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize(task, entries[:-1], SEEDS)


class UpstreamProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(
            os.environ.get(
                "FRANKENMOTION_ROOT", REPOSITORY.parents[1] / "frankenmotion"
            )
        )
        if not (root / "shared_motion").is_dir():
            raise unittest.SkipTest(
                "Set FRANKENMOTION_ROOT for upstream integration tests"
            )
        cls.root = root
        cls.plan = build_protocol(root)
        cls.protocol = copy.deepcopy(cls.plan)
        cls.protocol["references_ready"] = True
        for task in cls.protocol["tasks"]:
            task["reference"]["positions_f"] = reference_positions().tolist()
        cls.protocol.pop("protocol_id")
        cls.protocol["protocol_id"] = identity(cls.protocol)

    def test_real_release_counts_seeds_prompts_and_reference_selection(self):
        records = list(samples(self.plan))
        self.assertEqual(len(records), 23400)
        self.assertEqual(len({record["sample_id"] for record in records}), 23400)
        self.assertEqual(self.plan["seeds"], list(SEEDS))
        from shared_motion.tasks import get_task
        from shared_motion.tasks.evaluation import references_from_manifest
        from omegaconf import OmegaConf
        from shared_motion.tasks.command_grid import dense_commands

        for task in self.plan["tasks"]:
            self.assertEqual(task["prompts"], list(get_task(task["name"]).prompts))
            self.assertEqual(task["reference"]["source"]["dataset"], "amass")
            self.assertEqual(task["reference"]["source_split"], "train")
            if task["name"] != "reach":
                references = references_from_manifest(
                    self.root / "data/annotations/motion13_reach_v2_20261021/episodes",
                    task["name"],
                )
                commands, _ = dense_commands(
                    get_task(task["name"]),
                    references,
                    OmegaConf.create(
                        dict(bins=10, points_per_bin=3, design_seed=20261020)
                    ),
                )
                np.testing.assert_array_equal(
                    commands.numpy(), np.asarray(task["commands"], dtype=np.float32)
                )
                self.assertEqual(len(commands), 30)
        reach = task_by_name(self.plan, "reach")
        groups = reach["metadata"]["evaluation_groups"]
        self.assertEqual(groups.count("reference_verified"), 2292)
        self.assertEqual(groups.count("observed_training_anchor"), 68)
        self.assertNotIn("unverified", groups)
        self.assertEqual(len(list(samples(self.plan, ["reach"], 128))), 128 * 9)
        self.assertEqual(
            build_protocol(self.root)["protocol_id"], self.plan["protocol_id"]
        )

    def test_protocol_hash_and_missing_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            write_json(path, self.plan)
            self.assertEqual(
                load_protocol(path, False)["protocol_id"], self.plan["protocol_id"]
            )
            with self.assertRaisesRegex(ValueError, "Plan-only"):
                load_protocol(path)
            changed = copy.deepcopy(self.protocol)
            changed["tasks"][0]["commands"][0][0] += 0.1
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_protocol(path)

    def test_reference_export_uses_actual_conversion_and_first_crop_pose(self):
        from shared_motion.utils.geometry import Skeleton
        from data_processing.pipeline.transforms import pipeline

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rest = np.zeros((38, 3), dtype=np.float32)
            rest[:22] = reference_positions()
            rest[22], rest[37] = rest[20], rest[21]
            skeleton_path = directory / "skeleton.npz"
            np.savez(
                skeleton_path,
                J=rest,
                height=1.8,
                parents=[
                    -1,
                    0,
                    0,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    9,
                    9,
                    12,
                    13,
                    14,
                    16,
                    17,
                    18,
                    19,
                ],
            )
            source = directory / "raw.npz"
            translation = np.zeros((16, 3), dtype=np.float32)
            translation[:, 0] = np.linspace(0, 1, 16)
            np.savez(
                source,
                poses=np.zeros((16, 66), dtype=np.float32),
                trans=translation,
                mocap_framerate=20,
            )
            row = dict(
                source=dict(
                    dataset="amass", path="raw.npz", sha256=file_sha256(source)
                ),
                transforms=pipeline(start=2, stop=10),
            )
            pose = reference_pose(row, Skeleton(skeleton_path), [directory])
            expected = reference_positions().copy()
            expected[:, 2] -= expected[:, 2].min()
            np.testing.assert_allclose(pose, expected, atol=2e-6)
            row["source"]["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                reference_pose(row, Skeleton(skeleton_path), [directory])

    def test_all_nine_metrics_on_ideal_trajectories_and_padding(self):
        for task in self.protocol["tasks"]:
            with self.subTest(task=task["name"]):
                command = task["commands"][len(task["commands"]) // 2]
                frames = task["lengths"][0]
                motion, hint, mask = synthetic_motion(task, command, frames)
                measured, _ = measure_sample(task, motion, frames)
                dimensions = len(task["definition"]["bounds"])
                np.testing.assert_allclose(measured, command[:dimensions], atol=2e-6)
                motion[frames:] = 1e6
                padded, _ = measure_sample(task, motion, frames)
                np.testing.assert_array_equal(padded, measured)
                self.assertEqual(mask.sum(), frames)
                self.assertEqual(mask.any(0).sum(), 1)
                self.assertFalse(mask[frames:].any())
                if task["name"] == "reach":
                    target = np.array(command) + [0, 0, 1]
                    np.testing.assert_allclose(
                        humanml_to_franken(hint[frames - 5 : frames, 21]),
                        np.tile(target, (5, 1)),
                        atol=1e-7,
                    )

    def test_kick_metric_uses_ankle_not_controlled_foot(self):
        task = task_by_name(self.protocol, "kick")
        motion, _, _ = synthetic_motion(task, task["commands"][0], 60)
        motion[:60, 8] = franken_to_humanml(reference_positions()[8])
        measured, _ = measure_sample(task, motion, 60)
        self.assertEqual(float(measured[0]), 0)

    def test_all_targets_normalize_with_official_spatial_statistics(self):
        mean = np.load(REPOSITORY / "dataset/humanml_spatial_norm/Mean_raw.npy")
        std = np.load(REPOSITORY / "dataset/humanml_spatial_norm/Std_raw.npy")
        for task in self.protocol["tasks"]:
            for command, frames in zip(task["commands"], task["lengths"]):
                hint, mask = physical_hint(task, command, frames)
                normalized = normalize_hint(hint, mask, mean, std).reshape(
                    MODEL_FRAMES, 22, 3
                )
                restored = normalized * std.reshape(22, 3) + mean.reshape(22, 3)
                np.testing.assert_allclose(restored[mask], hint[mask], atol=2e-6)

    def test_reach_success_groups_bias_variance_and_quality(self):
        task = task_by_name(self.protocol, "reach")
        selected = [0, 2292]
        entries = []
        for prompt_index in range(3):
            for seed_index, seed in enumerate(SEEDS):
                for command_index in selected:
                    command = np.asarray(task["commands"][command_index])
                    measured = command + [0.04 if seed_index < 2 else 0.08, 0, 0]
                    entries.append(
                        dict(
                            command_index=command_index,
                            prompt_index=prompt_index,
                            seed=seed,
                            measured=measured.tolist(),
                            quality={"valid": float(seed_index != 1)},
                            hint_mean_error_m=0.01,
                            skating_ratio=0,
                        )
                    )
        result, _ = summarize(task, entries, SEEDS)
        metrics = result["position_metrics"]["all"]
        self.assertAlmostEqual(metrics["success_0.05m"], 2 / 3)
        self.assertAlmostEqual(metrics["joint_success_0.05m"], 1 / 3)
        self.assertEqual(metrics["success_0.1m"], 1)
        self.assertAlmostEqual(
            metrics["rmse_m"] ** 2,
            metrics["bias_rms_m"] ** 2 + metrics["std_rms_m"] ** 2,
            places=7,
        )
        self.assertEqual(
            result["position_metrics"]["observed_training_anchor"]["samples"], 9
        )
        self.assertEqual(result["position_metrics"]["reference_verified"]["samples"], 9)

    def test_generation_resume_and_changed_identity(self):
        class DiffusionStatistics:
            mean = torch.zeros(263)
            std = torch.ones(263)

        def fake_sample(model, diffusion, hints, records, device):
            return torch.zeros(len(records), 263, 1, MODEL_FRAMES)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = dict(
                protocol_id=self.protocol["protocol_id"], checkpoint_sha256="test-only"
            )
            run_id = prepare_run(output, metadata, self.protocol)
            self.assertEqual(prepare_run(output, metadata, self.protocol), run_id)
            with patch(
                "command_benchmark.generate.paired_sample", side_effect=fake_sample
            ) as sampling:
                result = generate(
                    self.protocol,
                    output,
                    None,
                    DiffusionStatistics(),
                    np.ones(66),
                    np.ones(66),
                    "cpu",
                    run_id,
                    ["walk"],
                    2,
                    2,
                )
                self.assertEqual(result["generated"], 18)
                self.assertEqual(sampling.call_count, 9)
                sampling.reset_mock()
                result = generate(
                    self.protocol,
                    output,
                    None,
                    DiffusionStatistics(),
                    np.ones(66),
                    np.ones(66),
                    "cpu",
                    run_id,
                    ["walk"],
                    2,
                    1,
                )
                self.assertEqual(result["reused"], 18)
                sampling.assert_not_called()
                record = next(samples(self.protocol, ["walk"], 2))
                archive_path(output, record).unlink()
                result = generate(
                    self.protocol,
                    output,
                    None,
                    DiffusionStatistics(),
                    np.ones(66),
                    np.ones(66),
                    "cpu",
                    run_id,
                    ["walk"],
                    2,
                    1,
                )
                self.assertEqual(result["generated"], 1)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    read_sample(archive_path(output, record), record, "wrong-run")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                prepare_run(
                    output, dict(metadata, checkpoint_sha256="changed"), self.protocol
                )

    def test_end_to_end_partial_scoring_reports_and_corrupt_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated"
            run = dict(protocol_id=self.protocol["protocol_id"])
            run["run_id"] = prepare_run(output, run, self.protocol)
            records = list(samples(self.protocol, maximum_commands=1))
            for record in records:
                task = task_by_name(self.protocol, record["task"])
                motion, hint, mask = synthetic_motion(
                    task, record["command"], record["frames"]
                )
                write_npz(
                    archive_path(output, record),
                    positions_h=motion,
                    hint_h=hint,
                    hint_mask=mask,
                    sample_id=np.array(record["sample_id"]),
                    run_id=np.array(run["run_id"]),
                    length=np.array(record["frames"]),
                    fps=np.array(20),
                    requested=np.asarray(record["command"], dtype=np.float32),
                )
            with self.assertRaisesRegex(ValueError, "Missing"):
                score_run(self.protocol, run, output)
            result, entries, panels = score_run(
                self.protocol, run, output, allow_partial=True
            )
            self.assertEqual(result["scored_samples"], 81)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["macro_normalized_mae"])
            self.assertEqual(len(result["tasks"]), 9)
            for task_result in result["tasks"]:
                self.assertLess(task_result["normalized_mae"], 1e-5)
            reports = Path(directory) / "reports"
            save_reports(reports, self.protocol, run, result, entries, panels, output)
            self.assertEqual(len(list(reports.glob("*_response.png"))), 9)
            self.assertEqual(len(list(reports.glob("*_motions.html"))), 9)
            self.assertEqual(
                len((reports / "samples.csv").read_text().splitlines()), 82
            )
            record = records[0]
            path = archive_path(output, record)
            arrays = read_sample(path, record, run["run_id"])
            arrays["hint_h"][0, 0, 0] += 0.1
            write_npz(path, **arrays)
            with self.assertRaisesRegex(ValueError, "Stored control differs"):
                score_run(self.protocol, run, output, allow_partial=True)


class NativeNoiseTests(unittest.TestCase):
    def test_native_ddpm_restarts_same_noise_for_chunk_and_prompt(self):
        from diffusion.gaussian_diffusion import (
            GaussianDiffusion,
            LossType,
            ModelMeanType,
            ModelVarType,
        )

        def statistics(path, *args, **kwargs):
            dimension = 66 if "raw" in str(path) else 263
            return (
                np.ones(dimension, dtype=np.float32)
                if "Std" in str(path)
                else np.zeros(dimension, dtype=np.float32)
            )

        with patch("diffusion.gaussian_diffusion.np.load", side_effect=statistics):
            diffusion = GaussianDiffusion(
                betas=np.linspace(0.0001, 0.02, 12),
                model_mean_type=ModelMeanType.START_X,
                model_var_type=ModelVarType.FIXED_SMALL,
                loss_type=LossType.MSE,
            )

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, motion, timesteps, y=None):
                bias = y["hint"].sum((1, 2)).view(-1, 1, 1, 1) * 1e-4
                return motion * 0.2 + bias + self.anchor

        model = ToyModel().eval()
        records = [
            dict(task="walk", prompt_index=0, seed=1928, frames=60, prompt="walk")
            for _ in range(3)
        ]
        hints = np.zeros((3, MODEL_FRAMES, 66), dtype=np.float32)
        hints[:, :60, 0] = np.asarray([1, 2, 3])[:, None]
        # Isolate the native stochastic sampler from expensive geometry guidance;
        # full-model inference is a separate GPU smoke test, never claimed here.
        with patch.object(
            diffusion, "guide", side_effect=lambda motion, *args, **kwargs: motion
        ):
            batched = paired_sample(model, diffusion, hints, records, "cpu")
            split = torch.cat(
                [
                    paired_sample(
                        model,
                        diffusion,
                        hints[index : index + 1],
                        records[index : index + 1],
                        "cpu",
                    )
                    for index in range(3)
                ]
            )
            torch.testing.assert_close(batched, split, rtol=0, atol=1e-6)
            repeat = paired_sample(model, diffusion, hints, records, "cpu")
            torch.testing.assert_close(batched, repeat, rtol=0, atol=0)
            changed = [dict(record, seed=1938) for record in records]
            alternate = paired_sample(model, diffusion, hints, changed, "cpu")
            self.assertFalse(torch.equal(batched, alternate))
            changed_prompt = [
                dict(record, prompt_index=1, prompt="another prompt")
                for record in records
            ]
            same_noise = paired_sample(model, diffusion, hints, changed_prompt, "cpu")
            torch.testing.assert_close(batched, same_noise, rtol=0, atol=0)

        # Exercise the original geometric loss as well, without a checkpoint.
        motion = torch.randn(1, 263, 1, MODEL_FRAMES)
        physical = torch.zeros(1, MODEL_FRAMES, 22, 3)
        physical[:, :60, 0, 1] = 1.0
        mask = torch.zeros(1, MODEL_FRAMES, 22, 1, dtype=torch.bool)
        mask[:, :60, 0] = True
        loss, gradient = diffusion.gradients(motion, physical, mask)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(float(gradient[..., 0].abs().sum()), 0)
        self.assertGreater(float(gradient[..., 1:60].abs().sum()), 0)
        self.assertAlmostEqual(
            float(diffusion.calc_grad_scale(mask)), 20 / 60, places=6
        )


if __name__ == "__main__":
    unittest.main()
