# Nine-task command-following benchmark

This evaluates the pretrained HumanML3D OmniControl model using the same physical
commands, three verbatim prompts and seeds `[1928, 1938, 1948]` as frankenmotion.
The adapter supplies one native spatially controlled joint, with a fixed analytic
trajectory. It does not train weights, add task-specific diffusion losses, select
successful seeds, or repair generated motions.

## Three independent stages

Run export and scoring in the **frankenmotion Python environment (Python ≥3.10)**.
Run generation in the **original OmniControl environment** from `environment.yml`.
JSON/NPZ are the boundary; the generation process never imports frankenmotion.
The scripts can be invoked from any working directory. Set these shell variables
to the actual checkout/asset locations:

```bash
OMNI_ROOT=/path/to/OmniControl-baseline
FRANKEN_ROOT=/path/to/frankenmotion
FRANKEN_PY=/path/to/frankenmotion/environment/bin/python
OMNI_PY=/path/to/omnicontrol/environment/bin/python
RUN_ROOT=/path/to/outputs/omnicontrol_command9
```

1. Export the frozen plan and static reference poses. Only the raw motion files
   for the nine selected real training episodes are needed, together with the
   skeleton matching the reach workspace. `--amass-root` can be repeated.

```bash
"$FRANKEN_PY" "$OMNI_ROOT/scripts/export_command_benchmark.py" \
  --frankenmotion-root "$FRANKEN_ROOT" \
  --skeleton /path/to/corpus_skeleton.npz \
  --amass-root /path/to/AMASS \
  --output "$RUN_ROOT/protocol.json"
```

   For inspecting commands before those assets are available, use `--plan-only`
   instead of `--skeleton/--amass-root`, and a separate `plan_only.json` output.
   This exports the actual upstream grids/prompts, selected reference episode IDs,
   source paths/hashes and metadata. **It is not accepted for generation.** No
   fabricated/reference-average pose is substituted. Different protocols cannot
   overwrite the same output path.

2. Inspect readiness, then generate. The official checkpoint's `args.json` is used
   when present; otherwise the repository model defaults apply. All effective
   arguments and asset hashes are stored in `run.json`.

```bash
"$OMNI_PY" "$OMNI_ROOT/scripts/generate_command_benchmark.py" \
  --protocol "$RUN_ROOT/protocol.json" \
  --checkpoint "$OMNI_ROOT/save/omnicontrol_ckpt/model_humanml3d.pt" \
  --preflight

# Smoke check: three evenly indexed commands per task, all three prompts/seeds.
"$OMNI_PY" "$OMNI_ROOT/scripts/generate_command_benchmark.py" \
  --protocol "$RUN_ROOT/protocol.json" \
  --checkpoint "$OMNI_ROOT/save/omnicontrol_ckpt/model_humanml3d.pt" \
  --output "$RUN_ROOT/generated" --device cuda:0 --batch-size 8 \
  --maximum-commands 3
```

   Inspect the smoke scores/gallery, then run the same command **without
   `--maximum-commands`** to complete all 23,400 samples. `--tasks walk reach`
   selects tasks for a bounded check; `--tasks reach --maximum-commands 128`
   reproduces the evenly indexed reach check subset. Neither option changes the
   protocol. Completed samples are reused when the full run resumes.

   Required native assets are the checkpoint, spatial `Mean_raw/Std_raw`,
   `dataset/HumanML3D/Mean.npy` and `Std.npy`, SMPL neutral model and extra joint
   regressor. Model construction also uses the native CLIP package/weights.
   Unlike the original demo, this entrypoint does not load test-set captions,
   motions, GloVe or the text-to-motion evaluator. It decodes XYZ using the exact
   motion statistics used by spatial guidance.

3. Score with the original task measurements and reach quality rules.

```bash
# Use --allow-partial only for the smoke check; it suppresses the official macro score.
"$FRANKEN_PY" "$OMNI_ROOT/scripts/score_command_benchmark.py" \
  --frankenmotion-root "$FRANKEN_ROOT" \
  --generated "$RUN_ROOT/generated" --output "$RUN_ROOT/smoke_reports" \
  --allow-partial

# Formal scoring requires every expected sample.
"$FRANKEN_PY" "$OMNI_ROOT/scripts/score_command_benchmark.py" \
  --frankenmotion-root "$FRANKEN_ROOT" \
  --generated "$RUN_ROOT/generated" --output "$RUN_ROOT/reports"
```

Scoring checks the exported upstream source hashes and recorded native generation
source hashes. Use the matching checkouts; do not update hashes to bypass changed
measurements. Partial scoring still requires all nine prompt/seed combinations for
every included command, and cannot silently drop failed generations.

## Frozen sampling protocol

Source release: `motion13_reach_v2_20261021`. The exporter verifies that the other
eight task manifests are byte-identical to `motion13_v1_20261020`. The old core
reach manifest has only 21 real training IDs and is incompatible with the current
68-ID reach workspace; it must not be substituted.

Eight scalar tasks use the real AMASS training-label min/max, ten equal-width
bins and three interior midpoint fractions per bin: `1/6, 1/2, 5/6`. Every bin is
included, even without training examples. Commands are exported directly from
upstream float32 plans; the displayed rounded numbers below are not used to
reconstruct them. `design_seed=20261020` is the upstream fallback design seed.

All measurements use 20 FPS, metres, and canonical X-forward/Y-left/Z-up joints.
For speed, duration is `(T-1)/20`, not `T/20`.

| Task | Measurement | Declared bounds | Actual sampled extent | Frames / hint joint |
|---|---|---|---|---|
| walk | Horizontal pelvis path length / duration | 0.48–1.88 m/s | 0.499513–1.617261 | 60 / pelvis 0 |
| back_walk | Negative pelvis forward displacement / duration | 0.49–1.39 m/s | 0.509076–1.363757 | 60 / pelvis 0 |
| sidestep | Maximum rightward pelvis displacement | 0.52–1.02 m | 0.553436–1.002351 | 30 / pelvis 0 |
| jog | Horizontal pelvis path length / duration | 1.16–3.64 m/s | 1.210642–3.593656 | 50 / pelvis 0 |
| reach | Final-five-frame mean right wrist XYZ relative to initial pelvis/heading | X −0.12–0.64, Y −0.69–0.34, Z −0.19–0.95 m | Frozen workspace below | 49 / right wrist 21 |
| jump | Maximum pelvis height minus first-frame height | 0.13–0.51 m | 0.137732–0.495891 | 66 / pelvis 0 |
| squat | Initial-five mean pelvis height minus minimum sliding-five mean | 0.31–0.64 m | 0.322497–0.628777 | 82 / pelvis 0 |
| strike | Event-window forward wrist excursion relative to shoulder | 0.20–0.47 m | 0.208999–0.461104 | 40 / right wrist 21 |
| kick | Maximum forward ankle-minus-pelvis coordinate minus its initial value | 0.46–1.14 m | 0.479534–1.117656 | 60 / right foot 11 |

Reach uses `reach_front_hemisphere_v2`, workspace SHA256
`8df745f08b937ecff8e9abb85ad6e79af3a544c0bd9a5b23613d12d86b0ad17d`.
The shoulder center is approximately `(0.025565,-0.176136,0.433760)` m, radius
0.20–0.51 m. The source stratifies radius cubed, forward-direction cosine and
azimuth into 10×10×10 cells, with three random interior points per cell and design
seed 20261021. Of 3,000 candidates, 2,292 reference-verified endpoints are admitted;
708 unverified points are excluded by the frozen source protocol. Another 68
observed real-training anchors are appended. Preserve order, groups and duplicates.
The 2,360 targets must not be resampled as a Cartesian box or filtered based on
OmniControl results. Endpoint certificates do not certify HumanML reachability or
complete trajectories.

Each command is evaluated with three exact upstream prompts and seeds
1928/1938/1948. Numeric commands are not inserted into the prompts. The exported
`tasks[].prompts` is the authoritative set of 27 strings; they are read directly
from `Task.prompts`. Unlike frankenmotion's global/part text conditioning,
OmniControl uses its native global CLIP text interface. `run.json` records full
token counts, actual 22-slot token IDs and whether the 20-text-token budget caused
truncation. Raw prompt strings are never shortened or paraphrased by the adapter.

Counts: eight tasks × 30 commands × 9 repetitions = 2,160; reach × 2,360 × 9 =
21,240; **23,400 total**. No best-seed selection or repeated resampling on failure.

## Analytic conditions and native inference

Static references come only from frozen real training data. Scalar tasks select
the episode nearest their median training command (lexical episode-ID tie-break).
Reach selects the workspace reference episode. The source hash and transforms are
verified, the crop is reconstructed using the upstream conversion, and only the
first FK pose is exported/used. Pose coordinates are canonical, with initial root
horizontal position zero. No trajectory from the reference is fed to the model.

The adapter constants live in `command_benchmark.protocol.ADAPTER`, and the full
dictionary is frozen into every protocol. Every real frame controls exactly one
joint's complete XYZ. All following coordinates are in the canonical frame:

- Walk/jog: `(v*t,0,h)`; back_walk: `(-v*t,0,h)`.
- Sidestep: `(0,-d*S(t/t_end),h)` with `S(u)=3u²−2u³`.
- Reach: first five frames at initial right wrist `W0`, smooth interpolation from
  frame 4 to `T−5`, last five frames at `(cx,cy,h+cz)`.
- Jump: `(0,0,h+max(0,d−0.5*9.81*(t−ta)²))`; apex at frame `floor((T−1)/2)`.
- Squat: `(0,0,h−d*P)`; first/last five frames standing. Smoothstep descent to
  `round(0.4*(T−1))`, plateau to `round(0.6*(T−1))`, smoothstep ascent to `T−5`.
- Strike: `W0+d*P*forward`; plateau at 45–55% of the timeline, with the same
  first/last-five-frame hold and smoothstep transitions.
- Kick: `F0+d*P*forward+0.35*P*up`; same timing as strike. The 0.35 m lift is a
  fixed adapter convention, not another sampled command.

HumanML coordinates satisfy `F=(H.z,H.x,H.y)`. Hints are converted to HumanML,
normalized with native spatial statistics, and then explicitly masked. Native
joint/sequence sum-based masks are checked for sentinel collisions; errors stop
the run instead of perturbing targets. Padding hints are zero.

Native DDPM, 1,000 diffusion steps, CFG 2.5 and the existing spatial/realism
guidance remain unchanged. Generation keeps the 196-frame tensor required by
native spatial guidance. Only the first T frames are conditioned and scored.
The native transformer does not enforce the length attention mask, so generated
tails can influence those frames. Native spatial gradients are zero at frame 0;
the initial hint is not a hard initial-pose constraint.

Each `(task,prompt,seed)` batch resets the seed before the native
`p_sample_loop(const_noise=True)` call. Both initial and per-step noise are shared
across command points. Resuming/repartitioning a command grid preserves that
stream. Different models do not have identical noise tensors merely because their
seed numbers match. Full-model batch equivalence is subject to floating-point
tolerance; the native noise sequence is tested directly.

## Metrics, artifacts and interpretation

Scoring converts raw HumanML XYZ back to the canonical frame and rotates by the
initial hip heading. It does not retarget, scale the body, fit SMPL, or repair the
motion. `Task.measure_positions` and `Task.diagnostics` are reused directly.

Per-task outputs include physical MAE, per-command mean/population std/MAE and all
nine prompt/seed response curves. Normalized MAE divides per-sample absolute error
by the **declared task bound span**, then averages. Full-run macro normalized MAE
weights all nine tasks equally; it is not dominated by reach sample count.

Reach additionally reports positional RMSE, bias RMS, std RMS, 5/10 cm successes,
quality pass rate and joint position+quality success, grouped into all targets,
verified workspace, observed anchors, ≤5 cm real-supported and >5 cm real-gap
targets. It uses the upstream quality thresholds without redefining them.

Spatial hint L2 error and the repository's native foot-skating ratio are separate
auxiliary measurements on the real frames. They are not substituted for command
error. In particular, strike is wrist-relative-to-shoulder; reach is relative to
the generated initial pelvis; kick is **ankle 8** relative to pelvis even though
the native control is **foot 11**. Body or foot orientation drift remains an error.
Scalar speed/amplitude metrics alone do not certify walking direction, gait,
two-foot takeoff, planted feet or retraction.

Generation layout:

```text
generated/
  protocol.json
  run.json
  samples/<task>/<task>__c0000__p0__s1928.npz
```

Each NPZ contains `positions_h[196,22,3]`, `hint_h[196,22,3]`, explicit
`hint_mask[196,22]`, real length, requested command, sample/run IDs, FPS, prompt
index and seed. Writes are atomic and completed samples are validated before
reuse. Changing checkpoint, protocol, model configuration, generation code,
normalization or runtime identity requires a separate output directory. Batch size
and selected subset are not run identities, so smoke samples can be reused.
Use one writer per output directory; these scripts do not coordinate concurrent
workers writing the same run.

Reports contain JSON/CSV summaries, per-sample CSV, per-task measured NPZ panels,
response PNGs, and standalone HTML motion galleries. The gallery includes all
prompt/seed combinations for fixed minimum/middle/maximum scalar commands; reach
uses shoulder-radius ordering. No generated result affects this selection. A
partial report is prominently marked and has no official macro score.

## Validation

For local CPU tests without CLIP/checkpoints/SMPL assets:

```bash
cd "$OMNI_ROOT"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-benchmark-test.txt
FRANKENMOTION_ROOT="$FRANKEN_ROOT" .venv/bin/python -m unittest discover -s tests -v
```

NumPy is pinned below 1.24 for unchanged upstream `np.float` usage. This test
environment is separate from the original generation environment. Tests cover
real exported grids/seeds/prompts, nine synthetic physical measurements, coordinate
conversion, padding exclusion, explicit masks and normalization, ankle/foot
distinction, per-sample MAE, reach group/success denominators, native DDPM random
stream pairing, atomic resume and report generation. Upstream integration tests
skip explicitly when no frankenmotion checkout is supplied.

These tests do not demonstrate pretrained model quality. The full-model GPU smoke
check and complete 23,400-sample run require the real model and reference assets.

Local validation on 2026-10-21: all 13 CPU tests passed against the available
frankenmotion checkout, with no skipped tests. The real plan exported 23,400 sample
identities. New runtime files passed Python 3.7 syntax parsing, Black checks and
`git diff --check`; the gallery JavaScript passed `node --check`. The response
figure was visually inspected using synthetic test data. Native checkpoint/GPU
inference was not run: checkpoint, HumanML motion statistics, SMPL assets and
production reference-pose assets were unavailable. The exported local plan is
therefore explicitly `references_ready=false`.

## Background execution on betail

The 2026-10-21 deployment uses `/home/psirobot/projects/OmniControl-baseline`
directly. Data symlinks are preserved. The small matching frankenmotion source and
annotation snapshot is under `benchmark_sources/frankenmotion`; original AMASS
and skeleton files are read from their existing locations. The exported production
protocol is `outputs/command9_20261021/protocol.json` with references ready and no
missing native assets. All 13 tests also passed on betail before launch.

`scripts/run_command_benchmark_pipeline.py` supervises four stages: smoke generation,
smoke scoring, full generation, final scoring. The deployment uses three evenly
indexed commands per task for the smoke stage (243 samples), batch size 32 for the
full run, and the native `omnicontrol` conda environment for generation. Export
and scoring use Python 3.11 with dependencies isolated in the output directory;
the original conda environments are not modified. Failure stops the pipeline,
and restarting the same command resumes already committed native samples.

Inspect the live phase and generation log:

```bash
ssh betail 'cat ~/projects/OmniControl-baseline/outputs/command9_20261021/pipeline_status.json'
ssh betail 'tail -20 ~/projects/OmniControl-baseline/outputs/command9_20261021/smoke_generate.log'
ssh betail 'tail -20 ~/projects/OmniControl-baseline/outputs/command9_20261021/full_generate.log'
```

Exact stage commands are saved in `pipeline_commands.json`, phase logs in
`smoke_generate.log`, `smoke_score.log`, `full_generate.log` and `full_score.log`.
The pipeline holds an exclusive lock to prevent concurrent supervisors. Final
results are only complete when `pipeline_status.json` reports `phase=complete`;
the final report is `reports/results.json`. A running process or a smoke report
must not be reported as a completed full benchmark.

## Fixed-target strike/kick revision (2026-10-22)

The original 23,400-sample run completed and is preserved under
`outputs/command9_20261021`; its generation sources were verified against
`generated/run.json` and archived on betail as `runtime_source_snapshot_v1.tar.gz`.
Do not overwrite its protocols, reports or sample archives.

`scripts/revise_limb_protocol.py` derives a separate 540-sample protocol under
`outputs/command_limb_fixed_target_20261022`. Commands, lengths, prompts, seeds,
measurement definitions and reference poses are copied exactly. Only the hint
adapter changes, and both parent identity and protocol hash are recorded.

The `limb_fixed_target_v2` adapter controls one joint at a constant world-space
position in the centered 20-frame interval: strike frames `[10,30)`, kick
`[20,40)`. All other frames, including padding, have zero hints and false masks.
The approach and recovery are left to the model; root is not constrained.
Coordinates below use the forward/left/up scoring convention:

* Strike wrist 21: reference right shoulder 17 + `(command, 0, 0)`. This replaces
  the already extended reference wrist as the command origin. It represents
  extension from the shoulder plane; the generated initial wrist position is
  unconstrained, so it remains a proxy for the scored excursion.
* Kick foot 11: `(reference_foot_X + command, reference_foot_Y, 1.0)`. The absolute
  1 m target height and constant temporal target follow the upstream manual kick
  example. The command remains the original relative ankle excursion for scoring.

All strike targets satisfy the reference arm-length distance bound. Twelve of
30 kick targets exceed the distance bound from the static reference hip; these
are retained and flagged, not clipped or removed. This bound does not establish
pose or motion feasibility. Body translation and initial-pose mismatch remain
possible and must be reported when interpreting the rerun.

The supervisor now reads expected counts from the protocol and supports
`--stop-after-smoke` for inspecting 54 smoke samples before resuming full
generation. Native weights, DDPM settings, CFG and noise pairing remain unchanged.
The two-task macro score must not be presented as the nine-task aggregate.

## Initial-root-frame measurement revision (2026-10-23)

`scripts/rescore_limb_metrics.py` remeasures the existing 540 fixed-target motions
with the current upstream strike/kick functions. Kick retains pelvis translation;
strike retains its arm-relative event detector but measures wrist travel in the
fixed initial-root frame. Their current normalization spans are 1.00 m and 1.30 m.
The generation protocol, commands, hints, seeds and sample identities stay frozen.

```bash
.venv/bin/python scripts/rescore_limb_metrics.py \
  --frankenmotion-root "$FRANKEN_ROOT" \
  --generated outputs/command_limb_fixed_target_20261022/generated \
  --previous-reports outputs/command_limb_fixed_target_20261022/reports \
  --original-nine-reports outputs/command9_20261021/reports \
  --output outputs/command_limb_initial_frame_eval_20261023
```

Use a new output directory for each scoring revision. It contains a snapshot of
the imported task sources, `evaluation.json`, input hashes, per-sample scores and
combined nine-task summaries. Generation metadata remains unchanged and separate
from evaluation definitions. The current rescore gives strike MAE 0.348538 m and
kick MAE 0.209865 m; the combined nine-task task-equal normalized MAE is 0.125679.
With the previous normalization spans, the same remeasurements give 0.246738.
These are changes in scoring, not new generated model performance.

The other seven tasks reuse their original scores and normalization, including
the original reach workspace. Existing commands are not replaced with the newer
training-label grid. This is therefore not an all-task run of the newest upstream
evaluation protocol. Previous reports and motions remain intact.

Plot the combined responses with:

```bash
.venv/bin/python scripts/plot_command_response.py \
  --reports outputs/command9_20261021/reports \
  --limb-reports outputs/command_limb_initial_frame_eval_20261023/reports \
  --output outputs/command_limb_initial_frame_eval_20261023/response_curves
```

The original regression suite must use the original frozen upstream source and
annotation snapshot, e.g. `FRANKENMOTION_ROOT=benchmark_sources/frankenmotion`.
The separate rescore audit verifies new measurements directly against the new
upstream functions and checks that padding does not affect the result.
