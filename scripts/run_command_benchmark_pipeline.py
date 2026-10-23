#!/usr/bin/env python3
"""Supervise smoke generation, scoring, full generation and final scoring.

Launch with nohup on the GPU host. Child failures stop the pipeline, with phase,
PID, command and exit status saved alongside separate logs. Existing native
sample files are reused by the normal generation entrypoint on restart.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_benchmark.protocol import REPOSITORY, load_protocol, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-python", type=Path, required=True)
    parser.add_argument("--scoring-python", type=Path, required=True)
    parser.add_argument("--scoring-pythonpath", default="")
    parser.add_argument("--frankenmotion-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke-commands", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after-smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.smoke_commands < 1:
        parser.error("Batch size and smoke commands must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    expected_samples = load_protocol(args.protocol)["expected_samples"]

    def status(phase, **details):
        write_json(
            output / "pipeline_status.json",
            dict(
                phase=phase,
                supervisor_pid=os.getpid(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.time() - started,
                **details,
            ),
        )

    generation_environment = os.environ.copy()
    generation_environment.pop("PYTHONPATH", None)
    generation_environment["OMP_NUM_THREADS"] = "4"
    scoring_environment = generation_environment.copy()
    if args.scoring_pythonpath:
        scoring_environment["PYTHONPATH"] = args.scoring_pythonpath
    generate = [
        str(args.generation_python.resolve()),
        "-u",
        str(REPOSITORY / "scripts/generate_command_benchmark.py"),
        "--protocol",
        str(args.protocol.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--output",
        str(output / "generated"),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    score = [
        str(args.scoring_python.resolve()),
        "-u",
        str(REPOSITORY / "scripts/score_command_benchmark.py"),
        "--frankenmotion-root",
        str(args.frankenmotion_root.resolve()),
        "--generated",
        str(output / "generated"),
    ]
    stages = [
        (
            "smoke_generate",
            generate + ["--maximum-commands", str(args.smoke_commands)],
            generation_environment,
        ),
        (
            "smoke_score",
            score + ["--allow-partial", "--output", str(output / "smoke_reports")],
            scoring_environment,
        ),
        ("full_generate", generate, generation_environment),
        (
            "full_score",
            score + ["--output", str(output / "reports")],
            scoring_environment,
        ),
    ]
    write_json(
        output / "pipeline_commands.json",
        {
            "stages": [
                {"phase": phase, "command": command} for phase, command, _ in stages
            ],
            "scoring_pythonpath": args.scoring_pythonpath,
            "omp_num_threads": 4,
        },
    )
    for phase, command, environment in stages:
        log_path = output / (phase + ".log")
        print("Starting " + phase, flush=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=str(REPOSITORY),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            status(phase, child_pid=process.pid, command=command, log=str(log_path))
            while process.poll() is None:
                time.sleep(30)
                count = sum(1 for _ in (output / "generated/samples").glob("*/*.npz"))
                status(
                    phase,
                    child_pid=process.pid,
                    command=command,
                    log=str(log_path),
                    completed_samples=count,
                    expected_samples=expected_samples,
                )
            if process.returncode:
                status(
                    "failed",
                    failed_phase=phase,
                    returncode=process.returncode,
                    log=str(log_path),
                )
                raise SystemExit(process.returncode)
        if phase == "smoke_score" and args.stop_after_smoke:
            status("smoke_complete", report=str(output / "smoke_reports/results.json"))
            return
    report = read_json(output / "reports/results.json")
    if not report["complete"] or report["scored_samples"] != expected_samples:
        status("failed", failed_phase="completion_validation")
        raise SystemExit("Final report incomplete")
    status(
        "complete",
        completed_samples=expected_samples,
        expected_samples=expected_samples,
        report=str(output / "reports/results.json"),
    )
    print("Full benchmark and scoring complete", flush=True)


if __name__ == "__main__":
    main()
