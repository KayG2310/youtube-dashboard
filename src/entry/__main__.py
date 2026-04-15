import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PipelineStep:
    name: str
    command: List[str]


def run_step(step: PipelineStep, continue_on_error: bool) -> bool:
    print(f"\n{'=' * 80}")
    print(f"STEP: {step.name}")
    print(f"COMMAND: {' '.join(step.command)}")
    print(f"{'=' * 80}")
    result = subprocess.run(step.command, check=False)
    if result.returncode == 0:
        print(f"[OK] {step.name}")
        return True

    print(f"[FAILED] {step.name} (exit code: {result.returncode})")
    if continue_on_error:
        print("Continuing because --continue-on-error is enabled.")
        return False

    raise SystemExit(result.returncode)


def build_pipeline_steps(project_root: Path) -> List[PipelineStep]:
    py = sys.executable
    return [
        PipelineStep(
            name="Data collection (trending/search/comments + thumbnails)",
            command=[py, str(project_root / "src" / "DataCollection" / "CollectorScript.py")],
        ),
        PipelineStep(
            name="Comment data processing",
            command=[py, str(project_root / "src" / "DataProcessing" / "CommentProcessor.py")],
        ),
        PipelineStep(
            name="Trending analytics processing",
            command=[py, str(project_root / "src" / "DataProcessing" / "trending_processing.py")],
        ),
        PipelineStep(
            name="Thumbnail analytics processing",
            command=[py, str(project_root / "src" / "DataProcessing" / "Thumbnail_processing.py")],
        ),
        PipelineStep(
            name="Publish search records to Kafka",
            command=[py, str(project_root / "src" / "DataCollection" / "SearchKafkaProducer.py")],
        ),
        PipelineStep(
            name="Search Delta Bronze/Silver processing",
            command=[py, str(project_root / "src" / "DataProcessing" / "SearchDataProcessorDelta.py")],
        ),
        PipelineStep(
            name="Search Delta Gold analysis",
            command=[py, str(project_root / "src" / "DataAnalysis" / "searchAnalysis" / "search_analysis_delta.py")],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end YouTube dashboard pipeline from one entrypoint.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run full pipeline but do not launch Streamlit dashboard.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps even if a step fails.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root: {project_root}")

    steps = build_pipeline_steps(project_root)
    completed = 0
    failed = 0

    for step in steps:
        ok = run_step(step, continue_on_error=args.continue_on_error)
        if ok:
            completed += 1
        else:
            failed += 1

    print(f"\nPipeline finished. Successful steps: {completed}, failed steps: {failed}")

    if args.no_dashboard:
        print("Dashboard launch skipped (--no-dashboard).")
        return

    dashboard_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(project_root / "src" / "Dashboard" / "app.py"),
    ]
    print(f"\nLaunching dashboard: {' '.join(dashboard_cmd)}")
    subprocess.run(dashboard_cmd, check=False)


if __name__ == "__main__":
    main()