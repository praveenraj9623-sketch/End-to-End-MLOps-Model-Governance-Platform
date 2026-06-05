"""Shared retraining orchestration for FastAPI, dashboard, and local demo."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RETRAIN_REPORT_PATH = Path("reports/performance_logs/manual_retrain_report.json")
PIPELINE_MODULES = [
    "src.data.validation",
    "src.features.engineering",
    "src.models.train",
    "src.models.evaluate",
    "src.monitoring.drift_detection",
    "src.governance.fairness",
    "src.models.register",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_python_module(module: str, timeout_seconds: int = 420) -> dict[str, Any]:
    """Run one Python module and return a compact, serializable result."""
    started = _now()
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "module": module,
        "started_at_utc": started,
        "finished_at_utc": _now(),
        "status": "success" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def run_local_retrain_pipeline() -> dict[str, Any]:
    """Run the same local retraining sequence used by setup_demo."""
    report = {
        "started_at_utc": _now(),
        "mode": "local_python_modules",
        "steps": [],
        "status": "success",
    }
    for module in PIPELINE_MODULES:
        step = run_python_module(module)
        report["steps"].append(step)
        if step["status"] != "success":
            report["status"] = "failed"
            break
    report["finished_at_utc"] = _now()
    RETRAIN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def trigger_retrain() -> dict[str, Any]:
    """Trigger Airflow when installed; otherwise run the local deterministic pipeline."""
    airflow = shutil.which("airflow")
    if airflow:
        command = [airflow, "dags", "trigger", "mlops_weekly_retraining"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            report = {
                "started_at_utc": _now(),
                "finished_at_utc": _now(),
                "mode": "airflow_cli",
                "dag_id": "mlops_weekly_retraining",
                "status": "triggered" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
            RETRAIN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            RETRAIN_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        except subprocess.TimeoutExpired as exc:
            return {
                "started_at_utc": _now(),
                "finished_at_utc": _now(),
                "mode": "airflow_cli",
                "dag_id": "mlops_weekly_retraining",
                "status": "timeout",
                "stdout": (exc.stdout or "")[-4000:],
                "stderr": (exc.stderr or "")[-4000:],
            }
    return run_local_retrain_pipeline()

