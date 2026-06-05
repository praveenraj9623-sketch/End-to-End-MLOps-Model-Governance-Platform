"""Reusable data validation checks for the HR attrition dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.ingestion import EXPECTED_CATEGORIES, EXPECTED_COLUMNS, RAW_DATA_PATH, VALUE_RANGES

REPORT_PATH = Path("reports/performance_logs/validation_report.json")


class DataValidator:
    """Validate schema, missingness, ranges, and categorical cardinality."""

    def __init__(self, report_path: Path = REPORT_PATH) -> None:
        self.report_path = report_path
        self.results: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "summary": {},
        }

    def check_schema(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate that expected columns are present and the target is well formed."""
        missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
        extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
        target_values = (
            sorted(df["Attrition"].dropna().astype(str).unique().tolist())
            if "Attrition" in df.columns
            else []
        )
        valid_target = set(target_values).issubset({"Yes", "No"}) and bool(target_values)

        result = {
            "status": "pass" if not missing_columns and valid_target else "fail",
            "expected_column_count": len(EXPECTED_COLUMNS),
            "observed_column_count": int(df.shape[1]),
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "target_values": target_values,
            "valid_target": valid_target,
        }
        self.results["checks"]["schema"] = result
        return result

    def check_missing_values(
        self, df: pd.DataFrame, threshold: float = 0.2
    ) -> dict[str, Any]:
        """Flag columns whose missing ratio exceeds the configured threshold."""
        missing_percentages = df.isna().mean().round(4).to_dict()
        failing_columns = [
            column for column, value in missing_percentages.items() if value > threshold
        ]
        result = {
            "status": "pass" if not failing_columns else "warn",
            "threshold": threshold,
            "missing_percentages": missing_percentages,
            "columns_above_threshold": failing_columns,
        }
        self.results["checks"]["missing_values"] = result
        return result

    def check_value_ranges(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate known numeric value ranges for the IBM HR dataset."""
        columns: dict[str, Any] = {}
        failing_columns: list[str] = []

        for column, (expected_min, expected_max) in VALUE_RANGES.items():
            if column not in df.columns:
                continue
            observed_min = float(df[column].min())
            observed_max = float(df[column].max())
            passed = observed_min >= expected_min and observed_max <= expected_max
            if not passed:
                failing_columns.append(column)
            columns[column] = {
                "expected_min": expected_min,
                "expected_max": expected_max,
                "observed_min": observed_min,
                "observed_max": observed_max,
                "status": "pass" if passed else "warn",
            }

        result = {
            "status": "pass" if not failing_columns else "warn",
            "columns": columns,
            "columns_outside_expected_range": failing_columns,
        }
        self.results["checks"]["value_ranges"] = result
        return result

    def check_cardinality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check cardinality and unexpected values for categorical columns."""
        columns: dict[str, Any] = {}
        failing_columns: list[str] = []

        categorical_columns = df.select_dtypes(include=["object", "category"]).columns
        for column in categorical_columns:
            observed_values = sorted(df[column].dropna().astype(str).unique().tolist())
            unexpected_values: list[str] = []
            if column in EXPECTED_CATEGORIES:
                unexpected_values = sorted(
                    set(observed_values) - EXPECTED_CATEGORIES[column]
                )
            if unexpected_values:
                failing_columns.append(column)
            columns[column] = {
                "unique_count": int(df[column].nunique(dropna=True)),
                "observed_values": observed_values,
                "unexpected_values": unexpected_values,
                "status": "pass" if not unexpected_values else "warn",
            }

        result = {
            "status": "pass" if not failing_columns else "warn",
            "columns": columns,
            "columns_with_unexpected_categories": failing_columns,
        }
        self.results["checks"]["cardinality"] = result
        return result

    def generate_validation_report(self) -> dict[str, Any]:
        """Finalize and save the validation report."""
        check_statuses = [
            check["status"] for check in self.results["checks"].values() if "status" in check
        ]
        self.results["summary"] = {
            "overall_status": "fail"
            if "fail" in check_statuses
            else "warn"
            if "warn" in check_statuses
            else "pass",
            "check_count": len(check_statuses),
            "failed_checks": int(check_statuses.count("fail")),
            "warning_checks": int(check_statuses.count("warn")),
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        return self.results


def main() -> dict[str, Any]:
    """Run validation checks against the raw dataset and save a JSON report."""
    df = pd.read_csv(RAW_DATA_PATH)
    validator = DataValidator()
    validator.check_schema(df)
    validator.check_missing_values(df)
    validator.check_value_ranges(df)
    validator.check_cardinality(df)
    report = validator.generate_validation_report()
    print(f"Validation report saved to {REPORT_PATH}")
    print(report["summary"])
    return report


if __name__ == "__main__":
    main()
