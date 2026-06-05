"""Data ingestion and quality checks for the HR attrition dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

RAW_DATA_PATH = Path("data/raw/hr_attrition.csv")
QUALITY_LOG_PATH = Path("reports/performance_logs/data_quality_log.csv")
SUMMARY_REPORT_PATH = Path("reports/performance_logs/data_quality_summary.json")

EXPECTED_COLUMNS = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CRITICAL_COLUMNS = [
    "Attrition",
    "Age",
    "JobRole",
    "MonthlyIncome",
    "OverTime",
    "YearsAtCompany",
]

EXPECTED_CATEGORIES = {
    "Attrition": {"Yes", "No"},
    "BusinessTravel": {"Non-Travel", "Travel_Rarely", "Travel_Frequently"},
    "Department": {"Human Resources", "Research & Development", "Sales"},
    "EducationField": {
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    },
    "Gender": {"Female", "Male"},
    "JobRole": {
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    },
    "MaritalStatus": {"Divorced", "Married", "Single"},
    "Over18": {"Y"},
    "OverTime": {"Yes", "No"},
}

VALUE_RANGES = {
    "Age": (18, 65),
    "DailyRate": (100, 1600),
    "DistanceFromHome": (1, 30),
    "Education": (1, 5),
    "EmployeeCount": (1, 1),
    "EnvironmentSatisfaction": (1, 4),
    "HourlyRate": (30, 100),
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "MonthlyIncome": (1000, 25000),
    "MonthlyRate": (2000, 27000),
    "NumCompaniesWorked": (0, 10),
    "PercentSalaryHike": (10, 25),
    "PerformanceRating": (3, 4),
    "RelationshipSatisfaction": (1, 4),
    "StandardHours": (80, 80),
    "StockOptionLevel": (0, 3),
    "TotalWorkingYears": (0, 45),
    "TrainingTimesLastYear": (0, 6),
    "WorkLifeBalance": (1, 4),
    "YearsAtCompany": (0, 40),
    "YearsInCurrentRole": (0, 18),
    "YearsSinceLastPromotion": (0, 15),
    "YearsWithCurrManager": (0, 17),
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw IBM HR attrition data."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path)


def _quality_row(
    *,
    check_name: str,
    column: str,
    status: str,
    details: str,
    observed_value: Any = "",
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "check_name": check_name,
        "column": column,
        "status": status,
        "observed_value": observed_value,
        "details": details,
    }


def run_quality_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run missingness, schema, range, and category quality checks."""
    rows: list[dict[str, Any]] = []

    missing_critical = [column for column in CRITICAL_COLUMNS if column not in df.columns]
    for column in CRITICAL_COLUMNS:
        status = "pass" if column in df.columns else "fail"
        rows.append(
            _quality_row(
                check_name="critical_column_present",
                column=column,
                status=status,
                observed_value=column in df.columns,
                details="Required for model training and governance workflows.",
            )
        )

    for column in EXPECTED_COLUMNS:
        rows.append(
            _quality_row(
                check_name="expected_column_present",
                column=column,
                status="pass" if column in df.columns else "warn",
                observed_value=column in df.columns,
                details="Expected IBM HR attrition dataset column.",
            )
        )

    for column in df.columns:
        null_pct = float(df[column].isna().mean())
        rows.append(
            _quality_row(
                check_name="null_percentage",
                column=column,
                status="pass" if null_pct <= 0.2 else "warn",
                observed_value=round(null_pct, 4),
                details="Share of missing values in the column.",
            )
        )

    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        observed_min = df[column].min()
        observed_max = df[column].max()
        if column in VALUE_RANGES:
            expected_min, expected_max = VALUE_RANGES[column]
            in_range = observed_min >= expected_min and observed_max <= expected_max
            status = "pass" if in_range else "warn"
            details = f"Expected range [{expected_min}, {expected_max}]."
        else:
            status = "pass"
            details = "No fixed expected range configured; min/max logged for monitoring."
        rows.append(
            _quality_row(
                check_name="numeric_value_range",
                column=column,
                status=status,
                observed_value=f"min={observed_min}, max={observed_max}",
                details=details,
            )
        )

    for column, allowed_values in EXPECTED_CATEGORIES.items():
        if column not in df.columns:
            continue
        observed_values = set(df[column].dropna().astype(str).unique())
        unexpected = sorted(observed_values - allowed_values)
        rows.append(
            _quality_row(
                check_name="unexpected_category_values",
                column=column,
                status="pass" if not unexpected else "warn",
                observed_value="; ".join(unexpected),
                details=f"Allowed values: {sorted(allowed_values)}",
            )
        )

    duplicate_count = int(df.duplicated().sum())
    rows.append(
        _quality_row(
            check_name="duplicate_rows",
            column="__dataset__",
            status="pass" if duplicate_count == 0 else "warn",
            observed_value=duplicate_count,
            details="Number of fully duplicated records.",
        )
    )

    quality_log = pd.DataFrame(rows)
    summary = {
        "dataset_path": str(RAW_DATA_PATH),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "critical_columns_missing": missing_critical,
        "checks_total": int(len(quality_log)),
        "checks_failed": int((quality_log["status"] == "fail").sum()),
        "checks_warned": int((quality_log["status"] == "warn").sum()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return quality_log, summary


def save_quality_outputs(
    quality_log: pd.DataFrame,
    summary: dict[str, Any],
    log_path: Path = QUALITY_LOG_PATH,
    summary_path: Path = SUMMARY_REPORT_PATH,
) -> None:
    """Persist the quality log and summary report."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    quality_log.to_csv(log_path, index=False)
    pd.Series(summary).to_json(summary_path, indent=2)


def main() -> dict[str, Any]:
    """Load raw data, run quality checks, persist reports, and return a summary."""
    df = load_raw_data()
    missing_critical = [column for column in CRITICAL_COLUMNS if column not in df.columns]
    if missing_critical:
        raise ValueError(f"Missing critical columns: {missing_critical}")

    quality_log, summary = run_quality_checks(df)
    save_quality_outputs(quality_log, summary)
    print(f"Data quality log saved to {QUALITY_LOG_PATH}")
    print(f"Data quality summary saved to {SUMMARY_REPORT_PATH}")
    print(summary)
    return summary


if __name__ == "__main__":
    main()
