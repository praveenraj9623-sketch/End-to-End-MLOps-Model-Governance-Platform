"""Tests for data validation."""

from pathlib import Path
import pandas as pd


def test_raw_dataset_exists():
    assert Path("data/raw/hr_attrition.csv").exists()


def test_raw_dataset_has_target_column():
    df = pd.read_csv("data/raw/hr_attrition.csv")
    assert "Attrition" in df.columns
