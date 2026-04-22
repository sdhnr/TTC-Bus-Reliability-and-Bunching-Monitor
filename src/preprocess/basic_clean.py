"""Minimal preprocess helpers for milestone-1 notebook use."""

import pandas as pd


def parse_timestamp_column(df: pd.DataFrame, column: str = "timestamp_utc") -> pd.DataFrame:
    """Parse a timestamp column to pandas datetime (UTC)."""
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    return df
