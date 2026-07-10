"""
data_ingestion.py
 
Pulls raw NFL / fantasy football data from nflverse (via nfl_data_py) and
saves it locally to data/raw/ as parquet files.
 
This script is idempotent: re-running it just re-downloads and overwrites
the raw files, which is fine since raw data is never edited by hand.
 
Usage:
    python src/data_ingestion.py
    python src/data_ingestion.py --start-year 2018 --end-year 2025
"""
 
import argparse
import os
 
import nfl_data_py as nfl
import pandas as pd
 
# Project root is one level up from this file (src/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
 
 
def ensure_raw_dir_exists() -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
 
 
def sanitize_mixed_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    nfl_data_py's raw tables occasionally have 'object' columns with mixed
    types inside them (e.g. jersey_number containing both floats and NaN,
    or strings and floats mixed together). Parquet writers require a single
    consistent type per column, so this coerces each object column to
    either numeric (if it's mostly numeric-like) or string (otherwise).
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        numeric_version = pd.to_numeric(df[col], errors="coerce")
        # If coercing to numeric doesn't lose much info, treat it as numeric.
        if numeric_version.notna().sum() >= non_null.shape[0] * 0.9:
            df[col] = numeric_version
        else:
            df[col] = df[col].astype(str).where(df[col].notna(), None)
    return df
 
 
def save_raw(df: pd.DataFrame, name: str) -> None:
    """Save a dataframe to data/raw/<name>.parquet and print a quick summary."""
    path = os.path.join(RAW_DATA_DIR, f"{name}.parquet")
    df = sanitize_mixed_type_columns(df)
    # Explicitly use pyarrow: fastparquet (the other engine pandas may pick
    # automatically) can choke on columns with mixed/nullable types, e.g.
    # a numeric column that also contains missing values.
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"  Saved {name}: {df.shape[0]:,} rows x {df.shape[1]} cols -> {path}")
 
 
def pull_weekly_data(years: list[int]) -> pd.DataFrame:
    """
    Weekly player stats, including fantasy points (standard, half-PPR, PPR).
    This is the core table for building week-by-week fantasy projections.
    """
    print("Pulling weekly player data...")
    df = nfl.import_weekly_data(years)
    save_raw(df, "weekly_player_stats")
    return df
 
 
def pull_seasonal_data(years: list[int]) -> pd.DataFrame:
    """
    Season-aggregated player stats. Useful for season-long draft projections
    (as opposed to week-to-week lineup decisions).
    """
    print("Pulling seasonal player data...")
    df = nfl.import_seasonal_data(years)
    save_raw(df, "seasonal_player_stats")
    return df
 
 
def pull_rosters(years: list[int]) -> pd.DataFrame:
    """
    Roster info per season: player position, team, age, status, etc.
    Needed to join stats tables with position/team context.
    """
    print("Pulling seasonal rosters...")
    df = nfl.import_seasonal_rosters(years)
    save_raw(df, "seasonal_rosters")
    return df
 
 
def pull_schedules(years: list[int]) -> pd.DataFrame:
    """
    Game schedules, including bye weeks and opponent info.
    Needed later for strength-of-schedule and bye-week-aware roster construction.
    """
    print("Pulling schedules...")
    df = nfl.import_schedules(years)
    save_raw(df, "schedules")
    return df
 
 
def pull_injuries(years: list[int]) -> pd.DataFrame:
    """
    Weekly injury report data. Useful for the injury-risk features later on.
    """
    print("Pulling injury reports...")
    df = nfl.import_injuries(years)
    save_raw(df, "injuries")
    return df
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Pull raw nflverse data for the fantasy draft bot project.")
    parser.add_argument("--start-year", type=int, default=2015, help="First season to pull (inclusive).")
    parser.add_argument("--end-year", type=int, default=2024, help="Last season to pull (inclusive).")
    args = parser.parse_args()
 
    years = list(range(args.start_year, args.end_year + 1))
    print(f"Pulling data for seasons: {years[0]}-{years[-1]}\n")
 
    ensure_raw_dir_exists()
 
    pull_weekly_data(years)
    pull_seasonal_data(years)
    pull_rosters(years)
    pull_schedules(years)
    pull_injuries(years)
 
    print("\nDone. Raw data saved to data/raw/")
 
 
if __name__ == "__main__":
    main()