"""
adp_ingestion.py

Parses the manually-downloaded FantasyPros ADP CSVs (one per season, saved
as data/raw/adp/adp_<year>.csv) into a single clean, combined table saved to
data/processed/adp_combined.parquet.

FantasyPros' raw export has two messy combined columns we need to split:
  - "Player (Bye)"  ->  player_name, team, bye_week
  - "POS"           ->  position, position_rank

Different years may have slightly different sets of per-platform columns
(e.g. ESPN, Sleeper, CBS, NFL, RTSports, Fantrax) since not every platform
has existed / been tracked the whole time. This script handles that by only
keeping the columns that are common bookkeeping (Rank, AVG) plus whatever
platform columns happen to exist in a given year's file -- it does not
assume a fixed column list.

Usage:
    python src/adp_ingestion.py
"""

import glob
import os
import re

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADP_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "adp")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Matches things like "Christian McCaffrey   SF (9)" -> name / team / bye
# Also tolerates missing bye week, e.g. "Christian McCaffrey   SF"
PLAYER_BYE_PATTERN = re.compile(
    r"^(?P<name>.+?)\s+(?P<team>[A-Z]{2,3})(?:\s*\((?P<bye>\d{1,2})\))?$"
)

# Matches things like "RB1" -> position / position_rank
POS_PATTERN = re.compile(r"^(?P<position>[A-Z]+)(?P<position_rank>\d+)$")


def parse_player_bye(raw: str) -> tuple[str, str, float]:
    """Split 'Player Name   TEAM (bye)' into (name, team, bye_week)."""
    if not isinstance(raw, str):
        return (raw, None, None)

    match = PLAYER_BYE_PATTERN.match(raw.strip())
    if match:
        bye = match.group("bye")
        return (
            match.group("name").strip(),
            match.group("team"),
            int(bye) if bye else None,
        )
    # Didn't match the expected pattern -- return the raw string as the name
    # so nothing is silently dropped, and leave team/bye as missing.
    return (raw.strip(), None, None)


def parse_position(raw: str) -> tuple[str, float]:
    """Split 'RB1' into ('RB', 1)."""
    if not isinstance(raw, str):
        return (raw, None)

    match = POS_PATTERN.match(raw.strip())
    if match:
        return (match.group("position"), int(match.group("position_rank")))
    return (raw.strip(), None)


def load_and_clean_one_year(filepath: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Identify the "Player (Bye)" column -- FantasyPros has used slightly
    # different exact header text across years, so match loosely.
    player_col = next((c for c in df.columns if "Player" in c), None)
    if player_col is None:
        raise ValueError(f"{filepath}: couldn't find a 'Player' column. Columns were: {list(df.columns)}")

    parsed_player = df[player_col].apply(parse_player_bye)
    df["player_name"] = [p[0] for p in parsed_player]
    df["team"] = [p[1] for p in parsed_player]
    df["bye_week"] = [p[2] for p in parsed_player]

    parsed_pos = df["POS"].apply(parse_position)
    df["position"] = [p[0] for p in parsed_pos]
    df["position_rank"] = [p[1] for p in parsed_pos]

    df["season"] = year

    # Keep: identifying/derived columns + Rank/AVG + whatever per-platform
    # columns exist in this particular year's file.
    known_non_platform_cols = {player_col, "POS", "Rank", "AVG", "player_name", "team", "bye_week", "position", "position_rank", "season"}
    platform_cols = [c for c in df.columns if c not in known_non_platform_cols]

    keep_cols = ["season", "player_name", "team", "position", "position_rank", "bye_week", "Rank", "AVG"] + platform_cols
    df = df[keep_cols].rename(columns={"Rank": "adp_rank", "AVG": "adp_avg"})

    # FantasyPros' export leaves team/bye blank for some players -- in
    # practice, mostly ones whose team status was unsettled (holdout, trade
    # rumor, suspension, etc.) at the moment of that ADP snapshot. This is a
    # real gap in their data, not a parsing bug. We don't rely on team/bye
    # from this source anyway -- team comes from seasonal_rosters.parquet
    # and bye week comes from schedules.parquet later, joined on
    # player_name + season. So we just note the count here and move on.
    unparsed_mask = df["team"].isna()
    if unparsed_mask.sum() > 0:
        print(f"  [{year}] Note: {unparsed_mask.sum()} row(s) had no team/bye in the source CSV "
              f"(expected -- will be filled in later from roster data).")

    return df


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(ADP_RAW_DIR, "adp_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No files found matching data/raw/adp/adp_*.csv. "
            f"Make sure your downloaded CSVs are named like adp_2024.csv"
        )

    print(f"Found {len(files)} ADP files.\n")

    all_years = []
    for filepath in files:
        filename = os.path.basename(filepath)
        year_match = re.search(r"(\d{4})", filename)
        if not year_match:
            print(f"  Skipping {filename}: couldn't find a 4-digit year in the filename.")
            continue
        year = int(year_match.group(1))

        print(f"Parsing {filename} (season {year})...")
        year_df = load_and_clean_one_year(filepath, year)
        all_years.append(year_df)

    combined = pd.concat(all_years, ignore_index=True)

    out_path = os.path.join(PROCESSED_DIR, "adp_combined.parquet")
    combined.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"\nSaved combined ADP table: {combined.shape[0]:,} rows x {combined.shape[1]} cols -> {out_path}")
    print(f"Seasons covered: {sorted(combined['season'].unique())}")


if __name__ == "__main__":
    main()
