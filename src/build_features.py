"""
build_features.py

Reusable feature engineering for the player projection model. Builds one
row per (player, season), where every feature is derived only from
information available BEFORE that season started (prior-season stats,
career/draft info, injury history). The target (fantasy_points_ppr) is the
season's actual outcome -- included in the output for training, but never
used as an input feature.

This module is used both for training (src/train_model.py) and, later, for
generating predictions ahead of an actual draft (the draft simulator will
call build_features() on the most recent completed season to project the
upcoming one).

Usage:
    from build_features import build_features
    features = build_features(seasonal, rosters, injuries)
"""

import numpy as np
import pandas as pd

SKILL_POSITIONS = ['QB', 'RB', 'WR', 'TE']


def safe_mode(x: pd.Series):
    """Most common value in a group; returns NaN instead of crashing if the
    group is all-NaN (plain .mode().iloc[0] raises IndexError in that case)."""
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan


def build_base_table(seasonal: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, season) with position attached and target
    (fantasy_points_ppr) included, restricted to skill positions."""
    roster_position = (
        rosters.groupby(['player_id', 'season'])['position']
        .agg(safe_mode)
        .reset_index()
    )
    base = seasonal.merge(roster_position, on=['player_id', 'season'], how='left')
    base = base[base['position'].isin(SKILL_POSITIONS)].copy()
    return base


def build_lag_features(base: pd.DataFrame) -> pd.DataFrame:
    """Adds every numeric stat column from `base`, lagged 1 and 2 seasons
    back, plus a season-over-season points trend."""
    exclude_cols = ['player_id', 'season', 'season_type', 'position']
    stat_cols = [c for c in base.columns if c not in exclude_cols]

    def build_lag(df, shift_amount, suffix):
        lagged = df[['player_id', 'season'] + stat_cols].copy()
        lagged['season'] = lagged['season'] + shift_amount
        lagged = lagged.rename(columns={c: f'{c}_{suffix}' for c in stat_cols})
        return lagged

    lag1 = build_lag(base, 1, 'lag1')
    lag2 = build_lag(base, 2, 'lag2')

    features = base.merge(lag1, on=['player_id', 'season'], how='left')
    features = features.merge(lag2, on=['player_id', 'season'], how='left')
    features['points_trend'] = features['fantasy_points_ppr_lag1'] - features['fantasy_points_ppr_lag2']
    return features


def add_career_features(features: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Adds age, years of experience (per player-season), and NFL draft
    capital (static per player's career)."""
    career_features = (
        rosters.groupby(['player_id', 'season'])[['age', 'years_exp']]
        .agg(safe_mode)
        .reset_index()
    )
    draft_capital = (
        rosters[rosters['draft_number'].notna()]
        .sort_values('season')
        .groupby('player_id')['draft_number']
        .first()
        .reset_index()
    )
    features = features.merge(career_features, on=['player_id', 'season'], how='left')
    features = features.merge(draft_capital, on='player_id', how='left')
    return features


def add_rate_features(features: pd.DataFrame) -> pd.DataFrame:
    """Per-game rate stats (prior season total / prior season games), so the
    model can separate talent from games-missed/availability."""
    features['pts_per_game_lag1'] = features['fantasy_points_ppr_lag1'] / features['games_lag1'].replace(0, np.nan)
    features['pts_per_game_lag2'] = features['fantasy_points_ppr_lag2'] / features['games_lag2'].replace(0, np.nan)

    for stat in ['targets', 'receptions', 'carries', 'receiving_yards', 'rushing_yards']:
        lag1_col = f'{stat}_lag1'
        if lag1_col in features.columns:
            features[f'{stat}_per_game_lag1'] = features[lag1_col] / features['games_lag1'].replace(0, np.nan)
    return features


def add_injury_features(features: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Prior-season injury report designation counts. Unlike other lag
    features, missing here genuinely means zero injury history (not
    unknown), so NaN is filled with 0."""
    injuries = injuries.copy()
    injuries['season'] = injuries['season'].astype(int)
    injuries_renamed = injuries.rename(columns={'gsis_id': 'player_id'})

    injury_counts = injuries_renamed.groupby(['player_id', 'season']).agg(
        weeks_on_report=('week', 'nunique'),
        weeks_out=('report_status', lambda x: (x == 'Out').sum()),
        weeks_questionable=('report_status', lambda x: (x == 'Questionable').sum()),
        weeks_doubtful=('report_status', lambda x: (x == 'Doubtful').sum()),
    ).reset_index()

    injury_counts['season'] = injury_counts['season'] + 1
    injury_lag_cols = ['weeks_on_report', 'weeks_out', 'weeks_questionable', 'weeks_doubtful']
    injury_counts = injury_counts.rename(columns={c: f'{c}_lag1' for c in injury_lag_cols})

    features = features.merge(injury_counts, on=['player_id', 'season'], how='left')
    for col in [f'{c}_lag1' for c in injury_lag_cols]:
        features[col] = features[col].fillna(0)
    return features


def get_feature_columns(features: pd.DataFrame) -> list:
    """The actual list of columns safe to use as model INPUTS -- excludes
    identifiers, the target, and current-season (un-lagged) stat columns,
    which would be data leakage.

    Rate-stat and injury columns ALSO end in _lag1/_lag2 (e.g.
    'pts_per_game_lag1', 'weeks_out_lag1'), so they must be excluded from
    the generic lag-suffix match below -- otherwise they'd be added twice,
    producing duplicate column names that break XGBoost's DataFrame
    handling (a duplicate-named selection returns a 2D slice instead of a
    single column)."""
    rate_cols = [c for c in features.columns if 'per_game' in c]
    injury_cols = ['weeks_on_report_lag1', 'weeks_out_lag1', 'weeks_questionable_lag1', 'weeks_doubtful_lag1']
    exclude_from_lag = set(rate_cols) | set(injury_cols)

    lag_cols = [
        c for c in features.columns
        if (c.endswith('_lag1') or c.endswith('_lag2')) and c not in exclude_from_lag
    ]
    other_cols = ['points_trend', 'age', 'years_exp', 'draft_number']
    return lag_cols + rate_cols + injury_cols + other_cols


def build_features(seasonal: pd.DataFrame, rosters: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw tables in, model-ready feature table out."""
    base = build_base_table(seasonal, rosters)
    features = build_lag_features(base)
    features = add_career_features(features, rosters)
    features = add_rate_features(features)
    features = add_injury_features(features, injuries)
    return features
