"""
draft_simulator.py

Simulates a 10-team snake draft (skill positions only: QB/RB/WR/TE, matching
the project's scope decision from the EDA phase). Provides:
  - The core draft mechanics (snake order, position caps, pick logic)
  - Our model-informed agent's blended ranking strategy
  - A library of named real-world draft strategies (Zero RB, Hero RB, etc.)
  - Roster scoring (optimal starting lineup from actual season outcomes)

This mirrors the logic validated in notebooks/04_draft_simulator.ipynb.

Headline findings from that notebook (see README for full writeup):
  - The model-blended agent strategy did NOT reliably beat pure ADP.
  - "Robust RB" (draft RB with 3 of your first 3 picks) was the strongest
    strategy tested, consistently beating the field across all 3 test
    seasons (2022-2024) -- likely because it hedges against RB's unusually
    high bust rate (see EDA notebook 01's bust/breakout analysis).
"""

import random

import numpy as np
import pandas as pd

from build_features import SKILL_POSITIONS

N_TEAMS = 10
ROSTER_SLOTS = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1}
BENCH_SIZE = 6
TOTAL_ROUNDS = sum(ROSTER_SLOTS.values()) + BENCH_SIZE

POSITION_CAPS = {'QB': 2, 'RB': 5, 'WR': 5, 'TE': 2}

# Final weights -- see notebook Step 9's revision attempt, which performed
# WORSE on clean holdout data and was explicitly rejected. These original
# weights (derived from the EDA's per-position ADP-vs-outcome correlation
# gap) are the ones actually used going forward.
MODEL_WEIGHT_BY_POSITION = {'QB': 0.30, 'RB': 0.55, 'WR': 0.35, 'TE': 0.55}

# Named draft strategies: forced/banned positions per round, otherwise
# best-available-by-ADP. See notebook Step 12 for the tournament that
# compared these head-to-head.
STRATEGIES = {
    'Pure ADP': {'forced': {}, 'banned': {}},
    'Zero RB': {'forced': {}, 'banned': {r: {'RB'} for r in range(1, 6)}},
    'Hero RB': {'forced': {1: 'RB'}, 'banned': {r: {'RB'} for r in range(2, 6)}},
    'Robust RB': {'forced': {1: 'RB', 2: 'RB', 3: 'RB'}, 'banned': {}},
    'Elite TE': {'forced': {2: 'TE'}, 'banned': {}},
    'Punt QB/TE': {'forced': {}, 'banned': {r: {'QB', 'TE'} for r in range(1, 8)}},
    'Model Blend': {'forced': {}, 'banned': {}, 'use_model': True},
}


def snake_order(n_teams: int, n_rounds: int) -> list:
    order = []
    for rnd in range(n_rounds):
        round_order = list(range(n_teams)) if rnd % 2 == 0 else list(range(n_teams - 1, -1, -1))
        order.extend(round_order)
    return order


def compute_agent_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Blended ranking for the model-informed agent, expressed on the ADP
    scale (lower is better) so the model and ADP signals are directly
    comparable. For each position, maps the model's positional rank to the
    ADP value of whoever holds that same positional rank by real ADP --
    this preserves real cross-position scarcity (unlike naive within-
    position percentiles, which lose that signal -- see notebook for the
    bug this fixed)."""
    pool = pool.copy()

    for pos in SKILL_POSITIONS:
        pos_players = pool[pool['position'] == pos]
        model_order = pos_players.sort_values('model_pred', ascending=False)
        adp_order = pos_players.sort_values('adp_avg', ascending=True)['adp_avg'].values
        model_implied_adp = pd.Series(adp_order[:len(model_order)], index=model_order.index)
        pool.loc[model_implied_adp.index, 'model_implied_adp'] = model_implied_adp

    weight_map = pool['position'].map(MODEL_WEIGHT_BY_POSITION)
    pool['agent_score'] = -(
        weight_map * pool['model_implied_adp'] + (1 - weight_map) * pool['adp_avg']
    )
    return pool


def rule_based_pick(eligible: pd.DataFrame, round_num: int, strategy: dict) -> pd.Series:
    forced_pos = strategy['forced'].get(round_num)
    banned_pos = strategy['banned'].get(round_num, set())

    candidates = eligible[~eligible['position'].isin(banned_pos)]
    if forced_pos and (candidates['position'] == forced_pos).any():
        candidates = candidates[candidates['position'] == forced_pos]
    if candidates.empty:
        candidates = eligible

    if strategy.get('use_model'):
        return candidates.sort_values('agent_score', ascending=False).iloc[0]
    return candidates.sort_values('adp_avg', ascending=True).iloc[0]


def run_strategy_draft(pool: pd.DataFrame, slot_assignments: dict) -> dict:
    """slot_assignments: {team_index: strategy_name}. Returns {team_index: [picks]}."""
    pool_scored = compute_agent_scores(pool)
    available = pool_scored.copy()
    rosters = {t: [] for t in range(N_TEAMS)}
    position_counts = {t: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0} for t in range(N_TEAMS)}
    pick_order = snake_order(N_TEAMS, TOTAL_ROUNDS)

    round_counter = {t: 0 for t in range(N_TEAMS)}
    for team in pick_order:
        round_counter[team] += 1
        strategy = STRATEGIES[slot_assignments[team]]

        eligible = available[available['position'].map(
            lambda p, t=team: position_counts[t][p] < POSITION_CAPS[p]
        )]
        if eligible.empty:
            eligible = available

        pick = rule_based_pick(eligible, round_counter[team], strategy)
        rosters[team].append(pick)
        position_counts[team][pick['position']] += 1
        available = available[available['player_id'] != pick['player_id']]

    return rosters


def run_draft(pool: pd.DataFrame, agent_team_index: int) -> dict:
    """Convenience wrapper: agent (Model Blend) in one slot, Pure ADP everywhere else."""
    slot_assignments = {t: ('Model Blend' if t == agent_team_index else 'Pure ADP') for t in range(N_TEAMS)}
    return run_strategy_draft(pool, slot_assignments)


def score_roster(roster_players: list) -> float:
    """Optimal starting lineup (QB/RB/RB/WR/WR/TE + best remaining FLEX) by
    actual season points. Greedy fill is optimal for this simple slot
    structure."""
    roster_df = pd.DataFrame(roster_players)
    used_ids = set()
    starters = []

    for pos, n_slots in [('QB', 1), ('RB', 2), ('WR', 2), ('TE', 1)]:
        pos_players = roster_df[roster_df['position'] == pos].sort_values('actual_points', ascending=False)
        pos_players = pos_players[~pos_players['player_id'].isin(used_ids)]
        picks = pos_players.head(n_slots)
        starters.append(picks)
        used_ids.update(picks['player_id'])

    flex_eligible = roster_df[roster_df['position'].isin(['RB', 'WR', 'TE'])]
    flex_eligible = flex_eligible[~flex_eligible['player_id'].isin(used_ids)]
    flex_pick = flex_eligible.sort_values('actual_points', ascending=False).head(1)
    starters.append(flex_pick)

    starters_df = pd.concat(starters, ignore_index=True)
    return starters_df['actual_points'].sum()


def score_draft(rosters: dict) -> dict:
    return {team: score_roster(players) for team, players in rosters.items()}


def build_player_pool(season, features, features_encoded, model, feature_cols, adp_with_outcomes) -> pd.DataFrame:
    """Builds the draftable player pool for a given season: model prediction,
    position, ADP, and actual outcome per player."""
    season_df = features_encoded[features_encoded['season'] == season].copy()
    season_df['model_pred'] = model.predict(season_df[feature_cols])

    position_lookup = features[features['season'] == season][['player_id', 'position']]
    adp_season = adp_with_outcomes[adp_with_outcomes['season'] == season][['player_id', 'player_name', 'adp_avg']]

    pool = season_df[['player_id', 'model_pred', 'fantasy_points_ppr']].rename(columns={'fantasy_points_ppr': 'actual_points'})
    pool = pool.merge(position_lookup, on='player_id', how='left')
    pool = pool.merge(adp_season, on='player_id', how='inner')
    return pool.sort_values('adp_avg').reset_index(drop=True)
