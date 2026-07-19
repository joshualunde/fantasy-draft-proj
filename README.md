# Beat-the-ADP Fantasy Draft Bot

An end-to-end ML project testing whether a statistical player-projection model can outperform crowd-sourced fantasy football draft consensus (ADP), and if not, what actually does.

**TL;DR**: The projection model roughly ties (and sometimes trails) ADP on raw predictive accuracy, and a model-informed draft strategy did **not** reliably beat drafting by pure ADP. But a simple, well-known heuristic — **"Robust RB" (draft running backs aggressively in the first 3 rounds)** — consistently outperformed both ADP and the model across three full seasons of backtesting. The most interesting result of this project turned out to be a negative one for the original hypothesis, and a positive one for positional-scarcity-aware drafting.

## Motivation

Fantasy football ADP (Average Draft Position) is the aggregated judgment of thousands of real drafters — a genuine "wisdom of crowds" signal. This project asks two questions:

1. Can a machine learning model, trained only on historical box-score stats, beat that consensus?
2. If not, what draft *strategy* actually performs best when backtested against real historical outcomes?

## Project structure

```
data/
  raw/            # nflverse stats, rosters, schedules, injuries; FantasyPros ADP CSVs
  processed/      # cleaned/joined derived datasets
notebooks/
  01_eda.ipynb                  # data quality, name matching, ADP baseline, bust/breakout analysis
  02_strength_of_schedule.ipynb # SOS explored and ruled out as a useful feature (documented negative result)
  03_modeling.ipynb             # feature engineering, XGBoost projection model, hyperparameter tuning
  04_draft_simulator.ipynb      # snake draft simulation, model-vs-ADP test, named strategy tournament
src/
  data_ingestion.py    # pulls nflverse data (stats, rosters, schedules, injuries)
  adp_ingestion.py     # parses/cleans FantasyPros ADP exports
  build_features.py    # reusable feature engineering pipeline
  train_model.py        # trains and saves the final XGBoost model
  draft_simulator.py   # draft mechanics, agent strategy, named strategy library
models/
  xgb_projection_model.json  # trained model
  feature_columns.json       # exact feature list/order the model expects
```

## Data

- **Player stats, rosters, schedules, injuries**: [nflverse](https://github.com/nflverse) via `nfl_data_py`, 2015-2024 (10 seasons)
- **ADP**: manually exported from FantasyPros (PPR scoring), 2015-2024
- **Scope**: restricted to QB/RB/WR/TE. Kickers are historically low-signal and rarely differentiate draft strategy; team defenses (DST) require a separate team-level scoring pipeline that was out of scope.

## Methodology & findings

### 1. Data cleaning & name matching
ADP data has no player ID, so matching it to nflverse stats required text-based name matching — nicknames, suffixes, and legal-name variants (`Hollywood Brown` vs. `Marquise Brown`) all needed handling. Final match rate: 90.9% raw, ~99.9% effective once players with zero recorded games that season (retired, suspended, injured before playing a snap) are excluded — those aren't matching failures, they're players with no real outcome to predict.

### 2. EDA — how good is ADP, really?
- Spearman correlation between ADP and actual season outcome: **QB -0.78, WR -0.71, RB -0.70, TE -0.68**. Even at its best (QB), ADP only explains ~61% of outcome variance — meaning **more than a third of what happens each season is not explained by pre-season consensus**, even at the position the market prices best.
- **Bust rate by position and round**: a first-round RB busts (scores well below its draft-slot expectation) **22.1%** of the time, vs. just **6.5%** for a first-round WR. This single finding turned out to be the seed of the project's most important result later on.
- **Strength of schedule** (season-long, position-specific "points allowed" by opponents faced) was tested as a potential feature and found to add essentially no signal beyond ADP (correlation with residual performance: 0.02-0.10, mostly not statistically significant). Documented as a negative result rather than discarded silently.

### 3. Projection model
XGBoost, trained on lagged prior-season stats, career/draft-capital features, per-game rate stats, and injury history — deliberately **excluding ADP itself** as an input, since it's the benchmark being tested against.

- Beat a naive "predict this year = last year" baseline by ~13-16% (MAE/RMSE).
- Did **not** clearly beat ADP's own predictive correlation: roughly tied at RB/TE, clearly behind at QB/WR, on the one fully clean holdout season (2024).
- Feature engineering (rate stats, injury history) and hyperparameter tuning each produced only marginal improvement — a sign the model was hitting the ceiling of what box-score history alone can predict. The remaining gap to ADP is plausibly made up of information no stats table captures: coaching changes, camp battles, contract situations.

### 4. Draft simulator & strategy tournament
Built a 10-team snake draft simulator (skill positions only) to test strategies against real historical outcomes across 2022-2024.

- **Model-informed agent vs. pure ADP**: 0-20% win rate depending on season; underperformed on the one clean test season (2024). An attempt to fix the strategy using cross-season diagnostic data made results *worse*, not better — a useful lesson in not over-fitting a strategy to a small sample.
- **Named strategy tournament** (Zero RB, Hero RB, Robust RB, Elite TE, Punt QB/TE, Pure ADP, Model Blend), 168 simulated team-drafts across 3 seasons and randomized draft slots:

| Strategy | Avg. rank (of 10) | Win rate | 
|---|---|---|
| **Robust RB** | **4.21** | **25.0%** |
| Zero RB | 5.21 | 4.2% |
| Hero RB | 5.29 | 12.5% |
| Elite TE | 5.46 | 8.3% |
| Pure ADP | 5.72 | 8.3% |
| Punt QB/TE | 5.83 | 8.3% |
| Model Blend | 6.13 | 8.3% |

**Robust RB — drafting RB with 3 of your first 3 picks — was the clear winner**, and held up consistently across all three individual seasons (not just in aggregate). This connects directly back to the EDA: RB's unusually high bust rate means securing multiple early RBs functions as a hedge against that specific volatility, capturing the position's workhorse upside while diversifying against its risk. The model-informed strategy, by contrast, finished **last** of all seven approaches tested.

- **Draft slot analysis**: no single slot won consistently across seasons under identical strategy — mostly noise — with one consistent exception: **the literal 1st overall pick finished below the median field in all three seasons**, likely due to the long wait before your next pick in a snake draft.

## Honest conclusions

- A model trained on historical box-score data alone does not reliably beat aggregated market consensus (ADP) — a legitimate and realistic finding, not a failure of methodology.
- Positional-scarcity-aware heuristics (Robust RB) outperformed both the model and pure ADP in backtesting, suggesting that *when* you draft a position matters as much as *who* you draft.
- All findings here were validated with explicit controls (a pure-ADP-vs-pure-ADP baseline confirming simulator correctness), multi-season testing, and — critically — a rejected attempt to "improve" the strategy using diagnostic data that made results worse, which is reported rather than hidden.

## Setup & running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/data_ingestion.py         # pulls nflverse data to data/raw/
# manually export FantasyPros ADP CSVs to data/raw/adp/adp_<year>.csv
python src/adp_ingestion.py          # cleans ADP data to data/processed/
python src/train_model.py            # trains and saves the model to models/
```

Then open the notebooks in `notebooks/` for the full analysis walkthrough, or use `src/draft_simulator.py` directly to run your own simulated drafts.

## Tech stack

Python, pandas, XGBoost, scikit-learn, Jupyter. Data via `nfl_data_py` (nflverse) and FantasyPros.

## Limitations & future work

- 754 player-seasons matched an ADP entry but lack a corresponding seasonal-stats row (likely preseason/playoff-only appearances) — not yet investigated.
- Season-long point totals were used to score simulated rosters, not real head-to-head weekly matchups — a full weekly simulation (with lineup-setting decisions) would add realistic variance currently missing.
- Draft-slot analysis used randomized, uneven slot assignments — a fully controlled factorial design (equal draws per slot) would sharpen that finding.
- Strength-of-schedule was tested at the season level only; a week-level version (relevant to weekly start/sit decisions rather than draft strategy) was not explored and would be a different project in its own right.
