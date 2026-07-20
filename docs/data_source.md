# Data Source

The submitted project uses public football match CSV files from football-data.co.uk.

- Website: https://www.football-data.co.uk/
- Data page: https://www.football-data.co.uk/downloadm.php
- Included raw files: `data/raw/*.csv`
- Final cleaned dataset: `data/champions_league_dataset.csv`

## Included Seasons

- 2021-2022
- 2022-2023
- 2023-2024
- 2024-2025

## Included Leagues

| Code | League |
|---|---|
| `E0` | England Premier League |
| `SP1` | Spain La Liga |
| `I1` | Italy Serie A |
| `D1` | Germany Bundesliga |
| `F1` | France Ligue 1 |

## Target

The target is `match_result`, derived from `FTR`:

- `H`: Home Win
- `D`: Draw
- `A`: Away Win

## Notes

The original project included a broad wish list of Champions League features. For this final assignment version, the model uses reliable columns available in the bundled CSV files. In-match statistics and final scores are kept for EDA but excluded from model features to avoid target leakage.
