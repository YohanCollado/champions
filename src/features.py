"""Feature names used by the assignment model.

The original project started with a broader UEFA feature wish list. For the
final submission, the production model uses features that are available in the
included historical CSVs before each match is played.
"""

TEAM_FEATURES = [
    "home_matches_played_before",
    "away_matches_played_before",
    "home_win_rate_before",
    "away_win_rate_before",
    "home_goals_scored_last_5",
    "away_goals_scored_last_5",
    "home_goals_conceded_last_5",
    "away_goals_conceded_last_5",
    "home_points_last_5",
    "away_points_last_5",
    "home_rest_days",
    "away_rest_days",
    "home_advantage",
    "team_experience_difference",
    "win_rate_before_difference",
    "goals_scored_last_5_difference",
    "goals_conceded_last_5_difference",
    "points_last_5_difference",
]
