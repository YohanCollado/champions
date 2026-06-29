# Here we will calculate and manipulate features using 
# Pandas is used to organize, clean and manipulate data
# Numpy is used for extremely fast numerical operations

import pandas as pd
import numpy as np

TEAM_FEATURES = [
    'league_difficulty',
    'league_win_percentage'
    'goals_scored',
    'goals_scored_last_5',
    'goals_conceded_last_5',
    'points_last_5',
    'goals_conceded',
    'goals_by_set_piece',
    'squad_value',
    'average_chances_created',
    'injured_players_value',
    'missing_starting_player',
    'wins_last_5',
    'draws_last_5',
    'losses_last_5',
    'uefa_coefficient',
    'average_posession',
    'average_shots',
    'average_shots_on_target',
    'average_corners',
    'average_saves',
    'average_fouls_conceded',
    'average_xG',
    'average_duels_won_percentage',
    'average_offsides',
    'average_passes',
    'average_tackles_won',
    'average_aerial_duels_won',
    'average_pass_accuracy',
    'average_total_crosses',
    'average_yellow_cards',
    'average_red_cards',
    'goal_difference',
    'home_win_percentage',
    'away_win_percentage',
    'clean_sheet_percentage',
    'ball_recoveries',
    'average_xGA',
    'average_team_age',
    'squad_value_rank',
    'manager_ucl_games',
    'player_maches_played',
]