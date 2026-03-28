import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


TOP_PLAYERS = [
    'AB de Villiers', 'CH Gayle', 'RG Sharma', 'DA Warner', 'V Kohli',
    'MS Dhoni', 'SR Watson', 'YK Pathan', 'RA Jadeja', 'SP Narine',
    'AD Russell', 'SK Raina', 'KA Pollard', 'G Gambhir', 'JC Buttler',
    'KL Rahul', 'S Dhawan', 'RV Uthappa', 'HH Pandya', 'KH Pandya'
]


def compute_player_features(df):
    """Compute player impact and star player features."""
    df = df.sort_values('date').reset_index(drop=True)
    
    pom_counts = df.groupby('player_of_match').size().to_dict()
    
    team1_pom_impact = []
    team2_pom_impact = []
    team1_star_count = []
    team2_star_count = []
    
    team_pom_history = {}
    team_star_history = {}
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        date = row['date']
        pom = row['player_of_match']
        
        t1_history = team_pom_history.get(team1, {})
        t2_history = team_pom_history.get(team2, {})
        
        t1_pom = sum(t1_history.values()) / max(len(t1_history), 1)
        t2_pom = sum(t2_history.values()) / max(len(t2_history), 1)
        
        team1_pom_impact.append(t1_pom)
        team2_pom_impact.append(t2_pom)
        
        t1_stars = team_star_history.get(team1, 0)
        t2_stars = team_star_history.get(team2, 0)
        
        team1_star_count.append(t1_stars)
        team2_star_count.append(t2_stars)
        
        if pom in TOP_PLAYERS:
            winner = row['winner']
            if winner == team1:
                team_pom_history[team1] = team_pom_history.get(team1, {})
                team_pom_history[team1][pom] = team_pom_history[team1].get(pom, 0) + 1
            elif winner == team2:
                team_pom_history[team2] = team_pom_history.get(team2, {})
                team_pom_history[team2][pom] = team_pom_history[team2].get(pom, 0) + 1
    
    df['team1_pom_impact'] = team1_pom_impact
    df['team2_pom_impact'] = team2_pom_impact
    df['team1_star_count'] = team1_star_count
    df['team2_star_count'] = team2_star_count
    df['pom_diff'] = df['team1_pom_impact'] - df['team2_pom_impact']
    df['star_diff'] = df['team1_star_count'] - df['team2_star_count']
    
    return df


def compute_venue_batting_stats(df):
    """Compute venue-specific batting statistics."""
    df = df.sort_values('date').reset_index(drop=True)
    
    venue_avg_runs = {}
    team1_venue_avg = []
    team2_venue_avg = []
    
    venue_team_runs = {}
    
    for idx, row in df.iterrows():
        venue = row['venue']
        
        if venue not in venue_avg_runs:
            venue_avg_runs[venue] = {'total': 0, 'count': 0}
        
        avg = venue_avg_runs[venue]['total'] / max(venue_avg_runs[venue]['count'], 1)
        team1_venue_avg.append(avg)
        team2_venue_avg.append(avg)
        
        if pd.notna(row['result']) and row['result'] in ['runs']:
            winner = row['winner']
            margin = row['result_margin'] if pd.notna(row['result_margin']) else 0
            
            if winner == row['team1']:
                team1_runs = margin
                team2_runs = 0
            else:
                team1_runs = 0
                team2_runs = margin
            
            venue_avg_runs[venue]['total'] += team1_runs + team2_runs
            venue_avg_runs[venue]['count'] += 1
    
    df['venue_avg_runs'] = team1_venue_avg
    
    return df


def compute_toss_impact(df):
    """Compute toss winner's historical win rate."""
    df = df.sort_values('date').reset_index(drop=True)
    
    toss_impact = []
    
    toss_winner_history = {}
    
    for idx, row in df.iterrows():
        toss_winner = row['toss_winner']
        
        history = toss_winner_history.get(toss_winner, {'wins': 0, 'toss': 0})
        
        if history['toss'] > 0:
            toss_impact.append(history['wins'] / history['toss'])
        else:
            toss_impact.append(0.5)
        
        if pd.notna(row['winner']):
            toss_winner_history[toss_winner] = {
                'wins': history['wins'] + (1 if row['winner'] == toss_winner else 0),
                'toss': history['toss'] + 1
            }
    
    df['toss_winner_hist_winrate'] = toss_impact
    
    return df


def compute_historical_ball_features(matches_df, deliveries_df):
    """Compute historical ball-by-ball features (before each match)."""
    print("Computing historical ball-by-ball features...")
    
    deliveries = deliveries_df.copy()
    matches = matches_df.sort_values('date').reset_index(drop=True)
    
    deliveries['total_runs'] = deliveries['total_runs'].fillna(0)
    deliveries['is_wicket'] = deliveries['is_wicket'].fillna(0)
    
    deliveries = deliveries.merge(matches[['id', 'date', 'team1', 'team2']], 
                                 left_on='match_id', right_on='id', how='left')
    
    deliveries['is_powerplay'] = (deliveries['over'] <= 6).astype(int)
    deliveries['is_death'] = (deliveries['over'] >= 16).astype(int)
    
    team_powerplay_runs = {}
    team_powerplay_balls = {}
    team_death_runs = {}
    team_death_balls = {}
    team_total_runs = {}
    team_total_balls = {}
    team_wickets = {}
    
    t1_powerplay = []
    t2_powerplay = []
    t1_death = []
    t2_death = []
    t1_strike_rate = []
    t2_strike_rate = []
    
    deliveries = deliveries.sort_values('date')
    
    for idx, row in matches.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        
        t1_pp_runs = team_powerplay_runs.get(team1, 0)
        t1_pp_balls = team_powerplay_balls.get(team1, 0)
        t2_pp_runs = team_powerplay_runs.get(team2, 0)
        t2_pp_balls = team_powerplay_balls.get(team2, 0)
        
        t1_death_r = team_death_runs.get(team1, 0)
        t1_death_b = team_death_balls.get(team1, 0)
        t2_death_r = team_death_runs.get(team2, 0)
        t2_death_b = team_death_balls.get(team2, 0)
        
        t1_total_r = team_total_runs.get(team1, 0)
        t1_total_b = team_total_balls.get(team1, 0)
        t2_total_r = team_total_runs.get(team2, 0)
        t2_total_b = team_total_balls.get(team2, 0)
        
        t1_powerplay.append(t1_pp_runs / max(t1_pp_balls, 1) * 6 if t1_pp_balls > 0 else 30)
        t2_powerplay.append(t2_pp_runs / max(t2_pp_balls, 1) * 6 if t2_pp_balls > 0 else 30)
        
        t1_death.append(t1_death_r / max(t1_death_b, 1) * 6 if t1_death_b > 0 else 30)
        t2_death.append(t2_death_r / max(t2_death_b, 1) * 6 if t2_death_b > 0 else 30)
        
        t1_strike_rate.append(t1_total_r / max(t1_total_b, 1) * 100 if t1_total_b > 0 else 100)
        t2_strike_rate.append(t2_total_r / max(t2_total_b, 1) * 100 if t2_total_b > 0 else 100)
        
        match_deliveries = deliveries[deliveries['match_id'] == row['id']]
        
        for _, ball in match_deliveries.iterrows():
            bat_team = ball['batting_team']
            
            if ball['is_powerplay']:
                if bat_team == team1:
                    team_powerplay_runs[team1] = team_powerplay_runs.get(team1, 0) + ball['total_runs']
                    team_powerplay_balls[team1] = team_powerplay_balls.get(team1, 0) + 1
                elif bat_team == team2:
                    team_powerplay_runs[team2] = team_powerplay_runs.get(team2, 0) + ball['total_runs']
                    team_powerplay_balls[team2] = team_powerplay_balls.get(team2, 0) + 1
            
            if ball['is_death']:
                if bat_team == team1:
                    team_death_runs[team1] = team_death_runs.get(team1, 0) + ball['total_runs']
                    team_death_balls[team1] = team_death_balls.get(team1, 0) + 1
                elif bat_team == team2:
                    team_death_runs[team2] = team_death_runs.get(team2, 0) + ball['total_runs']
                    team_death_balls[team2] = team_death_balls.get(team2, 0) + 1
            
            if bat_team == team1:
                team_total_runs[team1] = team_total_runs.get(team1, 0) + ball['total_runs']
                team_total_balls[team1] = team_total_balls.get(team1, 0) + 1
                team_wickets[team1] = team_wickets.get(team1, 0) + ball['is_wicket']
            elif bat_team == team2:
                team_total_runs[team2] = team_total_runs.get(team2, 0) + ball['total_runs']
                team_total_balls[team2] = team_total_balls.get(team2, 0) + 1
                team_wickets[team2] = team_wickets.get(team2, 0) + ball['is_wicket']
    
    matches['team1_powerplay_avg'] = t1_powerplay
    matches['team2_powerplay_avg'] = t2_powerplay
    matches['team1_death_avg'] = t1_death
    matches['team2_death_avg'] = t2_death
    matches['team1_strike_rate'] = t1_strike_rate
    matches['team2_strike_rate'] = t2_strike_rate
    
    print(f"Historical ball-by-ball features added.")
    
    return matches


def compute_player_level_features(matches_df, deliveries_df):
    """Compute player-level features from deliveries data."""
    print("Computing player-level features...")
    
    deliveries = deliveries_df.copy()
    matches = matches_df.sort_values('date').reset_index(drop=True)
    
    deliveries = deliveries.merge(matches[['id', 'date', 'team1', 'team2']], 
                                 left_on='match_id', right_on='id', how='left')
    
    team_batsman_runs = {}
    team_batsman_balls = {}
    team_bowler_wickets = {}
    team_bowler_balls = {}
    
    t1_top_scorer_avg = []
    t2_top_scorer_avg = []
    t1_bowler_wkt_avg = []
    t2_bowler_wkt_avg = []
    t1_boundary_pct = []
    t2_boundary_pct = []
    
    for idx, row in matches.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        
        t1_runs = team_batsman_runs.get(team1, {})
        t1_balls = team_batsman_balls.get(team1, {})
        t2_runs = team_batsman_runs.get(team2, {})
        t2_balls = team_batsman_balls.get(team2, {})
        
        t1_wkts = team_bowler_wickets.get(team1, {})
        t1_bowl_balls = team_bowler_balls.get(team1, {})
        t2_wkts = team_bowler_wickets.get(team2, {})
        t2_bowl_balls = team_bowler_balls.get(team2, {})
        
        t1_top_scores = sorted(t1_runs.values(), reverse=True)[:3]
        t2_top_scores = sorted(t2_runs.values(), reverse=True)[:3]
        
        t1_top_scorer_avg.append(np.mean(t1_top_scores) if t1_top_scores else 20)
        t2_top_scorer_avg.append(np.mean(t2_top_scores) if t2_top_scores else 20)
        
        t1_wkt_avg = np.mean(list(t1_wkts.values())) if t1_wkts else 0.5
        t2_wkt_avg = np.mean(list(t2_wkts.values())) if t2_wkts else 0.5
        t1_bowler_wkt_avg.append(t1_wkt_avg)
        t2_bowler_wkt_avg.append(t2_wkt_avg)
        
        t1_boundaries = sum(1 for b in t1_balls.values() if b >= 4)
        t1_total_balls = sum(t1_balls.values())
        t1_boundary_pct.append(t1_boundaries / max(t1_total_balls, 1) * 100)
        
        t2_boundaries = sum(1 for b in t2_balls.values() if b >= 4)
        t2_total_balls = sum(t2_balls.values())
        t2_boundary_pct.append(t2_boundaries / max(t2_total_balls, 1) * 100)
        
        match_deliveries = deliveries[deliveries['match_id'] == row['id']]
        
        for _, ball in match_deliveries.iterrows():
            bat_team = ball['batting_team']
            bowler = ball['bowler']
            batter = ball['batter']
            runs = ball['batsman_runs']
            
            if bat_team == team1:
                team_batsman_runs[team1] = team_batsman_runs.get(team1, {})
                team_batsman_runs[team1][batter] = team_batsman_runs[team1].get(batter, 0) + runs
                team_batsman_balls[team1] = team_batsman_balls.get(team1, {})
                team_batsman_balls[team1][batter] = team_batsman_balls[team1].get(batter, 0) + 1
            elif bat_team == team2:
                team_batsman_runs[team2] = team_batsman_runs.get(team2, {})
                team_batsman_runs[team2][batter] = team_batsman_runs[team2].get(batter, 0) + runs
                team_batsman_balls[team2] = team_batsman_balls.get(team2, {})
                team_batsman_balls[team2][batter] = team_batsman_balls[team2].get(batter, 0) + 1
            
            if ball['is_wicket'] == 1 and pd.notna(bowler):
                if bat_team == team1:
                    team_bowler_wickets[team2] = team_bowler_wickets.get(team2, {})
                    team_bowler_wickets[team2][bowler] = team_bowler_wickets[team2].get(bowler, 0) + 1
                    team_bowler_balls[team2] = team_bowler_balls.get(team2, {})
                    team_bowler_balls[team2][bowler] = team_bowler_balls[team2].get(bowler, 0) + 1
                elif bat_team == team2:
                    team_bowler_wickets[team1] = team_bowler_wickets.get(team1, {})
                    team_bowler_wickets[team1][bowler] = team_bowler_wickets[team1].get(bowler, 0) + 1
                    team_bowler_balls[team1] = team_bowler_balls.get(team1, {})
                    team_bowler_balls[team1][bowler] = team_bowler_balls[team1].get(bowler, 0) + 1
    
    matches['team1_top_scorer_avg'] = t1_top_scorer_avg
    matches['team2_top_scorer_avg'] = t2_top_scorer_avg
    matches['team1_bowler_wkt_avg'] = t1_bowler_wkt_avg
    matches['team2_bowler_wkt_avg'] = t2_bowler_wkt_avg
    matches['team1_boundary_pct'] = t1_boundary_pct
    matches['team2_boundary_pct'] = t2_boundary_pct
    
    print("Player-level features added.")
    return matches


def compute_external_features(matches_df):
    """Compute external features (venue patterns, time, etc.)."""
    print("Computing external features...")
    
    matches = matches_df.copy()
    
    matches['is_night_match'] = matches['venue'].apply(
        lambda x: 1 if 'Stadium' in str(x) and 'D/N' in str(x) else 0
    )
    
    venue_chase_win = {}
    venue_bat_first_win = {}
    
    chase_win_pct = []
    bat_first_win_pct = []
    
    for idx, row in matches.iterrows():
        venue = row['venue']
        
        chase = venue_chase_win.get(venue, {'wins': 0, 'total': 0})
        bat_first = venue_bat_first_win.get(venue, {'wins': 0, 'total': 0})
        
        chase_win_pct.append(chase['wins'] / max(chase['total'], 1) * 100 if chase['total'] > 0 else 50)
        bat_first_win_pct.append(bat_first['wins'] / max(bat_first['total'], 1) * 100 if bat_first['total'] > 0 else 50)
        
        if pd.notna(row['winner']):
            winner = row['winner']
            
            if row['team1'] == winner:
                venue_chase_win[venue] = {'wins': chase['wins'], 'total': chase['total'] + 1}
                if row['toss_decision'] == 'field':
                    venue_chase_win[venue] = {'wins': chase['wins'] + 1, 'total': chase['total'] + 1}
                
                venue_bat_first_win[venue] = {'wins': bat_first['wins'] + 1, 'total': bat_first['total'] + 1}
                if row['toss_decision'] == 'bat':
                    venue_bat_first_win[venue] = {'wins': bat_first['wins'] + 1, 'total': bat_first['total'] + 1}
                else:
                    venue_bat_first_win[venue] = {'wins': bat_first['wins'], 'total': bat_first['total'] + 1}
            else:
                venue_chase_win[venue] = {'wins': chase['wins'], 'total': chase['total'] + 1}
                if row['toss_decision'] == 'bat':
                    venue_chase_win[venue] = {'wins': chase['wins'] + 1, 'total': chase['total'] + 1}
                
                venue_bat_first_win[venue] = {'wins': bat_first['wins'], 'total': bat_first['total'] + 1}
                if row['toss_decision'] == 'field':
                    venue_bat_first_win[venue] = {'wins': bat_first['wins'] + 1, 'total': bat_first['total'] + 1}
    
    matches['venue_chase_win_pct'] = chase_win_pct
    matches['venue_bat_first_win_pct'] = bat_first_win_pct
    
    matches['month'] = pd.to_datetime(matches['date']).dt.month
    matches['is_playoffs'] = matches['match_type'].apply(
        lambda x: 1 if x in ['Final', 'Semi Final', 'Qualifier', 'Eliminator'] else 0
    )
    
    print("External features added.")
    return matches


def compute_live_features(matches_df, deliveries_df):
    """Compute simulated live features (based on historical patterns)."""
    print("Computing live match features...")
    
    deliveries = deliveries_df.copy()
    matches = matches_df.sort_values('date').reset_index(drop=True)
    
    deliveries = deliveries.merge(matches[['id', 'date', 'team1', 'team2', 'winner']], 
                                 left_on='match_id', right_on='id', how='left')
    
    team_powerplay_runs = {}
    team_mid_innings_runs = {}
    team_wickets_powerplay = {}
    
    t1_powerplay_runs = []
    t2_powerplay_runs = []
    t1_mid_runs = []
    t2_mid_runs = []
    t1_wickets_powerplay = []
    t2_wickets_powerplay = []
    
    for idx, row in matches.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        
        t1_powerplay_runs.append(team_powerplay_runs.get(team1, 30))
        t2_powerplay_runs.append(team_powerplay_runs.get(team2, 30))
        t1_mid_runs.append(team_mid_innings_runs.get(team1, 40))
        t2_mid_runs.append(team_mid_innings_runs.get(team2, 40))
        t1_wickets_powerplay.append(team_wickets_powerplay.get(team1, 1))
        t2_wickets_powerplay.append(team_wickets_powerplay.get(team2, 1))
        
        match_deliveries = deliveries[deliveries['match_id'] == row['id']]
        
        for _, ball in match_deliveries.iterrows():
            bat_team = ball['batting_team']
            
            if ball['over'] <= 6:
                if bat_team == team1:
                    team_powerplay_runs[team1] = team_powerplay_runs.get(team1, 0) + ball['total_runs']
                    team_wickets_powerplay[team1] = team_wickets_powerplay.get(team1, 0) + ball['is_wicket']
                elif bat_team == team2:
                    team_powerplay_runs[team2] = team_powerplay_runs.get(team2, 0) + ball['total_runs']
                    team_wickets_powerplay[team2] = team_wickets_powerplay.get(team2, 0) + ball['is_wicket']
            
            if 7 <= ball['over'] <= 15:
                if bat_team == team1:
                    team_mid_innings_runs[team1] = team_mid_innings_runs.get(team1, 0) + ball['total_runs']
                elif bat_team == team2:
                    team_mid_innings_runs[team2] = team_mid_innings_runs.get(team2, 0) + ball['total_runs']
    
    matches['team1_powerplay_runs_hist'] = t1_powerplay_runs
    matches['team2_powerplay_runs_hist'] = t2_powerplay_runs
    matches['team1_mid_runs_hist'] = t1_mid_runs
    matches['team2_mid_runs_hist'] = t2_mid_runs
    matches['team1_wickets_pp_hist'] = t1_wickets_powerplay
    matches['team2_wickets_pp_hist'] = t2_wickets_powerplay
    
    print("Live features added.")
    return matches


class ELORatingSystem:
    """Format-specific ELO rating system for cricket teams with margin weighting."""
    
    def __init__(self, k_factor=20, home_advantage=50, default_elo=1500):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_elo = default_elo
        self.ratings = {}
    
    def get_elo(self, team, match_type, date):
        """Get team ELO for a specific format on a given date."""
        key = (team, match_type, date.year)
        return self.ratings.get(key, self.default_elo)
    
    def get_expected_score(self, elo1, elo2):
        """Calculate expected score using logistic curve."""
        return 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    def update_elo(self, winner, loser, match_type, date, result_margin=None, is_draw=False):
        """Update ELO ratings after a match with margin weighting."""
        winner_key = (winner, match_type, date.year)
        loser_key = (loser, match_type, date.year)
        
        winner_elo = self.ratings.get(winner_key, self.default_elo)
        loser_elo = self.ratings.get(loser_key, self.default_elo)
        
        expected_winner = self.get_expected_score(winner_elo, loser_elo)
        
        if is_draw:
            actual_winner = 0.5
            actual_loser = 0.5
            k = self.k_factor / 2
        else:
            actual_winner = 1.0
            actual_loser = 0.0
            k = self.k_factor
            
            if result_margin is not None and result_margin > 0:
                margin_factor = np.log1p(result_margin) / 10
                margin_factor = min(margin_factor, 1.5)
                k = k * (1 + margin_factor)
        
        new_winner_elo = winner_elo + k * (actual_winner - expected_winner)
        new_loser_elo = loser_elo + k * (actual_loser - (1 - expected_winner))
        
        self.ratings[winner_key] = new_winner_elo
        self.ratings[loser_key] = new_loser_elo


def process_matches_with_elo(df):
    """Process matches chronologically and compute ELO ratings."""
    df = df.sort_values('date').reset_index(drop=True)
    
    elo_system = ELORatingSystem(k_factor=20)
    
    team1_elo = []
    team2_elo = []
    expected_team1 = []
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        match_type = row['match_type']
        date = row['date']
        
        elo1 = elo_system.get_elo(team1, match_type, date)
        elo2 = elo_system.get_elo(team2, match_type, date)
        
        exp1 = elo_system.get_expected_score(elo1, elo2)
        
        team1_elo.append(elo1)
        team2_elo.append(elo2)
        expected_team1.append(exp1)
        
        if pd.notna(row['winner']):
            winner = row['winner']
            if winner == team1:
                loser = team2
            elif winner == team2:
                loser = team1
            else:
                continue
            
            is_draw = row['result'] == 'tie'
            result_margin = row['result_margin'] if pd.notna(row['result_margin']) else None
            elo_system.update_elo(winner, loser, match_type, date, result_margin, is_draw)
    
    df['team1_elo'] = team1_elo
    df['team2_elo'] = team2_elo
    df['expected_team1_win'] = expected_team1
    df['elo_diff'] = df['team1_elo'] - df['team2_elo']
    
    return df


def compute_streak_and_rest(df):
    """Compute win streak, loss streak, and days rest for each team."""
    df = df.sort_values('date').reset_index(drop=True)
    
    team1_streak = []
    team2_streak = []
    team1_rest = []
    team2_rest = []
    
    team_last_date = {}
    team_win_streak = {}
    team_loss_streak = {}
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        date = row['date']
        winner = row['winner'] if pd.notna(row['winner']) else None
        
        t1_last_date = team_last_date.get(team1)
        t2_last_date = team_last_date.get(team2)
        
        team1_rest.append((date - t1_last_date).days if t1_last_date else 7)
        team2_rest.append((date - t2_last_date).days if t2_last_date else 7)
        
        t1_streak = team_win_streak.get(team1, 0)
        t2_streak = team_win_streak.get(team2, 0)
        
        team1_streak.append(t1_streak if t1_streak > 0 else -team_loss_streak.get(team1, 0))
        team2_streak.append(t2_streak if t2_streak > 0 else -team_loss_streak.get(team2, 0))
        
        if winner:
            if winner == team1:
                team_win_streak[team1] = team_win_streak.get(team1, 0) + 1
                team_win_streak[team2] = 0
                team_loss_streak[team2] = team_loss_streak.get(team2, 0) + 1
                team_loss_streak[team1] = 0
            elif winner == team2:
                team_win_streak[team2] = team_win_streak.get(team2, 0) + 1
                team_win_streak[team1] = 0
                team_loss_streak[team1] = team_loss_streak.get(team1, 0) + 1
                team_loss_streak[team2] = 0
        
        team_last_date[team1] = date
        team_last_date[team2] = date
    
    df['team1_streak'] = team1_streak
    df['team2_streak'] = team2_streak
    df['team1_rest'] = team1_rest
    df['team2_rest'] = team2_rest
    df['streak_diff'] = df['team1_streak'] - df['team2_streak']
    df['rest_diff'] = df['team2_rest'] - df['team1_rest']
    
    return df


def compute_knockout_and_chase(df):
    """Compute knockout match flag and chase preference."""
    knockout_types = ['Final', 'Semi Final', 'Qualifier 1', 'Qualifier 2', 'Eliminator', 'Elimination Final', '3rd Place Play-Off']
    
    df['is_knockout'] = df['match_type'].apply(lambda x: 1 if x in knockout_types else 0)
    
    df = df.sort_values('date').reset_index(drop=True)
    
    team1_chase_pref = []
    team2_chase_pref = []
    team1_defend_pref = []
    team2_defend_pref = []
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        date = row['date']
        
        t1_matches = df[((df['team1'] == team1) | (df['team2'] == team1)) & (df['date'] < date)]
        t2_matches = df[((df['team1'] == team2) | (df['team2'] == team2)) & (df['date'] < date)]
        
        if len(t1_matches) < 3:
            team1_chase_pref.append(0.5)
            team1_defend_pref.append(0.5)
        else:
            t1_chases = t1_matches[t1_matches['team1'] == team1]
            t1_defends = t1_matches[t1_matches['team2'] == team1]
            
            if len(t1_chases) > 0:
                chase_wins = (t1_chases['winner'] == team1).sum()
                team1_chase_pref.append(chase_wins / len(t1_chases))
            else:
                team1_chase_pref.append(0.5)
            
            if len(t1_defends) > 0:
                defend_wins = (t1_defends['winner'] == team1).sum()
                team1_defend_pref.append(defend_wins / len(t1_defends))
            else:
                team1_defend_pref.append(0.5)
        
        if len(t2_matches) < 3:
            team2_chase_pref.append(0.5)
            team2_defend_pref.append(0.5)
        else:
            t2_chases = t2_matches[t2_matches['team1'] == team2]
            t2_defends = t2_matches[t2_matches['team2'] == team2]
            
            if len(t2_chases) > 0:
                chase_wins = (t2_chases['winner'] == team2).sum()
                team2_chase_pref.append(chase_wins / len(t2_chases))
            else:
                team2_chase_pref.append(0.5)
            
            if len(t2_defends) > 0:
                defend_wins = (t2_defends['winner'] == team2).sum()
                team2_defend_pref.append(defend_wins / len(t2_defends))
            else:
                team2_defend_pref.append(0.5)
    
    df['team1_chase_rate'] = team1_chase_pref
    df['team2_chase_rate'] = team2_chase_pref
    df['team1_defend_rate'] = team1_defend_pref
    df['team2_defend_rate'] = team2_defend_pref
    
    return df


def compute_recent_form(df, n_games=5):
    """Compute recent form (last N games) for each team."""
    df = df.sort_values('date').reset_index(drop=True)
    
    team1_recent = []
    team2_recent = []
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        date = row['date']
        
        t1_past = df[
            ((df['team1'] == team1) | (df['team2'] == team1)) &
            (df['date'] < date)
        ].tail(n_games)
        
        t2_past = df[
            ((df['team1'] == team2) | (df['team2'] == team2)) &
            (df['date'] < date)
        ].tail(n_games)
        
        if len(t1_past) == 0:
            team1_recent.append(0.5)
        else:
            wins = (t1_past['winner'] == team1).sum()
            team1_recent.append(wins / len(t1_past))
        
        if len(t2_past) == 0:
            team2_recent.append(0.5)
        else:
            wins = (t2_past['winner'] == team2).sum()
            team2_recent.append(wins / len(t2_past))
    
    df['team1_recent_form'] = team1_recent
    df['team2_recent_form'] = team2_recent
    df['recent_form_diff'] = df['team1_recent_form'] - df['team2_recent_form']
    
    return df


def compute_home_city_advantage(df):
    """Compute home city advantage for each team."""
    df = df.sort_values('date').reset_index(drop=True)
    
    team_cities = {}
    team1_home = []
    team2_home = []
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        city = row['city']
        
        if team1 not in team_cities:
            team_cities[team1] = {}
        if team2 not in team_cities:
            team_cities[team2] = {}
        
        if city not in team_cities[team1]:
            team_cities[team1][city] = [0, 0]
        if city not in team_cities[team2]:
            team_cities[team2][city] = [0, 0]
        
        t1_home = 1 if city in team_cities[team1] and sum(team_cities[team1][city]) >= 2 else 0
        t2_home = 1 if city in team_cities[team2] and sum(team_cities[team2][city]) >= 2 else 0
        
        team1_home.append(t1_home)
        team2_home.append(t2_home)
        
        if pd.notna(row['winner']):
            winner = row['winner']
            if winner == team1:
                team_cities[team1][city][0] += 1
            else:
                team_cities[team1][city][1] += 1
            
            if winner == team2:
                team_cities[team2][city][0] += 1
            else:
                team_cities[team2][city][1] += 1
    
    df['team1_home'] = team1_home
    df['team2_home'] = team2_home
    df['home_advantage'] = df['team1_home'] - df['team2_home']
    
    return df


df = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

df = df.dropna(subset=['winner'])
df = df[df['result'] != 'no result']

print(f"Dataset shape: {df.shape}")

df['date'] = pd.to_datetime(df['date'])

df['team1_won'] = np.where(df['winner'] == df['team1'], 1, 0)
df['toss_winner_is_team1'] = np.where(df['toss_winner'] == df['team1'], 1, 0)
df['toss_decision_bat'] = np.where(df['toss_decision'] == 'bat', 1, 0)

df['match_type'] = df['match_type'].fillna('T20')
df['match_type'] = df['match_type'].replace({
    'League': 'T20',
    'Qualifier': 'T20',
    'Eliminator': 'T20',
    'Semi Final': 'T20',
    'Final': 'T20'
})

df = compute_historical_ball_features(df, deliveries)
df = compute_player_level_features(df, deliveries)
df = compute_external_features(df)
df = compute_live_features(df, deliveries)
df = compute_player_features(df)
df = compute_toss_impact(df)
df = process_matches_with_elo(df)

df['team1_form'] = df.apply(
    lambda row: df[
        ((df['team1'] == row['team1']) | (df['team2'] == row['team1'])) &
        (df['date'] < row['date'])
    ].tail(10)['winner'].eq(row['team1']).mean() if len(df[df['date'] < row['date']]) > 0 else 0.5,
    axis=1
)
df['team2_form'] = df.apply(
    lambda row: df[
        ((df['team1'] == row['team2']) | (df['team2'] == row['team2'])) &
        (df['date'] < row['date'])
    ].tail(10)['winner'].eq(row['team2']).mean() if len(df[df['date'] < row['date']]) > 0 else 0.5,
    axis=1
)
df['form_diff'] = df['team1_form'] - df['team2_form']


def compute_h2h_features(df):
    """Compute head-to-head win rate for team1 vs team2."""
    df = df.sort_values('date').reset_index(drop=True)
    
    h2h_rates = []
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        date = row['date']
        
        h2h = df[
            (
                ((df['team1'] == team1) & (df['team2'] == team2)) |
                ((df['team1'] == team2) & (df['team2'] == team1))
            ) &
            (df['date'] < date)
        ]
        
        if len(h2h) == 0:
            h2h_rates.append(0.5)
        else:
            wins = (h2h['winner'] == team1).sum()
            h2h_rates.append(wins / len(h2h))
    
    df['h2h_team1_winrate'] = h2h_rates
    return df


def compute_venue_elo(df):
    """Compute venue-specific ELO ratings."""
    df = df.sort_values('date').reset_index(drop=True)
    
    venue_elo = {}
    team1_venue = []
    team2_venue = []
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        venue = row['venue']
        date = row['date']
        
        v1 = venue_elo.get((team1, venue), 1500)
        v2 = venue_elo.get((team2, venue), 1500)
        
        team1_venue.append(v1)
        team2_venue.append(v2)
        
        if pd.notna(row['winner']):
            winner = row['winner']
            if winner == team1:
                loser = team2
            elif winner == team2:
                loser = team1
            else:
                continue
            
            k = 20
            exp = 1 / (1 + 10 ** ((v2 - v1) / 400))
            actual = 1.0 if winner == team1 else 0.0
            
            new_v1 = v1 + k * (actual - exp)
            new_v2 = v2 + k * ((1 - actual) - (1 - exp))
            
            venue_elo[(team1, venue)] = new_v1
            venue_elo[(team2, venue)] = new_v2
    
    df['team1_venue_elo'] = team1_venue
    df['team2_venue_elo'] = team2_venue
    df['venue_elo_diff'] = df['team1_venue_elo'] - df['team2_venue_elo']
    
    return df


df = compute_h2h_features(df)
df = compute_venue_elo(df)
df = compute_streak_and_rest(df)
df = compute_knockout_and_chase(df)
df = compute_recent_form(df, n_games=5)
df = compute_home_city_advantage(df)

df = df.sort_values('date').reset_index(drop=True)

df['season_year'] = pd.to_datetime(df['date']).dt.year


features = [
    'toss_winner_is_team1',
    'toss_decision_bat',
    'team1_elo',
    'team2_elo',
    'elo_diff',
    'expected_team1_win',
    'team1_form',
    'team2_form',
    'form_diff',
    'h2h_team1_winrate',
    'team1_venue_elo',
    'team2_venue_elo',
    'venue_elo_diff',
    'toss_winner_hist_winrate',
    'team1_top_scorer_avg',
    'team2_top_scorer_avg',
    'team1_bowler_wkt_avg',
    'team2_bowler_wkt_avg',
    'team1_boundary_pct',
    'team2_boundary_pct',
    'venue_chase_win_pct',
    'venue_bat_first_win_pct',
    'is_playoffs',
    'team1_powerplay_runs_hist',
    'team2_powerplay_runs_hist',
    'team1_mid_runs_hist',
    'team2_mid_runs_hist'
]

target = 'team1_won'


def train(df, features, target, train_year=2020, n_estimators=100, max_depth=4, random_state=42):
    """
    Train a GradientBoostingClassifier on data before train_year.
    Returns: (model, X_train, y_train, X_test, y_test)
    """
    train_df = df[df['season_year'] < train_year].copy()
    test_df = df[df['season_year'] >= train_year].copy()
    
    X_train = train_df[features].fillna(0.5)
    y_train = train_df[target]
    X_test = test_df[features].fillna(0.5)
    y_test = test_df[target]
    
    model = GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        learning_rate=0.05,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    return model, X_train, y_train, X_test, y_test


def test(model, X_test, y_test):
    """
    Evaluate model on test set.
    Returns: accuracy score and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report, y_pred


if __name__ == '__main__':
    print("\n=== Testing different models with tuning ===")
    best_acc = 0
    best_year = 2020
    best_model_name = ""
    
    train_df_full = df[df['season_year'] < 2021]
    test_df_full = df[df['season_year'] >= 2021]
    
    X_train = train_df_full[features].fillna(0.5)
    y_train = train_df_full[target]
    X_test = test_df_full[features].fillna(0.5)
    y_test = test_df_full[target]
    
    print("\n1. Testing GradientBoosting with different params...")
    gb_configs = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.03},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.02},
    ]
    
    for config in gb_configs:
        gb = GradientBoostingClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            random_state=42
        )
        gb.fit(X_train, y_train)
        acc = accuracy_score(y_test, gb.predict(X_test))
        print(f"  GB (n={config['n_estimators']}, d={config['max_depth']}, lr={config['learning_rate']}): {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_name = f"GB (n={config['n_estimators']}, d={config['max_depth']}, lr={config['learning_rate']})"
    
    print("\n2. Testing XGBoost with different params...")
    xgb_configs = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.03},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.02},
    ]
    
    for config in xgb_configs:
        xgb_model = xgb.XGBClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        acc = accuracy_score(y_test, xgb_model.predict(X_test))
        print(f"  XGB (n={config['n_estimators']}, d={config['max_depth']}, lr={config['learning_rate']}): {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_name = f"XGB (n={config['n_estimators']}, d={config['max_depth']}, lr={config['learning_rate']})"
    
    print("\n3. Testing RandomForest with different params...")
    rf_configs = [
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 150, 'max_depth': 8},
        {'n_estimators': 200, 'max_depth': 10},
    ]
    
    for config in rf_configs:
        rf = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=42
        )
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"  RF (n={config['n_estimators']}, d={config['max_depth']}): {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model_name = f"RF (n={config['n_estimators']}, d={config['max_depth']})"
    
    print("\n4. Testing Ensemble (soft voting)...")
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42)
    xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    
    from sklearn.ensemble import VotingClassifier
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('xgb', xgb_m), ('rf', rf)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    acc = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"  Ensemble: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model_name = "Ensemble"
    
    baseline = (X_test['expected_team1_win'] > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, baseline)
    print(f"\n  Baseline (ELO): {baseline_acc:.4f}")
    
    print(f"\n=== Best Model: {best_model_name} ({best_acc:.4f}) ===")
    
    print(f"\n=== Final: train < {best_year} ({best_acc:.4f}) ===")
    
    train_df = df[df['season_year'] < best_year].copy()
    test_df = df[df['season_year'] >= best_year].copy()
    
    X_train = train_df[features].fillna(0.5)
    y_train = train_df[target]
    X_test = test_df[features].fillna(0.5)
    y_test = test_df[target]
    
    model = GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    predictions = y_pred
    
    baseline_pred = (X_test['expected_team1_win'] > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    
    baseline_pred = (X_test['expected_team1_win'] > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"\n=== RESULTS ===")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Baseline (ELO only): {baseline_acc:.4f}")
    print("\nClassification Report:")
    print(report)

    print("\nFeature Importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
