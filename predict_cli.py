#!/usr/bin/env python3
"""
IPL Match Predictor - Interactive CLI
Simple clickable-style menu for match predictions
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb

from ipl_data import (
    TEAMS, VENUES, IPL_2026_SCHEDULE, 
    get_team_short, get_venue_full_name
)


class IPLPredictor:
    """IPL Match Predictor with ML model."""
    
    def __init__(self):
        self.df = None
        self.model = None
        self.features = None
        print("Initializing IPL Predictor...")
    
    def load_data(self):
        """Load and prepare data."""
        print("Loading match data...")
        self.df = pd.read_csv("matches.csv")
        self.df = self.df.dropna(subset=['winner'])
        self.df = self.df[self.df['result'] != 'no result']
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print("Computing features...")
        self.compute_features()
        
        print("Training model...")
        self.features = [
            'toss_winner_is_team1', 'toss_decision_bat', 'team1_elo', 'team2_elo',
            'elo_diff', 'expected_team1_win', 'team1_form', 'team2_form', 'form_diff',
            'h2h_team1_winrate'
        ]
        
        train_df = self.df[self.df['date'].dt.year < 2021]
        X_train = train_df[self.features].fillna(0.5)
        y_train = train_df['team1_won']
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("Ready!")
    
    def compute_features(self):
        """Compute features."""
        self.df['team1_won'] = np.where(self.df['winner'] == self.df['team1'], 1, 0)
        self.df['toss_winner_is_team1'] = np.where(self.df['toss_winner'] == self.df['team1'], 1, 0)
        self.df['toss_decision_bat'] = np.where(self.df['toss_decision'] == 'bat', 1, 0)
        
        default_elo = 1500
        k_factor = 20
        ratings = {}
        team1_elo = []
        team2_elo = []
        
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        for idx, row in self.df.iterrows():
            team1 = row['team1']
            team2 = row['team2']
            
            elo1 = ratings.get(team1, default_elo)
            elo2 = ratings.get(team2, default_elo)
            
            team1_elo.append(elo1)
            team2_elo.append(elo2)
            
            if pd.notna(row['winner']):
                winner = row['winner']
                loser = team2 if winner == team1 else team1
                
                expected = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
                new_elo1 = elo1 + k_factor * (1 - expected)
                new_elo2 = elo2 + k_factor * (0 - (1 - expected))
                
                ratings[team1] = new_elo1
                ratings[team2] = new_elo2
        
        self.df['team1_elo'] = team1_elo
        self.df['team2_elo'] = team2_elo
        self.df['elo_diff'] = self.df['team1_elo'] - self.df['team2_elo']
        self.df['expected_team1_win'] = self.df.apply(
            lambda r: 1 / (1 + 10 ** ((r['team2_elo'] - r['team1_elo']) / 400)), axis=1
        )
        
        def get_form(team, date):
            past = self.df[
                ((self.df['team1'] == team) | (self.df['team2'] == team)) &
                (self.df['date'] < date)
            ].tail(10)
            if len(past) == 0:
                return 0.5
            return (past['winner'] == team).sum() / len(past)
        
        self.df['team1_form'] = self.df.apply(lambda r: get_form(r['team1'], r['date']), axis=1)
        self.df['team2_form'] = self.df.apply(lambda r: get_form(r['team2'], r['date']), axis=1)
        self.df['form_diff'] = self.df['team1_form'] - self.df['team2_form']
        
        def get_h2h(t1, t2, date):
            h2h = self.df[
                ((self.df['team1'] == t1) & (self.df['team2'] == t2)) |
                ((self.df['team1'] == t2) & (self.df['team2'] == t1))
            ]
            h2h = h2h[h2h['date'] < date]
            if len(h2h) == 0:
                return 0.5
            return (h2h['winner'] == t1).sum() / len(h2h)
        
        self.df['h2h_team1_winrate'] = self.df.apply(lambda r: get_h2h(r['team1'], r['team2'], r['date']), axis=1)
    
    def get_team_elo(self, team, match_date):
        past = self.df[
            ((self.df['team1'] == team) | (self.df['team2'] == team)) &
            (self.df['date'] < match_date)
        ]
        
        if len(past) == 0:
            return 1500
        
        if team == past.iloc[-1]['team1']:
            return past.iloc[-1]['team1_elo']
        else:
            return past.iloc[-1]['team2_elo']
    
    def get_team_form(self, team, match_date):
        past = self.df[
            ((self.df['team1'] == team) | (self.df['team2'] == team)) &
            (self.df['date'] < match_date)
        ].tail(10)
        
        if len(past) == 0:
            return 0.5
        
        return (past['winner'] == team).sum() / len(past)
    
    def get_h2h(self, team1, team2, match_date):
        h2h = self.df[
            ((self.df['team1'] == team1) & (self.df['team2'] == team2)) |
            ((self.df['team1'] == team2) & (self.df['team2'] == team1))
        ]
        h2h = h2h[h2h['date'] < match_date]
        
        if len(h2h) == 0:
            return 0.5
        
        return (h2h['winner'] == team1).sum() / len(h2h)
    
    def predict(self, match):
        """Predict a match."""
        home = match["home"]
        away = match["away"]
        venue = get_venue_full_name(match["venue"])
        match_date = datetime.strptime(match["date"], "%Y-%m-%d")
        
        elo1 = self.get_team_elo(home, match_date)
        elo2 = self.get_team_elo(away, match_date)
        
        form1 = self.get_team_form(home, match_date)
        form2 = self.get_team_form(away, match_date)
        
        h2h = self.get_h2h(home, away, match_date)
        
        expected_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        
        match_data = {
            'toss_winner_is_team1': 0.5,
            'toss_decision_bat': 0.5,
            'team1_elo': elo1,
            'team2_elo': elo2,
            'elo_diff': elo1 - elo2,
            'expected_team1_win': expected_elo,
            'team1_form': form1,
            'team2_form': form2,
            'form_diff': form1 - form2,
            'h2h_team1_winrate': h2h
        }
        
        X_pred = pd.DataFrame([match_data])[self.features].fillna(0.5)
        ml_prob = self.model.predict_proba(X_pred)[0]
        
        home_ml = ml_prob[1]
        away_ml = ml_prob[0]
        
        ml_winner = home if home_ml > away_ml else away
        
        return {
            'home': home,
            'away': away,
            'venue': venue,
            'date': match['date'],
            'home_prob': home_ml * 100,
            'away_prob': away_ml * 100,
            'winner': ml_winner,
            'home_elo': elo1,
            'away_elo': elo2,
            'home_form': form1 * 100,
            'away_form': form2 * 100,
            'h2h': h2h * 100
        }


def print_header():
    """Print header."""
    print("\n" + "="*60)
    print("       IPL 2026 MATCH PREDICTOR")
    print("="*60)


def print_prediction(pred):
    """Print prediction result."""
    print("\n" + "-"*50)
    print(f"MATCH: {pred['home']} vs {pred['away']}")
    print(f"DATE:  {pred['date']}")
    print(f"VENUE: {pred['venue']}")
    print("-"*50)
    print()
    print(f"  {get_team_short(pred['home']):>3} Win Probability: {pred['home_prob']:.1f}%")
    print(f"  {get_team_short(pred['away']):>3} Win Probability: {pred['away_prob']:.1f}%")
    print()
    print(f"  >>> PREDICTED WINNER: {pred['winner']} <<<")
    print()
    print("-"*50)
    print(f"  ELO Ratings: {get_team_short(pred['home'])}={pred['home_elo']:.0f} | {get_team_short(pred['away'])}={pred['away_elo']:.0f}")
    print(f"  Form:        {get_team_short(pred['home'])}={pred['home_form']:.0f}% | {get_team_short(pred['away'])}={pred['away_form']:.0f}%")
    print(f"  H2H:         {get_team_short(pred['home'])} leads with {pred['h2h']:.0f}%")
    print("-"*50)


def main():
    """Main function."""
    print_header()
    
    predictor = IPLPredictor()
    predictor.load_data()
    
    today = datetime.now()
    upcoming = [m for m in IPL_2026_SCHEDULE if datetime.strptime(m["date"], "%Y-%m-%d") >= today]
    
    while True:
        print_header()
        print("\n=== MAIN MENU ===")
        print("\n[1] Predict a Match")
        print("[2] Today's Match")
        print("[3] Team Rankings")
        print("[4] Recent Results")
        print("[5] Exit")
        
        menu_choice = input("\nSelect option: ").strip()
        
        if menu_choice == '5' or menu_choice.lower() == 'q':
            print("\nThanks for using IPL Predictor!")
            break
        
        elif menu_choice == '1':
            # Original match prediction
            print_header()
            print("\n=== UPCOMING MATCHES ===\n")
            
            for idx, match in enumerate(upcoming[:20]):
                match_date = datetime.strptime(match["date"], "%Y-%m-%d")
                date_str = match_date.strftime("%b %d")
                home = get_team_short(match["home"])
                away = get_team_short(match["away"])
                venue = get_venue_full_name(match["venue"])[:20]
                
                print(f"  [{idx+1:>2}] {date_str} | {home:>3} vs {away:<3} | {venue}")
            
            print("\n" + "-"*60)
            choice = input("Select match number to predict (or 'b' for back): ").strip()
            
            if choice.lower() == 'b':
                continue
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(upcoming):
                    pred = predictor.predict(upcoming[idx])
                    print_prediction(pred)
                    input("\nPress Enter to continue...")
                else:
                    print("Invalid selection!")
            except ValueError:
                print("Please enter a valid number!")
        
        elif menu_choice == '2':
            # Today's match
            today_str = today.strftime("%Y-%m-%d")
            today_matches = [m for m in upcoming if m["date"] == today_str]
            
            if today_matches:
                for m in today_matches:
                    pred = predictor.predict(m)
                    print_prediction(pred)
            else:
                print("\nNo match scheduled for today!")
                print(f"Next match: {upcoming[0]['date']}")
            
            input("\nPress Enter to continue...")
        
        elif menu_choice == '3':
            # Team rankings
            print("\n=== TEAM RANKINGS ===")
            print("(Based on ELO ratings)")
            
            # Get current ELO for all teams
            teams_elo = {}
            for match in upcoming:
                for team in [match["home"], match["away"]]:
                    if team not in teams_elo:
                        elo = predictor.get_team_elo(team, datetime.strptime(match["date"], "%Y-%m-%d"))
                        teams_elo[team] = elo
            
            sorted_teams = sorted(teams_elo.items(), key=lambda x: x[1], reverse=True)
            
            print("\n" + "-"*40)
            print(f"{'Rank':<6}{'Team':<25}{'ELO':>8}")
            print("-"*40)
            for i, (team, elo) in enumerate(sorted_teams, 1):
                print(f"{i:<6}{get_team_short(team):<25}{elo:>8.0f}")
            
            input("\nPress Enter to continue...")
        
        elif menu_choice == '4':
            # Recent results
            print("\n=== RECENT RESULTS ===")
            recent = predictor.df.tail(10)
            
            for _, row in recent.iterrows():
                date_str = row['date'].strftime("%b %d")
                winner = get_team_short(row['winner'])
                t1 = get_team_short(row['team1'])
                t2 = get_team_short(row['team2'])
                result = f"{winner} won"
                print(f"  {date_str} | {t1} vs {t2} | {result}")
            
            input("\nPress Enter to continue...")
        
        else:
            print("Invalid option!")


if __name__ == "__main__":
    main()
