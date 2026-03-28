#!/usr/bin/env python3
"""
IPL Match Predictor - Textual TUI
Interactive terminal UI for IPL match predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button, DataTable, Label
from textual.events import Mount
from textual import work
import xgboost as xgb

from ipl_data import (
    TEAMS, VENUES, IPL_2026_SCHEDULE, 
    get_team_short, get_venue_full_name,
    get_today_match, get_upcoming_matches
)


class MatchRow(Static):
    """A clickable match row."""
    
    def __init__(self, match_data, **kwargs):
        super().__init__(**kwargs)
        self.match_data = match_data
        self.home_short = get_team_short(match_data["home"])
        self.away_short = get_team_short(match_data["away"])
        self.venue_name = get_venue_full_name(match_data["venue"])
    
    def compose(self) -> ComposeResult:
        date_obj = datetime.strptime(self.match_data["date"], "%Y-%m-%d")
        date_str = date_obj.strftime("%b %d")
        
        yield Label(
            f"[{date_str}] {self.home_short} vs {self.away_short} @ {self.venue_name[:20]}",
            classes="match-label"
        )


class MatchPredictorApp(App):
    """Main application."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #header-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        height: 3;
    }
    
    #main-container {
        height: 100%;
    }
    
    .match-row {
        height: auto;
        padding: 1 2;
        border: solid $surface-lighten-1;
        margin: 1 0;
        background: $surface-darken-1;
    }
    
    .match-row:hover {
        background: $accent-darken-1;
    }
    
    #matches-container {
        height: 40%;
        border: solid $primary;
        margin: 1 2;
    }
    
    #prediction-panel {
        height: 60%;
        border: solid $success;
        margin: 1 2;
        padding: 2;
    }
    
    #prediction-title {
        text-style: bold;
        color: $success;
        height: 3;
    }
    
    .pred-label {
        height: auto;
        padding: 1;
    }
    
    .team-name {
        text-style: bold;
        color: $text;
    }
    
    .prob-high {
        color: $success;
    }
    
    .winner {
        color: $warning;
        text-style: bold;
    }
    
    #status-bar {
        dock: bottom;
        height: 1;
        background: $accent;
        color: $text;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("t", "today", "Today"),
    ]
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.model = None
        self.features = None
        self.selected_match = None
        self.current_prediction = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="main-container"):
            with Vertical(id="matches-section"):
                yield Static("IPL 2026 SCHEDULE - Press Enter to Predict", id="header-title", classes="header")
                with ScrollableContainer(id="matches-container"):
                    yield Container(id="matches-list")
            
            with Vertical(id="prediction-panel"):
                yield Static("PREDICTION", id="prediction-title")
                yield Container(id="prediction-content")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize data on mount."""
        self.loading = True
        self.status = "Loading data..."
        
        @work
        def load_data():
            self.df = pd.read_csv("matches.csv")
            self.df = self.df.dropna(subset=['winner'])
            self.df = self.df[df['result'] != 'no result']
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Compute basic features
            self.compute_features()
            
            # Train model
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
            
            # Load matches
            self.load_matches()
            
            self.loading = False
            self.status = "Ready - Press Enter on a match to predict"
        
        load_data()
    
    def compute_features(self):
        """Compute basic features."""
        self.df['team1_won'] = np.where(self.df['winner'] == self.df['team1'], 1, 0)
        self.df['toss_winner_is_team1'] = np.where(self.df['toss_winner'] == self.df['team1'], 1, 0)
        self.df['toss_decision_bat'] = np.where(self.df['toss_decision'] == 'bat', 1, 0)
        
        # ELO ratings
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
        
        # Form
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
        
        # H2H
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
    
    def load_matches(self):
        """Load matches into the UI."""
        matches_list = self.query_one("#matches-list")
        matches_list.remove_children()
        
        today = datetime.now()
        
        for idx, match in enumerate(IPL_2026_SCHEDULE):
            match_date = datetime.strptime(match["date"], "%Y-%m-%d")
            
            if match_date >= today:
                date_str = match_date.strftime("%b %d")
                home = get_team_short(match["home"])
                away = get_team_short(match["away"])
                venue = get_venue_full_name(match["venue"])[:15]
                
                btn = Button(
                    f"[{date_str}] {home:>3} vs {away:<3} @ {venue}",
                    id=f"match_{idx}",
                    variant="primary"
                )
                btn.match_data = match
                btn.idx = idx
                
                matches_list.mount(btn)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle match selection."""
        if hasattr(event.button, 'match_data'):
            self.selected_match = event.button.match_data
            self.predict_match(self.selected_match)
    
    def predict_match(self, match):
        """Predict a match."""
        home = match["home"]
        away = match["away"]
        venue = get_venue_full_name(match["venue"])
        match_date = datetime.strptime(match["date"], "%Y-%m-%d")
        
        # Get ELO
        elo1 = self.get_team_elo(home, match_date)
        elo2 = self.get_team_elo(away, match_date)
        
        # Get form
        form1 = self.get_team_form(home, match_date)
        form2 = self.get_team_form(away, match_date)
        
        # Get H2H
        h2h = self.get_h2h(home, away, match_date)
        
        # Calculate prediction
        expected_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        
        # Simple weighted prediction
        score = (expected_elo * 0.5) + (form1 * 0.25) + (form2 * 0.15) + (h2h * 0.1)
        
        if score > 0.5:
            winner = home
            prob_home = score
            prob_away = 1 - score
        else:
            winner = away
            prob_away = 1 - score
            prob_home = score
        
        # ML model prediction
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
        
        # Display prediction
        content = self.query_one("#prediction-content")
        content.remove_children()
        
        content.mount(Label(f"MATCH: {home} vs {away}", classes="pred-label team-name"))
        content.mount(Label(f"DATE: {match['date']}", classes="pred-label"))
        content.mount(Label(f"VENUE: {venue}", classes="pred-label"))
        content.mount(Label("", classes="pred-label"))
        content.mount(Label(f"--- PREDICTION ---", classes="pred-label team-name"))
        content.mount(Label(f"{get_team_short(home)} Win Probability: {home_ml*100:.1f}%", classes="pred-label"))
        content.mount(Label(f"{get_team_short(away)} Win Probability: {away_ml*100:.1f}%", classes="pred-label"))
        content.mount(Label("", classes="pred-label"))
        
        ml_winner = home if home_ml > away_ml else away
        content.mount(Label(f"ML PREDICTED WINNER: {get_team_short(ml_winner)}", classes="pred-label winner"))
        
        content.mount(Label("", classes="pred-label"))
        content.mount(Label(f"--- STATS ---", classes="pred-label team-name"))
        content.mount(Label(f"ELO: {home}={elo1:.0f}, {away}={elo2:.0f}", classes="pred-label"))
        content.mount(Label(f"Form: {home}={form1*100:.0f}%, {away}={form2*100:.0f}%", classes="pred-label"))
        content.mount(Label(f"H2H: {home} leads {h2h*100:.0f}%", classes="pred-label"))
    
    def get_team_elo(self, team, match_date):
        """Get team ELO at match date."""
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
        """Get team form."""
        past = self.df[
            ((self.df['team1'] == team) | (self.df['team2'] == team)) &
            (self.df['date'] < match_date)
        ].tail(10)
        
        if len(past) == 0:
            return 0.5
        
        return (past['winner'] == team).sum() / len(past)
    
    def get_h2h(self, team1, team2, match_date):
        """Get head-to-head record."""
        h2h = self.df[
            ((self.df['team1'] == team1) & (self.df['team2'] == team2)) |
            ((self.df['team1'] == team2) & (self.df['team2'] == team1))
        ]
        h2h = h2h[h2h['date'] < match_date]
        
        if len(h2h) == 0:
            return 0.5
        
        return (h2h['winner'] == team1).sum() / len(h2h)
    
    def action_refresh(self):
        """Refresh matches."""
        self.load_matches()
    
    def action_today(self):
        """Show today's match."""
        today = datetime.now()
        for idx, match in enumerate(IPL_2026_SCHEDULE):
            match_date = datetime.strptime(match["date"], "%Y-%m-%d")
            if match_date.date() == today.date():
                self.predict_match(match)
                break


if __name__ == "__main__":
    app = MatchPredictorApp()
    app.run()
