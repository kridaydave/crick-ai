#!/usr/bin/env python3
"""
Cricket Match Prediction TUI
Interactive terminal user interface for IPL match predictions
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box

console = Console()


def load_and_prepare_data():
    """Load and prepare data."""
    df = pd.read_csv("matches.csv")
    
    df = df.dropna(subset=['winner'])
    df = df[df['result'] != 'no result']
    df['date'] = pd.to_datetime(df['date'])
    df['season_year'] = df['date'].dt.year
    
    df['team1_won'] = np.where(df['winner'] == df['team1'], 1, 0)
    
    console.print("[dim]Computing ELO ratings...[/dim]")
    df = compute_elo_ratings(df)
    
    return df


def compute_elo_ratings(df):
    """Compute simple ELO ratings."""
    df = df.sort_values('date').reset_index(drop=True)
    
    default_elo = 1500
    k_factor = 20
    
    ratings = {}
    team1_elo = []
    team2_elo = []
    
    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        
        elo1 = ratings.get(team1, default_elo)
        elo2 = ratings.get(team2, default_elo)
        
        team1_elo.append(elo1)
        team2_elo.append(elo2)
        
        if pd.notna(row['winner']):
            winner = row['winner']
            
            if winner == team1:
                loser = team2
            elif winner == team2:
                loser = team1
            else:
                continue
            
            expected = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
            
            new_elo1 = elo1 + k_factor * (1 - expected)
            new_elo2 = elo2 + k_factor * (0 - (1 - expected))
            
            ratings[team1] = new_elo1
            ratings[team2] = new_elo2
    
    df['team1_elo'] = team1_elo
    df['team2_elo'] = team2_elo
    
    return df


def get_teams(df):
    """Get list of all teams."""
    teams = sorted(pd.concat([df['team1'], df['team2']]).unique())
    return teams


def get_venues(df):
    """Get list of all venues."""
    venues = sorted(df['venue'].unique())
    return venues


def show_header():
    """Display header."""
    console.print()
    console.print(Panel.fit(
        Text("IPL CRICKET MATCH PREDICTOR", justify="center", style="bold cyan"),
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()


def show_team_stats(df, team):
    """Show team statistics."""
    team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
    wins = (team_matches['winner'] == team).sum()
    total = len(team_matches)
    win_rate = (wins / total * 100) if total > 0 else 0
    
    return {
        'matches': total,
        'wins': wins,
        'win_rate': round(win_rate, 1)
    }


def get_team_elo(df, team, match_date):
    """Get team ELO at a given date."""
    past_matches = df[(df['date'] < match_date) & ((df['team1'] == team) | (df['team2'] == team))]
    if len(past_matches) == 0:
        return 1500
    
    if team == past_matches.iloc[-1]['team1']:
        return past_matches.iloc[-1]['team1_elo']
    else:
        return past_matches.iloc[-1]['team2_elo']


def get_team_form(df, team, match_date):
    """Get team form (last 10 matches)."""
    past = df[(df['date'] < match_date) & ((df['team1'] == team) | (df['team2'] == team))].tail(10)
    if len(past) == 0:
        return 0.5
    return (past['winner'] == team).sum() / len(past)


def get_h2h(df, team1, team2, match_date):
    """Get head-to-head record."""
    h2h = df[
        ((df['team1'] == team1) & (df['team2'] == team2)) |
        ((df['team1'] == team2) & (df['team2'] == team1))
    ]
    h2h = h2h[h2h['date'] < match_date]
    if len(h2h) == 0:
        return 0.5
    return (h2h['winner'] == team1).sum() / len(h2h)


def predict_simple(df, team1, team2, venue, toss_winner, toss_decision, match_date):
    """Simple prediction based on ELO and form."""
    elo1 = get_team_elo(df, team1, match_date)
    elo2 = get_team_elo(df, team2, match_date)
    
    form1 = get_team_form(df, team1, match_date)
    form2 = get_team_form(df, team2, match_date)
    
    h2h_rate = get_h2h(df, team1, team2, match_date)
    
    expected_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    score = (expected_elo * 0.5) + (form1 * 0.25) + (form2 * 0.15) + (h2h_rate * 0.1)
    
    if score > 0.5:
        winner = team1
        prob1 = score * 100
        prob2 = (1 - score) * 100
    else:
        winner = team2
        prob2 = (1 - score) * 100
        prob1 = score * 100
    
    return {
        'predicted_winner': winner,
        'team1_win_probability': round(prob1, 1),
        'team2_win_probability': round(prob2, 1),
        'team1_elo': round(elo1, 1),
        'team2_elo': round(elo2, 1),
        'team1_form': round(form1 * 100, 1),
        'team2_form': round(form2 * 100, 1),
        'h2h_team1': round(h2h_rate * 100, 1),
        'team1': team1,
        'team2': team2
    }


def display_prediction(result):
    """Display prediction in a nice table."""
    console.print()
    
    prob_table = Table(box=box.ROUNDED, show_header=False)
    prob_table.add_column("Team", style="cyan bold")
    prob_table.add_column("Win Probability", justify="center", style="green bold")
    prob_table.add_column("ELO Rating", justify="center")
    prob_table.add_column("Form %", justify="center")
    
    prob_table.add_row(
        result['team1'],
        f"{result['team1_win_probability']}%",
        str(result['team1_elo']),
        f"{result['team1_form']}%"
    )
    prob_table.add_row(
        result['team2'],
        f"{result['team2_win_probability']}%",
        str(result['team2_elo']),
        f"{result['team2_form']}%"
    )
    
    console.print(Panel.fit(
        prob_table,
        title="[bold]Match Prediction[/bold]",
        border_style="green"
    ))
    
    console.print()
    
    winner_text = Text(f"PREDICTED WINNER: {result['predicted_winner']}", style="bold yellow")
    console.print(Panel.fit(winner_text, border_style="yellow", box=box.DOUBLE))
    
    console.print()
    console.print(f"[dim]Head-to-Head Win Rate ({result['team1']}): {result['h2h_team1']}%[/dim]")


def display_team_rankings(df):
    """Display team rankings."""
    teams = get_teams(df)
    
    rankings = []
    for team in teams:
        stats = show_team_stats(df, team)
        current_elo = df[df['team1'] == team]['team1_elo'].iloc[-1] if len(df[df['team1'] == team]) > 0 else 1500
        rankings.append({
            'Team': team,
            'Matches': stats['matches'],
            'Wins': stats['wins'],
            'Win Rate': stats['win_rate'],
            'ELO': round(current_elo, 0)
        })
    
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values('ELO', ascending=False)
    
    table = Table(title="Team Rankings (by ELO)", box=box.ROUNDED)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Team", style="bold")
    table.add_column("Matches", justify="center")
    table.add_column("Wins", justify="center")
    table.add_column("Win Rate", justify="center", style="green")
    table.add_column("ELO", justify="center", style="yellow")
    
    for idx, row in rankings_df.head(10).iterrows():
        table.add_row(
            str(len(table.rows) + 1),
            row['Team'],
            str(row['Matches']),
            str(row['Wins']),
            f"{row['Win Rate']}%",
            str(int(row['ELO']))
        )
    
    console.print(table)


def main():
    """Main TUI loop."""
    show_header()
    
    console.print("[yellow]Loading data...[/yellow]")
    df = load_and_prepare_data()
    
    console.print("[green]Data loaded successfully![/green]")
    console.print(f"[dim]Total matches: {len(df)}[/dim]")
    console.print(f"[dim]Date range: {df['date'].min().strftime('%Y')} - {df['date'].max().strftime('%Y')}[/dim]")
    console.print()
    
    teams = get_teams(df)
    venues = get_venues(df)
    
    while True:
        console.print(Panel.fit(
            Text("MAIN MENU", justify="center", style="bold cyan"),
            border_style="cyan"
        ))
        console.print("[1] Predict a Match")
        console.print("[2] View Team Rankings")
        console.print("[3] View Recent Matches")
        console.print("[4] Exit")
        
        choice = Prompt.ask("[bold cyan]Select option[/bold cyan]", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            console.print()
            
            console.print(Panel.fit(
                Text("Match Details", justify="center", style="bold green"),
                border_style="green"
            ))
            
            team1_idx = Prompt.ask(
                "[bold]Select Team 1 (Home Team)[/bold]\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(teams[:15])]),
                choices=[str(i+1) for i in range(min(15, len(teams)))],
                default="1"
            )
            team1 = teams[int(team1_idx) - 1]
            
            remaining_teams = [t for t in teams if t != team1]
            team2_idx = Prompt.ask(
                f"[bold]Select Team 2 (Away Team)[/bold]\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(remaining_teams[:15])]),
                choices=[str(i+1) for i in range(min(15, len(remaining_teams)))],
                default="1"
            )
            team2 = remaining_teams[int(team2_idx) - 1]
            
            venue_idx = Prompt.ask(
                "[bold]Select Venue[/bold]\n" + "\n".join([f"{i+1}. {v[:40]}" for i, v in enumerate(venues[:10])]),
                choices=[str(i+1) for i in range(min(10, len(venues)))],
                default="1"
            )
            venue = venues[int(venue_idx) - 1]
            
            toss_winner = Prompt.ask(
                f"Who won the toss? ({team1} or {team2})",
                default=team1
            )
            
            toss_decision = Prompt.ask(
                "Toss decision (bat/field)",
                default="field"
            )
            
            match_date = Prompt.ask(
                "[bold]Match date (YYYY-MM-DD)[/bold]",
                default="2024-04-15"
            )
            
            match_date = pd.to_datetime(match_date)
            
            console.print()
            console.print("[yellow]Predicting...[/yellow]")
            
            result = predict_simple(
                df=df,
                team1=team1,
                team2=team2,
                venue=venue,
                toss_winner=toss_winner,
                toss_decision=toss_decision,
                match_date=match_date
            )
            
            display_prediction(result)
            
        elif choice == "2":
            console.print()
            display_team_rankings(df)
            
        elif choice == "3":
            console.print()
            recent = df.tail(10)[['date', 'team1', 'team2', 'winner', 'venue']]
            
            table = Table(title="Recent Matches", box=box.ROUNDED)
            table.add_column("Date", style="cyan")
            table.add_column("Team 1", style="white")
            table.add_column("Team 2", style="white")
            table.add_column("Winner", style="green bold")
            
            for _, row in recent.iterrows():
                table.add_row(
                    row['date'].strftime('%Y-%m-%d'),
                    row['team1'][:20],
                    row['team2'][:20],
                    row['winner'][:20]
                )
            
            console.print(table)
            
        elif choice == "4":
            console.print()
            console.print(Panel.fit(
                Text("Thanks for using IPL Predictor!", justify="center", style="bold green"),
                border_style="green"
            ))
            break
        
        console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
