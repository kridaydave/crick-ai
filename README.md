# Cricket Match Prediction AI

A machine learning model to predict IPL (Indian Premier League) T20 cricket match outcomes.

## Features

- **ELO Rating System**: Format-specific team ratings with margin weighting
- **Venue ELO**: Team performance at specific venues
- **Player Impact**: Historical player of match impact
- **Toss Analysis**: Toss winner historical win rate
- **External Features**: Venue patterns, playoffs
- **Ball-by-Ball Analysis**: Powerplay, death over performance
- **Player-Level Features**: Top scorer average, bowler wickets, boundary %

## Installation

```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage

### Run the model
```bash
python cricket_ai.py
```

### Predict a match

```python
from cricket_ai import predict_match, df, deliveries, model, features

result = predict_match(
    team1='Mumbai Indians',
    team2='Chennai Super Kings',
    venue='Wankhede Stadium',
    city='Mumbai',
    toss_winner='Mumbai Indians',
    toss_decision='field',
    match_date='2024-04-15',
    df=df,
    deliveries=deliveries,
    model=model,
    features=features
)

print(f"Winner: {result['predicted_winner']}")
print(f"MI: {result['team1_win_probability']}%")
print(f"CSK: {result['team2_win_probability']}%")
```

## Results

| Metric | Value |
|--------|-------|
| **Best Accuracy** | **56.1%** (XGBoost) |
| Baseline (ELO only) | 52.9% |
| Improvement | +3.2% |

## Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 27 features including ELO, form, venue, player data
- **Training Data**: Matches before 2021
- **Test Data**: Matches from 2021 onwards

## Dataset

- `matches.csv`: Match-level data (1090 matches, 2008-2024)
- `deliveries.csv`: Ball-by-ball data (260k+ deliveries)

## Key Features

| Feature | Importance |
|---------|------------|
| Team ELO | 15-20% |
| Venue ELO | 10-15% |
| Player Impact | 5-8% |
| Toss Winner Rate | 5-6% |
| Boundary % | 5-6% |

## Prediction Example

```
Example: Predicting MI vs CSK at Wankhede

PREDICTION RESULT
----------------------------------------
Predicted Winner: Chennai Super Kings
Mumbai Indians Win Probability: 37.0%
Chennai Super Kings Win Probability: 63.0%
Team1 ELO: 1471.6
Team2 ELO: 1528.4
Team1 Form: 40.0%
Team2 Form: 70.0%
H2H Win Rate (Team1): 54.1%
```

## License

MIT
