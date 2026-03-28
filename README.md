# Cricket Match Prediction AI

A machine learning model to predict IPL (Indian Premier League) T20 cricket match outcomes.

## Features

- **ELO Rating System**: Format-specific team ratings
- **Venue ELO**: Team performance at specific venues
- **Player Impact**: Historical player of match impact
- **Toss Analysis**: Toss winner historical win rate
- **External Features**: Venue patterns, playoffs

## Installation

```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage

```bash
python cricket_ai.py
```

## Results

- **Best Accuracy**: 56.1% (XGBoost)
- **Baseline (ELO only)**: 52.9%
- **Improvement**: +3.2%

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

## License

MIT
