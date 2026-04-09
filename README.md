#Bayesian Sports Prediction System

## Overview
This project implements a **Bayesian framework for predicting competitive sports outcomes**, based on a graduate capstone research study.

The system combines:
- Bayesian inference (Bradley–Terry model)
- Dynamic skill tracking (Kalman-style updates)
- Machine learning (Logistic Regression + XGBoost ensemble)
- Feature engineering with strict temporal integrity (no data leakage)

It is applied to:
- FIFA World Cup matches (historical graph-based model)
- UFC fights (dynamic, feature-rich prediction system)

---

## Key Features

- **Bayesian Skill Modeling**
  - Probabilistic representation of team/fighter strength
  - Uncertainty-aware predictions

- **Dynamic Skill Updates**
  - Skill evolves over time using Bayesian updates
  - Accounts for inactivity and performance changes

- **Machine Learning Ensemble**
  - Logistic Regression (calibration)
  - XGBoost (non-linear patterns)
  - Combined for improved accuracy

- **Leakage-Free Pipeline**
  - Strict chronological feature construction
  - Prevents future data contamination

- **Multi-Outcome Prediction**
  - Predicts:
    - Win probability
    - Method of victory (KO/TKO, Submission, Decision)

---

## Results

### UFC Model Performance
- Accuracy: **65.7%**
- 95% CI: **[63.3%, 68.1%]**
- Brier Score: **0.210**
- Expected Calibration Error (ECE): **0.056**

### Key Findings
- Bayesian models outperform traditional baselines
- Data leakage can inflate accuracy by **+6.7%**
- Physical attributes (especially age) are highly predictive
- Model is well-calibrated but slightly overconfident in close matchups

---

## Methodology

### 1. Bayesian Modeling
- Uses the **Bradley–Terry model** for pairwise comparisons
- Maintains distributions over skill (mean + variance)

### 2. Dynamic Updating
- Skill evolves using a **Kalman-style Bayesian update**
- Uncertainty increases during inactivity

### 3. Feature Engineering
- 21 engineered features, including:
  - Physical attributes (age, reach)
  - Career statistics
  - Recency-weighted performance
  - Bayesian-smoothed win rates

### 4. Ensemble Learning
- Logistic Regression → calibration
- XGBoost → non-linear interactions
- Combined for optimal performance

### 5. Evaluation
- Strict **time-based train/test split**
- Metrics:
  - Accuracy
  - Brier Score
  - Calibration (ECE)
