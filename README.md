# MoodMix: ML-Powered Music Recommender

A Streamlit app that recommends songs based on a listener's preferred genres, listening history, ratings, and time of day. It combines supervised ML models, content-based similarity, and collaborative filtering into one explainable recommendation score.

- **Live demo:** link coming soon
- **Companion project (SQL version):** [music-recommendation-app](https://github.com/Shreyashree-Mondal/music-recommendation-app), a 3NF MySQL design of the same domain

Built as a team project (Group 14) for a graduate analytics course at the University of Houston.

## What the app does

- **Sign-in and onboarding:** users pick their favourite genres.
- **Explore and rate songs:** listening history and 1–5 star ratings are recorded.
- **Personalised recommendations:** each song gets a score, with a plain-language reason ("users with similar ratings also liked this song", "fits your evening listening", and so on).
- **Analytics dashboard:** interactive Plotly charts for EDA, regression, classification, feature importance, and user behaviour.

## How recommendations are scored

```mermaid
flowchart LR
    A[Song audio features] --> B[Popularity regressor]
    A --> C[Like/dislike classifier]
    A --> D[Similarity to user's liked songs]
    E[Ratings matrix] --> F[User-user CF]
    E --> G[Item-item CF]
    H[Time of day + genre prefs] --> I[Context score]
    B & C & D & F & G & I --> J[Weighted final score + reason]
```

| Signal | Weight |
|---|---|
| Audio similarity to the user's liked/listened songs (cosine, scaled features) | 0.20 |
| Predicted probability the user likes the song (classifier) | 0.18 |
| Predicted popularity (regressor) | 0.16 |
| Time-of-day fit (energy, acousticness, tempo, valence) | 0.14 |
| User-user collaborative filtering match | 0.12 |
| Item-item collaborative filtering match | 0.10 |
| Genre the user plays at this time of day | 0.06 |
| Selected preferred genre | 0.04 |

Songs the user has already heard or rated are excluded.

## Data

- **Songs:** 587 Spotify top-chart songs from 2010–2019 (`top10s.csv`) with audio features: tempo, energy, danceability, loudness, liveness, valence, duration, acousticness, and speechiness. 16 duplicates removed using regex-normalised title/artist keys.
- **Users:** 51 users, 1,253 listening records, and 819 ratings. Most of this interaction data was **simulated** to bootstrap the recommender: each synthetic user has 2–4 preferred genres, listens mostly within them, and rates songs using genre match, popularity, and random noise.

## Models and results

All models use scikit-learn pipelines (median imputation, scaling where needed) with a fixed seed of 42 and a 70/30 split. Hyperparameters were tuned with GridSearchCV / RandomizedSearchCV and 5-fold cross-validation.

**1. Popularity regression** (Linear, Polynomial, Decision Tree, Random Forest, log-target Random Forest)

| Best model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest Regressor | 10.10 | 13.42 | 0.07 |

Audio features explain very little of a song's popularity (OLS R² = 0.12). Only release year and loudness (positive) and energy (negative) were statistically significant. Polynomial regression overfit (R² = −0.13).

**2. Like/dislike prediction** (Logistic Regression, Decision Tree, Random Forest, SVM; balanced class weights)

| Best model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 57.7% | 57.3% | 70.9% | 0.634 |

The model was selected on recall, to avoid missing songs a user would like. Because the ratings are simulated from genre and popularity, these scores measure how well audio features recover that simulated preference signal rather than real listener taste.

**3. Genre classification** (Decision Tree, Random Forest, SVM)

| Best model | Accuracy | Weighted F1 |
|---|---|---|
| Random Forest | 60.4% | 0.48 |

This is only slightly above the majority-class baseline: "dance pop" alone is about 60% of the test set, and the models rarely predict the smaller genres correctly. With dozens of fine-grained genres and few examples of each, audio features alone can't separate them.

## Limitations and next steps

- User interactions are mostly simulated, so model scores don't reflect real listener behaviour. Collecting real ratings is the most valuable next step.
- Genre labels are very fine-grained; grouping them into broader families (pop, hip hop, electronic, and so on) would make genre prediction meaningful.
- On Streamlit Community Cloud, new users and ratings are stored in CSV files that reset when the app restarts. A hosted database (as in the SQL version of this project) would make them persistent.
- Recommendation weights were set by hand; they could be tuned against held-out ratings.

## Project structure

```
app.py                                   Streamlit app (UI, models, recommender, analytics)
top10s.csv                               Song dataset
users.csv, user_preferences.csv,         User interaction data (mostly simulated)
listening_history.csv, ratings.csv,
recommendation_logs.csv
notebooks/music_predictive_analytics.ipynb   EDA, model comparison, and recommender development
requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, NumPy, scikit-learn (Linear/Logistic Regression, Decision Tree, Random Forest, SVM, GridSearchCV), statsmodels (notebook), Plotly, Streamlit, Streamlit Community Cloud.
