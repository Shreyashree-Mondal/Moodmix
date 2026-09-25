"""Unit tests for the data, scoring and recommendation logic in app.py."""
import pandas as pd
import pytest


# ---------- Data loading ----------
def test_songs_load_with_required_columns(app, songs):
    df, features = songs
    for col in [app.TRACK_COL, app.ARTIST_COL, app.GENRE_COL, app.POPULARITY_COL]:
        assert col in df.columns
    assert len(features) >= 5
    assert not df[app.POPULARITY_COL].isna().any()


def test_no_duplicate_songs(app, songs):
    df, _ = songs
    assert not df.duplicated(subset=[app.TRACK_COL, app.ARTIST_COL]).any()


def test_normalize_renames_and_drops_duplicates(app):
    raw = pd.DataFrame({
        "Unnamed: 0": [1, 2, 3],
        "title": ["A", "A", "B"], "artist": ["X", "X", "Y"],
        "top genre": ["pop", "pop", "edm"], "pop": [50, 50, 70], "bpm": [120, 120, 90],
    })
    out = app.normalize_feature_columns(raw)
    assert "popularity" in out.columns and "top_genre" in out.columns
    assert not any(c.startswith("unnamed") for c in out.columns)
    assert len(out) == 2


# ---------- Helpers ----------
@pytest.mark.parametrize("value,expected", [
    ("pop;edm", ["pop", "edm"]),
    (" pop ; ; edm ", ["pop", "edm"]),
    ("", []),
    (None, []),
])
def test_split_genres(app, value, expected):
    assert app.split_genres(value) == expected


@pytest.mark.parametrize("period", ["Morning", "Afternoon", "Evening", "Night"])
def test_time_score_is_between_0_and_1(app, songs, period):
    df, _ = songs
    scores = df.apply(lambda r: app.get_time_feature_score(r, period), axis=1)
    assert scores.between(0, 1).all()


def test_morning_prefers_calm_acoustic_songs(app):
    calm = {"acous": 90, "nrgy": 20, "val": 60, "dnce": 40, "bpm": 80}
    loud = {"acous": 0, "nrgy": 95, "val": 60, "dnce": 40, "bpm": 170}
    assert app.get_time_feature_score(calm, "Morning") > app.get_time_feature_score(loud, "Morning")
    assert app.get_time_feature_score(loud, "Night") > app.get_time_feature_score(calm, "Night")


# ---------- Models ----------
def test_models_trained(models):
    assert "best_reg_model" in models and "pref_model" in models
    assert set(models["classification_results"]) == {"Logistic Reg", "Decision Tree", "Random Forest", "SVM"}


def test_predictions_in_valid_ranges(app, songs, models):
    df, features = songs
    pop = models["best_reg_model"].predict(df[features])
    prob = models["pref_model"].predict_proba(df[features])[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()
    assert len(pop) == len(df)


# ---------- Users and ratings ----------
def test_create_then_login_same_user(app):
    first = app.create_or_login_user("Tester", "Tester@Example.com")
    again = app.create_or_login_user("Tester", "tester@example.com")
    assert first["success"] and again["success"]
    assert first["user_id"] == again["user_id"]
    assert again["message"] == "Welcome back!"


def test_login_requires_username_and_email(app):
    assert app.create_or_login_user("", "a@b.com")["success"] is False
    assert app.create_or_login_user("Name", "  ")["success"] is False


def test_rating_updates_instead_of_duplicating(app, songs):
    df, _ = songs
    uid = app.create_or_login_user("Rater", "rater@example.com")["user_id"]
    song = df.iloc[0]
    app.add_rating(uid, song, 2)
    app.add_rating(uid, song, 5)
    mine = app.load_ratings()
    mine = mine[(mine["user_id"].astype(int) == uid) & (mine["title"] == song[app.TRACK_COL])]
    assert len(mine) == 1
    assert int(mine.iloc[0]["rating"]) == 5 and int(mine.iloc[0]["liked"]) == 1


# ---------- Recommendations ----------
def test_recommendations_are_ranked_and_explained(app, songs, models):
    df, features = songs
    uid = app.create_or_login_user("Recs", "recs@example.com")["user_id"]
    genres = df[app.GENRE_COL].value_counts().index[:2].tolist()
    app.save_preferred_genres(uid, genres)

    recs, tp, _, pref = app.recommend_for_user(uid, df, features, models, top_n=10)
    assert len(recs) == 10
    assert pref == genres
    assert recs["final_score"].is_monotonic_decreasing
    assert recs["final_score"].between(0, 1).all()
    assert recs["recommendation_reason"].str.startswith("Recommended because").all()


def test_recommendations_skip_songs_already_rated(app, songs, models):
    df, features = songs
    uid = app.create_or_login_user("Skip", "skip@example.com")["user_id"]
    app.save_preferred_genres(uid, [df[app.GENRE_COL].iloc[0]])
    first, *_ = app.recommend_for_user(uid, df, features, models, top_n=5)
    top_song = first.iloc[0]
    app.add_rating(uid, top_song, 4)
    second, *_ = app.recommend_for_user(uid, df, features, models, top_n=5)
    pairs = set(zip(second[app.TRACK_COL], second[app.ARTIST_COL]))
    assert (top_song[app.TRACK_COL], top_song[app.ARTIST_COL]) not in pairs


def test_recommendations_are_logged(app, songs, models):
    df, features = songs
    uid = app.create_or_login_user("Logger", "logger@example.com")["user_id"]
    app.save_preferred_genres(uid, [df[app.GENRE_COL].iloc[0]])
    app.recommend_for_user(uid, df, features, models, top_n=3)
    logs = app.load_recommendation_logs()
    assert (logs["user_id"].astype(int) == uid).sum() == 3
