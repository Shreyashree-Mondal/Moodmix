"""Shared fixtures.

Every test runs against a COPY of app.py and top10s.csv in a temporary folder,
because the app writes its user data (users.csv, ratings.csv, ...) next to
app.py. This keeps the real data files in the repo untouched.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("moodmix")
    shutil.copy(ROOT / "app.py", d / "app.py")
    shutil.copy(ROOT / "top10s.csv", d / "top10s.csv")
    return d


@pytest.fixture(scope="module")
def app(app_dir):
    """Import app.py as a module (Streamlit runs in 'bare' mode, no UI)."""
    spec = importlib.util.spec_from_file_location("moodmix_app", app_dir / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["moodmix_app"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def songs(app):
    df, features = app.load_song_data()
    return df, features


@pytest.fixture(scope="module")
def models(app, songs):
    df, features = songs
    return app.train_models_cached(df, tuple(features))
