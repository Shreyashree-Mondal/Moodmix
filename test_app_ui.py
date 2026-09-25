"""End-to-end UI tests: run the real Streamlit app headlessly with AppTest."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["Dashboard", "Explore Songs", "My History", "Recommendations",
         "Analytics Dashboard", "About Project"]


def _logged_in(app_dir):
    at = AppTest.from_file(str(app_dir / "app.py"), default_timeout=300)
    at.run()
    at.text_input[0].input("UITester")
    at.text_input[1].input("uitester@example.com")
    at.button[0].click().run()
    if at.multiselect:  # first login: choose genres
        ms = at.multiselect[0]
        ms.select(ms.options[0]).select(ms.options[1])
        next(b for b in at.button if "Save Preferences" in b.label).click().run()
    return at


def test_welcome_page_loads(app_dir):
    at = AppTest.from_file(str(app_dir / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Username", "Email"]


@pytest.fixture(scope="module")
def session(app_dir):
    at = _logged_in(app_dir)
    assert not at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(session, page):
    session.sidebar.radio[0].set_value(page).run()
    assert not session.exception, [e.value for e in session.exception]


def test_generate_recommendations_button(session):
    session.sidebar.radio[0].set_value("Recommendations").run()
    next(b for b in session.button if "Generate Recommendations" in b.label).click().run()
    assert not session.exception, [e.value for e in session.exception]
