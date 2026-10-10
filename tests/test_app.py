from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


def test_app_runs_end_to_end_on_demo_data():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception
    at.radio[0].set_value("Demo data")
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    # Run the Monte Carlo simulation too
    run = next(b for b in at.button if b.label == "Run simulation")
    run.click()
    at.run()
    assert not at.exception, at.exception
