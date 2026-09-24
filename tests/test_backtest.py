import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from forecast_fm.backtest import backtest, qcol
from forecast_fm.config import ExperimentConfig
from forecast_fm.data import SERIES, TS, Y
from forecast_fm.metrics import score
from forecast_fm.models import REGISTRY
from forecast_fm.models.base import Forecaster
from forecast_fm.plans import HORIZON

from .conftest import make_exp, make_project, make_raw, panel_from


class Spy(Forecaster):
    """Records what the harness hands the model."""

    seen: list = []

    def fit(self, history, project, cutoff):
        Spy.seen.append(("fit", history[TS].max(), cutoff))

    def predict(self, history, future, project):
        Spy.seen.append(("predict", set(future.columns)))
        return np.ones(len(future)), None


@pytest.fixture
def spy(monkeypatch):
    Spy.seen = []
    monkeypatch.setitem(REGISTRY, "spy", Spy)
    return Spy


def test_model_sees_only_history_and_declared_known(spy):
    p = make_project()
    backtest(panel_from(make_raw(), p), p, make_exp("spy"))
    fits = [s for s in spy.seen if s[0] == "fit"]
    assert fits and all(hmax <= cutoff for _, hmax, cutoff in fits)
    for _, cols in (s for s in spy.seen if s[0] == "predict"):
        assert cols == {SERIES, TS, HORIZON, "price", "promo_flag", "promo_type"}


@pytest.mark.parametrize("model", ["naive", "seasonal_naive", "croston"])
def test_backtest_blind_to_future_targets_and_past_covariates(model):
    p = make_project()
    panel = panel_from(make_raw(), p)
    base, _ = backtest(panel, p, make_exp(model))
    tampered = panel.copy()
    late = tampered[TS] > base["cutoff"].min()
    tampered.loc[late, Y] = 9999.0
    tampered.loc[late, "sessions"] = 1e6
    again, _ = backtest(tampered, p, make_exp(model))
    first = base["cutoff"] == base["cutoff"].min()
    assert np.allclose(base.loc[first, "y_pred"], again.loc[first, "y_pred"])


def test_seasonal_naive_repeats_last_week():
    p = make_project(in_stock_col=None, past_covariate_cols=["sessions"])
    panel = panel_from(make_raw(n_series=1), p)
    preds, _ = backtest(panel, p, make_exp("seasonal_naive"))
    c = preds["cutoff"].iloc[0]
    y = panel.set_index(TS)[Y]
    for h in (1, 7, 8, 14):
        got = preds[(preds["cutoff"] == c) & (preds[HORIZON] == h)]["y_pred"].iloc[0]
        assert got == y[c - pd.Timedelta(days=6) + pd.Timedelta(days=(h - 1) % 7)]


def test_metrics_known_values():
    p = make_project(horizon=4, horizon_buckets=[2, 4])
    preds = pd.DataFrame({
        "fold": 0, SERIES: ["a"] * 4, HORIZON: [1, 2, 3, 4], "y_true": [10.0, 10, 10, 10],
        "y_pred": [12.0, 8, 10, 10], "stockout": False, "mase_scale": 2.0,
        qcol(0.5): [12.0, 8, 10, 10], qcol(0.9): [20.0, 20, 20, 20],
    })
    r = score(preds, p)
    assert r["overall"]["wape"] == pytest.approx(4 / 40)
    assert r["overall"]["bias"] == pytest.approx(0.0)
    assert r["overall"]["mase"] == pytest.approx(1.0 / 2.0)
    assert r["overall"]["cov0.9"] == 1.0
    b = r["tables"]["bucket"].set_index("bucket")
    assert b.loc["h01-2", "wape"] == pytest.approx(0.2) and b.loc["h03-4", "wape"] == 0


def test_stockout_days_not_scored():
    p = make_project(horizon=2, horizon_buckets=[2])
    preds = pd.DataFrame({"fold": 0, SERIES: "a", HORIZON: [1, 2], "y_true": [10.0, 0.0],
                          "y_pred": [10.0, 50.0], "stockout": [False, True], "mase_scale": 1.0})
    assert score(preds, p)["overall"]["wape"] == 0


def test_run_ledgers_and_verdict(tmp_path, monkeypatch):
    from forecast_fm.ledger import load_metrics, record, verdict

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "init"], check=True)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    p = make_project()
    panel = panel_from(make_raw(), p)
    for model, based_on in (("naive", None), ("seasonal_naive", "exp001")):
        exp = ExperimentConfig(name=model, hypothesis=f"{model} test", model=model, based_on=based_on)
        preds, _ = backtest(panel, p, exp)
        record(exp, p, score(preds, p), commit=True)
    m = load_metrics("exp002")
    assert m["based_on"] == "exp001" and m["verdict"] in {"improved", "regressed", "inconclusive"}
    assert "exp002" in (tmp_path / "experiments/LEDGER.md").read_text()
    log = subprocess.run(["git", "log", "--oneline"], capture_output=True, text=True).stdout
    assert "exp002 [" in log
    ref = {"overall": {"wape": 1.0}, "tables": {"fold": [{"fold": 0, "wape": 1.0}, {"fold": 1, "wape": 1.0}]}}
    new = {"overall": {"wape": 0.9}, "tables": {"fold": [{"fold": 0, "wape": 0.8}, {"fold": 1, "wape": 1.0}]}}
    assert verdict(new, ref, p)[0] == "inconclusive"  # one fold drives it
    new["tables"]["fold"][1]["wape"] = 0.95
    assert verdict(new, ref, p)[0] == "improved"
    json.dumps(m)
