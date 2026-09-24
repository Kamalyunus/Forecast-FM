"""`finetune`: train once on history <= as_of, save a checkpoint with a
manifest, forecast from it; never backtest it on data it trained on."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from forecast_fm.backtest import backtest, forecast_at
from forecast_fm.config import ExperimentConfig, load_experiment
from forecast_fm.data import SERIES, TS
from forecast_fm.finetune import finetune
from forecast_fm.models import create_model
from forecast_fm.models.chronos2 import MANIFEST, TrainedOnFutureError

from .conftest import make_exp, make_project, make_raw, panel_from

FT = {"num_steps": 2, "batch_size": 4, "learning_rate": 1e-4}
# no plan file in these tests: hold the known covariates flat in production
CF = {"price": "carry_forward", "promo_flag": "carry_forward", "promo_type": "carry_forward"}


def _recipe(model_id):
    return ExperimentConfig(name="LoRA recipe", hypothesis="h", model="chronos2",
                            model_params={"model_id": model_id, "device": "cpu", "fine_tune": FT})


def test_saved_checkpoint_refuses_earlier_cutoffs_before_loading(tmp_path):
    """Runs without torch: the guard fires before any weights load."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / MANIFEST).write_text(json.dumps({"train_end": "2024-06-01"}))
    p = make_project()
    panel = panel_from(make_raw(), p)
    m = create_model("chronos2", {"model_id": str(ckpt)})
    with pytest.raises(TrainedOnFutureError, match="2024-06-01"):
        m.fit(panel, p, pd.Timestamp("2024-05-31"))


def test_finetune_trains_only_on_history_up_to_as_of(tiny, tmp_path, monkeypatch):
    from chronos import Chronos2Pipeline

    seen = {}
    real_fit = Chronos2Pipeline.fit

    def spy(self, inputs, **kw):
        seen["lengths"] = [len(d["target"]) for d in inputs]
        seen["cov_lengths"] = {len(v) for d in inputs for v in d["past_covariates"].values()}
        return real_fit(self, inputs, **kw)

    monkeypatch.setattr(Chronos2Pipeline, "fit", spy)
    p = make_project()
    panel = panel_from(make_raw(n_series=3), p)
    as_of = panel[TS].max() - pd.Timedelta(days=30)
    out = finetune(p, _recipe(tiny), panel, as_of=as_of, out=tmp_path / "m")

    expect = panel[panel[TS] <= as_of].groupby(SERIES, observed=True).size().tolist()
    assert sorted(seen["lengths"]) == sorted(expect)
    assert seen["cov_lengths"] <= set(expect)
    man = json.loads((out / MANIFEST).read_text())
    assert man["train_end"] == str(as_of.date())
    assert man["base_model_id"] == tiny and man["recipe"]["model_params"]["fine_tune"] == FT
    assert man["training"]["n_series"] == 3
    assert (out / "finetuned-ckpt").is_dir() and not (tmp_path / "m.partial").exists()


def test_forecast_from_checkpoint_and_guard(tiny, tmp_path):
    p = make_project(covariate_eval_policy=CF)
    panel = panel_from(make_raw(n_series=3), p)
    as_of = panel[TS].max() - pd.Timedelta(days=10)
    out = finetune(p, _recipe(tiny), panel, as_of=as_of, out=tmp_path / "m")
    serve = load_experiment(out / "forecast_config.yaml")
    assert serve.model_params["model_id"] == str(out) and "fine_tune" not in serve.model_params

    pred, stats = forecast_at(panel, panel[TS].max(), p, serve, None, production=True)
    assert np.isfinite(pred["y_pred"]).all() and len(pred) == 3 * p.horizon
    assert stats["checkpoint_train_end"] == str(as_of.date()) and stats["checkpoint_age_days"] == 10
    pred2, _ = forecast_at(panel, as_of, p, serve, None)  # the train_end itself is fine
    assert len(pred2) == 3 * p.horizon

    with pytest.raises(TrainedOnFutureError):  # backtest folds sit before train_end
        backtest(panel, p, serve)


def test_output_is_never_silently_overwritten(tiny, tmp_path):
    p = make_project()
    panel = panel_from(make_raw(n_series=2), p)
    out = finetune(p, _recipe(tiny), panel, out=tmp_path / "m")
    with pytest.raises(FileExistsError):
        finetune(p, _recipe(tiny), panel, out=out)
    finetune(p, _recipe(tiny), panel, out=out, force=True)
    assert (out / MANIFEST).exists()


def test_finetune_rejects_non_finetune_configs(tmp_path):
    p = make_project()
    panel = panel_from(make_raw(n_series=1), p)
    with pytest.raises(ValueError, match="fine_tune"):
        finetune(p, make_exp("chronos2"), panel, out=tmp_path / "m")
    with pytest.raises(ValueError, match="after the last date"):
        finetune(p, _recipe("x"), panel, as_of="2099-01-01", out=tmp_path / "m")


def test_cli_finetune_then_forecast(tiny, tmp_path, monkeypatch):
    from forecast_fm.cli import main

    monkeypatch.chdir(tmp_path)
    make_raw(n_series=3).to_parquet("sales.parquet")
    proj = dict(vars(make_project(covariate_eval_policy=CF)))
    proj["data_path"] = "sales.parquet"
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(proj))
    cfg = {"name": "lora", "hypothesis": "h", "model": "chronos2",
           "model_params": {"model_id": tiny, "device": "cpu", "fine_tune": FT}}
    (tmp_path / "ft.yaml").write_text(yaml.safe_dump(cfg))

    main(["finetune", "ft.yaml", "--out", "models/lora"])
    main(["forecast", "models/lora/forecast_config.yaml", "--out", "fc.parquet"])
    fc = pd.read_parquet("fc.parquet")
    assert fc["series_id"].nunique() == 3 and fc["y_pred"].notna().all()
