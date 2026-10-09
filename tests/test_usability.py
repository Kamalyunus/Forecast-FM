"""First-run traps: config mistakes and missing files must fail early, with
one readable line, never after a backtest."""

import subprocess

import pytest
import yaml

from forecast_fm.config import load_experiment, load_project, read_yaml
from forecast_fm.ledger import check_reference

from .conftest import make_project, make_raw


def test_duplicate_yaml_keys_are_an_error(tmp_path):
    (tmp_path / "p.yaml").write_text("horizon: 7\nbacktest:\n  n_folds: 2\nhorizon: 9\n")
    with pytest.raises(ValueError, match="duplicate key 'horizon' on lines 1 and 4"):
        load_project(tmp_path / "p.yaml")
    (tmp_path / "e.yaml").write_text("name: a\nname: b\nhypothesis: h\nmodel: naive\n")
    with pytest.raises(ValueError, match="duplicate key 'name'"):
        load_experiment(tmp_path / "e.yaml")
    (tmp_path / "empty.yaml").write_text("")
    assert read_yaml(tmp_path / "empty.yaml") == {}
    with pytest.raises(FileNotFoundError, match="config file"):
        load_project(tmp_path / "missing.yaml")
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_project(tmp_path / "list.yaml")


def test_repo_project_yaml_targets_the_mac_by_default():
    p = load_project("project.yaml")
    assert p.model_params("chronos2", {})["device"] == "auto"  # a pasted block once set cuda here
    assert p.model_params("chronos2", {})["dtype"] == "float32"
    assert load_project("project.yaml", profile="cuda").model_params("chronos2", {})["device"] == "cuda"
    raw = yaml.safe_load(open("project.yaml"))
    assert all(isinstance(v, dict) for v in raw["profiles"].values())  # no empty profile


def test_experiment_config_needs_name_hypothesis_model(tmp_path):
    (tmp_path / "e.yaml").write_text("name: x\n")
    with pytest.raises(ValueError, match=r"needs \['hypothesis', 'model'\]"):
        load_experiment(tmp_path / "e.yaml")


def test_missing_based_on_fails_before_a_ledgered_run_and_warns_in_debug(tmp_path, monkeypatch, capsys):
    from forecast_fm.config import ExperimentConfig

    monkeypatch.chdir(tmp_path)
    exp = ExperimentConfig(name="x", hypothesis="h", model="naive", based_on="exp001")
    with pytest.raises(ValueError, match="based_on 'exp001' is not in"):
        check_reference(exp, commit=True)
    check_reference(exp, commit=False)
    assert "no verdict" in capsys.readouterr().out
    check_reference(ExperimentConfig(name="x", hypothesis="h", model="naive"), commit=True)


def test_debug_run_with_missing_reference_completes(tmp_path, monkeypatch, capsys):
    from forecast_fm.cli import main

    monkeypatch.chdir(tmp_path)
    make_raw(n_series=2).to_parquet("sales.parquet")
    d = dict(vars(make_project()))
    d["data_path"] = "sales.parquet"
    (tmp_path / "proj.yaml").write_text(yaml.safe_dump(d))
    (tmp_path / "e.yaml").write_text(yaml.safe_dump({"name": "sn", "hypothesis": "h",
                                                     "model": "seasonal_naive", "based_on": "exp009"}))
    main(["-p", "proj.yaml", "run", "e.yaml", "--no-commit"])
    out = capsys.readouterr().out
    assert "verdict=reference" in out and "[run] metrics and slice tables ->" in out


def test_ledgered_run_with_missing_reference_refuses_before_compute(tmp_path, monkeypatch):
    from forecast_fm import runner
    from forecast_fm.cli import main

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    make_raw(n_series=2).to_parquet("sales.parquet")
    d = dict(vars(make_project()))
    d["data_path"] = "sales.parquet"
    (tmp_path / "proj.yaml").write_text(yaml.safe_dump(d))
    (tmp_path / "e.yaml").write_text(yaml.safe_dump({"name": "sn", "hypothesis": "h",
                                                     "model": "seasonal_naive", "based_on": "exp009"}))
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
    called = []
    monkeypatch.setattr(runner, "backtest", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="based_on 'exp009'"):
        main(["-p", "proj.yaml", "run", "e.yaml"])
    assert not called


def test_entry_turns_user_errors_into_one_message(tmp_path, monkeypatch, capsys):
    from forecast_fm.cli import entry

    monkeypatch.chdir(tmp_path)
    d = dict(vars(make_project()))
    d["data_path"] = "nope.parquet"
    (tmp_path / "proj.yaml").write_text(yaml.safe_dump(d))
    assert entry(["-p", "proj.yaml", "audit"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "nope.parquet" in err
    assert "proj.yaml" in err and "DATA_REQUIREMENTS" in err and "FORECAST_FM_DEBUG" in err
    monkeypatch.setenv("FORECAST_FM_DEBUG", "1")
    with pytest.raises(FileNotFoundError):  # the traceback when asked for
        entry(["-p", "proj.yaml", "audit"])
    monkeypatch.delenv("FORECAST_FM_DEBUG")
    (tmp_path / "e.yaml").write_text(yaml.safe_dump({"name": "c", "hypothesis": "h", "model": "chronos2",
                                                     "model_params": {"device": "cpu"}}))
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "torch" else object())
    assert entry(["-p", "proj.yaml", "run", "e.yaml", "--no-commit"]) == 1
    assert 'pip install -e ".[foundation,dev]"' in capsys.readouterr().err  # refused before loading data
    monkeypatch.setenv("FORECAST_FM_PROJECT", "proj.yaml")
    assert entry(["audit"]) == 1  # -p defaulted from the environment
    assert "nope.parquet" in capsys.readouterr().err
    assert entry(["models", "croston"]) == 0
    assert "Syntetos-Boylan" in capsys.readouterr().out
    assert entry(["models", "nope"]) == 1
