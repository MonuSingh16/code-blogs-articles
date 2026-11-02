import numpy as np
import pandas as pd
import pytest

from housing.monitoring import drift_report, performance_report, psi


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_psi_identical_is_near_zero(rng):
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) == pytest.approx(0, abs=1e-6)


def test_psi_same_distribution_is_small(rng):
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.1


def test_psi_shifted_distribution_is_large(rng):
    a, b = rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)
    assert psi(a, b) > 0.25


def test_psi_handles_values_outside_reference_range(rng):
    a = rng.normal(0, 1, 5000)
    b = np.concatenate([a, [1e6, -1e6]])
    assert np.isfinite(psi(a, b))


def test_drift_report_flags_only_shifted_feature(rng):
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(2, 1, 5000)})
    report = drift_report(ref, cur).set_index("feature")
    assert report.loc["a", "status"] == "ok"
    assert report.loc["b", "status"] == "alert"
    assert report.index[0] == "b"  # sorted by PSI, worst first


def test_performance_ok_when_matching_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = y + np.array([0.1, -0.1, 0.1, -0.1])
    r = performance_report(y, pred, pred - 0.5, pred + 0.5, baseline_rmse=0.1)
    assert r["rmse"] == pytest.approx(0.1)
    assert not r["rmse_alert"]
    assert r["coverage"] == 1.0
    assert not r["coverage_alert"]


def test_performance_alerts_on_degradation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = y - 1.0  # model under-predicts everything
    r = performance_report(y, pred, pred - 0.1, pred + 0.1, baseline_rmse=0.1)
    assert r["bias"] == pytest.approx(1.0)
    assert r["rmse_alert"]
    assert r["coverage"] == 0.0
    assert r["coverage_alert"]
