"""Phase 3 acceptance: forecasters honor the interface, the metrics, the registry."""

import numpy as np
import pandas as pd
import pytest

from forecast_to_dispatch.config import load_config
from forecast_to_dispatch.data.ingest import load_sample
from forecast_to_dispatch.features.build import build_features
from forecast_to_dispatch.forecast.base import enforce_monotone, quantile_col
from forecast_to_dispatch.forecast.lstm import LSTMForecaster
from forecast_to_dispatch.forecast.metrics import (
    evaluate_quantile_forecast,
    interval_coverage,
    pinball_loss,
    scarcity_recall,
)
from forecast_to_dispatch.forecast.quantile_lgbm import QuantileLGBM
from forecast_to_dispatch.forecast.train import chronological_split, climatology_baseline

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


@pytest.fixture(scope="module")
def data():
    panel = load_sample(load_config())
    X, targets = build_features(panel)
    y = targets["y_price"]
    return chronological_split(X, y, test_fraction=0.3)


@pytest.fixture(scope="module")
def lgbm(data):
    X_tr, y_tr, _, _ = data
    # Sample-sized hyperparameters: the committed sample has ~16 training days,
    # so the default min_child_samples would leave the trees nearly unsplit.
    return QuantileLGBM(
        QUANTILES, seed=42, n_estimators=200, min_child_samples=10, num_leaves=31
    ).fit(X_tr, y_tr)


def test_split_is_chronological(data):
    X_tr, _, X_te, _ = data
    assert X_tr.index.max() < X_te.index.min(), "test window must be the future"


def test_lgbm_predictions_are_monotone_and_aligned(data, lgbm):
    _, _, X_te, _ = data
    preds = lgbm.predict_quantiles(X_te)
    assert list(preds.columns) == [quantile_col(q) for q in QUANTILES]
    assert preds.index.equals(X_te.index)
    diffs = preds.to_numpy()[:, 1:] - preds.to_numpy()[:, :-1]
    assert (diffs >= 0).all(), "quantile crossing must be repaired"


def test_lgbm_beats_climatology_on_pinball():
    """Model-quality gate: needs the full study dataset. On the one-month CI
    sample (~16 training days) hour-of-day climatology is statistically hard
    to beat and the comparison is meaningless, so the test skips there."""
    from forecast_to_dispatch.config import resolve_path

    processed = resolve_path(load_config(), "processed")
    feat_file = processed / "features.parquet"
    if not feat_file.exists():
        pytest.skip("full feature matrix not built")
    X = pd.read_parquet(feat_file)
    if len(X) < 2000:
        pytest.skip("sample-sized dataset — model-vs-climatology needs the full study window")
    y = pd.read_parquet(processed / "targets.parquet")["y_price"]
    X_tr, y_tr, X_te, y_te = chronological_split(X, y)
    preds = QuantileLGBM(QUANTILES, seed=42).fit(X_tr, y_tr).predict_quantiles(X_te)
    base = climatology_baseline(y_tr, X_te, QUANTILES)
    model_pb = np.mean([pinball_loss(y_te, preds[quantile_col(q)], q) for q in QUANTILES])
    base_pb = np.mean([pinball_loss(y_te, base[quantile_col(q)], q) for q in QUANTILES])
    assert model_pb < base_pb, "a model that can't beat climatology has learned nothing"


def test_conformal_wrapper_calibrates_and_roundtrips(tmp_path, data):
    from forecast_to_dispatch.forecast.conformal import ConformalizedForecaster

    X_tr, y_tr, X_te, _ = data
    model = ConformalizedForecaster(
        QuantileLGBM(QUANTILES, seed=42, n_estimators=60, min_child_samples=10)
    ).fit(X_tr, y_tr)
    assert set(model.offsets) == {"q05_q95", "q25_q75"}
    preds = model.predict_quantiles(X_te)
    raw = model.base.predict_quantiles(X_te)
    # Positive offsets must widen the interval; negative tighten. Either way
    # the adjusted band differs measurably from the raw one.
    assert not np.allclose(preds["q95"], raw["q95"])
    model.save(tmp_path / "cqr")
    reloaded = ConformalizedForecaster.load(tmp_path / "cqr")
    pd.testing.assert_frame_equal(preds, reloaded.predict_quantiles(X_te))


def test_lgbm_save_load_roundtrip(tmp_path, data, lgbm):
    _, _, X_te, _ = data
    lgbm.save(tmp_path / "m")
    reloaded = QuantileLGBM.load(tmp_path / "m")
    pd.testing.assert_frame_equal(lgbm.predict_quantiles(X_te), reloaded.predict_quantiles(X_te))


def test_lgbm_refuses_wrong_features(data, lgbm):
    _, _, X_te, _ = data
    with pytest.raises(ValueError, match="Feature columns differ"):
        lgbm.predict_quantiles(X_te.iloc[:, ::-1])


def test_lstm_interface_and_shapes(data):
    X_tr, y_tr, X_te, _ = data
    model = LSTMForecaster(QUANTILES, seed=42, max_epochs=3, hidden=16, layers=1)
    model.fit(X_tr, y_tr)
    # keep only complete 24h days in the test slice
    full_days = X_te.groupby(X_te.index.normalize()).size() == 24
    keep = X_te.index.normalize().isin(full_days[full_days].index)
    preds = model.predict_quantiles(X_te.loc[keep])
    assert len(preds) == int(keep.sum())
    diffs = preds.to_numpy()[:, 1:] - preds.to_numpy()[:, :-1]
    assert (diffs >= 0).all()


def test_lstm_save_load_roundtrip(tmp_path, data):
    X_tr, y_tr, X_te, _ = data
    model = LSTMForecaster(QUANTILES, seed=42, max_epochs=2, hidden=8, layers=1)
    model.fit(X_tr, y_tr)
    model.save(tmp_path / "lstm")
    reloaded = LSTMForecaster.load(tmp_path / "lstm")
    full_days = X_te.groupby(X_te.index.normalize()).size() == 24
    keep = X_te.index.normalize().isin(full_days[full_days].index)
    pd.testing.assert_frame_equal(
        model.predict_quantiles(X_te.loc[keep]), reloaded.predict_quantiles(X_te.loc[keep])
    )


def test_pinball_p50_is_half_mae():
    y = pd.Series([10.0, 20.0, 30.0])
    pred = pd.Series([12.0, 18.0, 33.0])
    assert np.isclose(pinball_loss(y, pred, 0.5), 0.5 * (y - pred).abs().mean())


def test_coverage_and_recall_edge_cases():
    y = pd.Series([1.0, 2.0, 100.0, 200.0])
    lo = pd.Series([0.0, 0.0, 0.0, 0.0])
    hi = pd.Series([5.0, 5.0, 150.0, 80.0])
    # y=1 in [0,5]; y=2 in [0,5]; y=100 in [0,150]; y=200 NOT in [0,80] -> 3/4
    assert interval_coverage(y, lo, hi) == 0.75
    # scarcity hours (>=90): y=100 (p95=150, seen) and y=200 (p95=80, missed)
    recall, n = scarcity_recall(y, p95=hi, threshold=90.0)
    assert n == 2 and recall == 0.5
    recall_nan, n0 = scarcity_recall(y, hi, threshold=1e9)
    assert n0 == 0 and np.isnan(recall_nan)


def test_enforce_monotone_repairs_crossing():
    df = pd.DataFrame({"q05": [5.0], "q50": [3.0], "q95": [4.0]})
    fixed = enforce_monotone(df)
    assert fixed.iloc[0].tolist() == [3.0, 4.0, 5.0]


def test_evaluate_produces_complete_card(data, lgbm):
    X_tr, y_tr, X_te, y_te = data
    preds = lgbm.predict_quantiles(X_te)
    card = evaluate_quantile_forecast(y_te, preds, QUANTILES, y_tr.quantile(0.99))
    for key in [
        "pinball_mean",
        "coverage_90",
        "coverage_50",
        "mae_p50",
        "scarcity_recall",
        "scarcity_hours",
        "n_hours",
    ]:
        assert key in card
