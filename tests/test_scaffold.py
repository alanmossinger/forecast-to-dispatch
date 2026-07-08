"""Phase 0 acceptance: the package installs, the config loads, the layout exists."""

from pathlib import Path

import pytest

from forecast_to_dispatch.config import REPO_ROOT, load_config, resolve_path


def test_package_imports():
    import forecast_to_dispatch

    assert forecast_to_dispatch.__version__


def test_config_loads_and_is_complete():
    config = load_config()
    assert config["market"]["hub"] == "HB_HOUSTON"
    assert config["battery"]["power_mw"] > 0
    assert config["battery"]["energy_mwh"] > 0
    assert 0.05 in config["forecast"]["quantiles"]
    assert 0.95 in config["forecast"]["quantiles"]
    assert len(config["ancillary_services"]) == 5


def test_battery_round_trip_efficiency_is_physical():
    """Round-trip efficiency must be < 1 — a battery that creates energy would
    make every downstream revenue number fiction."""
    config = load_config()
    rte = config["battery"]["eta_charge"] * config["battery"]["eta_discharge"]
    assert 0.5 < rte < 1.0


def test_repo_layout_exists():
    assert (REPO_ROOT / "reports" / "figures").is_dir()
    assert (REPO_ROOT / "config" / "config.yaml").is_file()
    assert resolve_path(load_config(), "figures") == REPO_ROOT / "reports" / "figures"


def test_missing_config_fails_loudly():
    with pytest.raises(FileNotFoundError):
        load_config(Path("does/not/exist.yaml"))
