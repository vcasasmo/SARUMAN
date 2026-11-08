"""Configuration loading and validation for SARUMAN.

This module does not modify the scientific pipeline. Existing modules still need
explicit wiring to consume the returned settings.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import argparse
import os

try:
    import yaml
except ImportError as exc:
    raise ImportError("SARUMAN configuration requires PyYAML (pip install pyyaml).") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("default_config.yaml")
INTERNAL_PATH_KEYS = {
    ("paths", "data_dir"),
    ("paths", "detector_coordinates_file"),
    ("detector_surrogate", "nn", "model_path"),
    ("detector_surrogate", "nn", "input_scaler"),
    ("detector_surrogate", "nn", "output_scaler"),
    ("ct2lac", "rf_model_path"),
    ("ct2lac", "density_calibration_path"),
    ("ct2lac", "nist_database_path"),
}

class ConfigurationError(ValueError):
    """Invalid configuration or conflicting settings."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration must contain a YAML mapping: {path}")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    for key, value in override.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ConfigurationError(f"Unknown configuration parameter: {name}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigurationError(f"Expected a mapping at {name}")
            _merge(base[key], value, name)
        else:
            base[key] = value
    return base


def _nested(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    val: Any = config
    for key in keys:
        val = val[key]
    return val


def _set_nested(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    node = config
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _validate(config: dict[str, Any]) -> None:
    ct = config["ct_processing"]
    if ct["downsampling_method"] not in {"baseline", "bspline", "oversample_block_average"}:
        raise ConfigurationError("ct_processing.downsampling_method is invalid")
    if not isinstance(ct["oversample_factor"], int) or ct["oversample_factor"] < 1:
        raise ConfigurationError("ct_processing.oversample_factor must be a positive integer")
    for key in ("grid_size", "grid_spacing_mm", "shift_mm"):
        values = ct[key]
        if not isinstance(values, list) or len(values) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
            raise ConfigurationError(f"ct_processing.{key} must have three numeric values")
    if any(int(x) != x or x <= 0 for x in ct["grid_size"]):
        raise ConfigurationError("ct_processing.grid_size must be positive integers")
    if any(x <= 0 for x in ct["grid_spacing_mm"]):
        raise ConfigurationError("ct_processing.grid_spacing_mm must be positive")
    if config["path_calculation"]["start_method"] not in {"spawn", "fork", "forkserver"}:
        raise ConfigurationError("path_calculation.start_method must be spawn, fork, or forkserver")
    if config["path_calculation"]["max_workers"] < 1 or config["path_calculation"]["chunk_size"] < 1:
        raise ConfigurationError("Path calculation workers and chunk size must be positive")
    if config["solver"]["iterations"] < 1 or config["solver"]["posterior_samples"] < 1:
        raise ConfigurationError("Solver iterations and posterior_samples must be positive")
    prior = config["solver"]["prior"]
    if prior["magnitude_exponent_lower"] >= prior["magnitude_exponent_upper"]:
        raise ConfigurationError("Magnitude prior lower bound must be below upper bound")
    for key in ("beta_alpha", "beta_beta", "noise_gamma_shape", "noise_gamma_rate"):
        if prior[key] <= 0:
            raise ConfigurationError(f"solver.prior.{key} must be positive")


def load_config(config_path: str | os.PathLike[str] | None = None, *, cwd: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load bundled defaults, then optionally override from a user YAML.

    Search order: explicit --config; ./config.yaml; bundled config.yaml.
    Only resource paths inside the installation are resolved against PROJECT_ROOT.
    Input/output paths will be CLI arguments and are not handled here.
    """
    if not DEFAULT_CONFIG.is_file():
        raise ConfigurationError(f"Bundled defaults missing: {DEFAULT_CONFIG}")
    config = deepcopy(_read_yaml(DEFAULT_CONFIG))
    wd = Path(cwd).resolve() if cwd is not None else Path.cwd()
    if config_path is not None:
        selected = Path(config_path).expanduser()
        if not selected.is_absolute():
            selected = wd / selected
        selected = selected.resolve()
    elif (wd / "config.yaml").resolve() != DEFAULT_CONFIG.resolve() and (wd / "config.yaml").is_file():
        selected = (wd / "config.yaml").resolve()
    else:
        selected = DEFAULT_CONFIG
    if selected != DEFAULT_CONFIG.resolve():
        _merge(config, _read_yaml(selected))
    _validate(config)
    for keys in INTERNAL_PATH_KEYS:
        raw = _nested(config, keys)
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigurationError(f"Invalid internal path: {'.'.join(keys)}")
        path = Path(raw).expanduser()
        _set_nested(config, keys, (path if path.is_absolute() else PROJECT_ROOT / path).resolve())
    return config


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(val) for val in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and inspect SARUMAN configuration")
    parser.add_argument("--config", help="Override YAML configuration path")
    parser.add_argument("--show", action="store_true", help="Print resolved configuration")
    args = parser.parse_args()
    config = load_config(args.config)
    print('SARUMAN: "The configuration is sound. The seeing stone is ready."')
    if args.show:
        print(yaml.safe_dump(_stringify_paths(config), sort_keys=False, allow_unicode=True))

if __name__ == "__main__":
    main()
