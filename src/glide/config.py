# Functions for handling the configuration file

import copy
import logging
from datetime import datetime, timezone

from yaml import safe_load_all

_log = logging.getLogger(__name__)

# Helper functions


def _ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_core() -> tuple[dict, dict, dict]:
    """Load core variable definitions from bundled core.yml.

    Returns
    -------
    core_variables : dict
        Core variables that are always included.
    flight_attitude : dict
        Optional flight attitude variables (heading, pitch, roll).
    derived_thermo : dict
        Optional derived thermodynamic variables.
    """
    from importlib import resources

    core_file = str(resources.files("glide").joinpath("assets/core.yml"))

    with open(core_file) as f:
        docs = [doc for doc in safe_load_all(f)]

    core = docs[0] if docs else {}
    flight = docs[1] if len(docs) > 1 else {}
    thermo = docs[2] if len(docs) > 2 else {}

    return core, flight, thermo


def _apply_qc_overrides(variables: dict, qc_overrides: dict) -> dict:
    """Apply QC parameter overrides to variable definitions.

    Only allows overriding: valid_min, valid_max, max_gap, interpolate_missing
    """
    allowed_keys = {"valid_min", "valid_max", "max_gap", "interpolate_missing"}

    for var_name, overrides in qc_overrides.items():
        if var_name not in variables:
            _log.warning("QC override for unknown variable: %s", var_name)
            continue

        for key, value in overrides.items():
            if key not in allowed_keys:
                _log.warning(
                    "Ignoring invalid QC override key '%s' for %s", key, var_name
                )
                continue

            # valid_min/max go in CF attributes
            if key in ("valid_min", "valid_max"):
                if "CF" not in variables[var_name]:
                    variables[var_name]["CF"] = {}
                variables[var_name]["CF"][key] = value
                _log.debug("Override %s.CF.%s = %s", var_name, key, value)
            else:
                variables[var_name][key] = value
                _log.debug("Override %s.%s = %s", var_name, key, value)

    return variables


def _build_slocum_name_map(variables: dict) -> dict:
    """Build mapping from Slocum variable names to output variable names."""
    slocum_name_map = {}
    for variable_name, specs in variables.items():
        if "source" not in specs:
            continue
        sources = specs["source"]
        if not isinstance(sources, list):
            sources = [sources]
        for source in sources:
            slocum_name_map[source] = variable_name

    _log.debug("Slocum name mapping dict %s", slocum_name_map)
    return slocum_name_map


# Public functions


def load_config(file: str | None = None) -> dict:
    """Load and merge configuration from core and user files.

    Parameters
    ----------
    file : str, optional
        Path to user configuration file. If None, uses bundled default.

    Returns
    -------
    dict
        Merged configuration with keys:
        - globals: trajectory and netcdf_attributes
        - variables: all variable definitions (core + optional + user)
        - slocum: mapping from Slocum names to output names
        - merged_variables: variables for higher-level processing
    """
    # Load core definitions
    core, flight, thermo = _load_core()

    # Load user config
    if file is None:
        from importlib import resources

        file = str(resources.files("glide").joinpath("assets/config.yml"))

    with open(file) as f:
        docs = [doc for doc in safe_load_all(f)]

    # Parse user config documents
    global_config = docs[0] if docs else {}
    include_config = docs[1] if len(docs) > 1 else {}
    qc_config = docs[2] if len(docs) > 2 else {}
    l1_variables = docs[3] if len(docs) > 3 else {}
    merged_variables = docs[4] if len(docs) > 4 else {}

    # Extract include toggles (default to True for backward compatibility)
    include = include_config.get("include", {})
    include_flight = include.get("flight", True)
    include_thermo = include.get("thermo", True)

    # Build variable set: start with core
    variables = copy.deepcopy(core)

    # Add optional suites if enabled
    if include_flight:
        variables.update(copy.deepcopy(flight))
        _log.debug("Including flight_attitude suite")
    if include_thermo:
        variables.update(copy.deepcopy(thermo))
        _log.debug("Including derived_thermo suite")

    # Apply QC overrides
    qc_overrides = qc_config.get("qc", {}) if isinstance(qc_config, dict) else {}
    variables = _apply_qc_overrides(variables, qc_overrides)

    # Add user L1 variables
    l1_vars = (
        l1_variables.get("l1_variables", {}) if isinstance(l1_variables, dict) else {}
    )
    for var_name, var_spec in l1_vars.items():
        if var_name in variables:
            _log.warning("L1 variable '%s' conflicts with core variable", var_name)
        else:
            variables[var_name] = var_spec
            _log.debug("Added L1 variable: %s", var_name)

    # Build Slocum name mapping
    slocum_name_map = _build_slocum_name_map(variables)

    # Handle time valid_min/max UTC conversion
    if "time" in variables and "CF" in variables["time"]:
        for attr in ["valid_min", "valid_max"]:
            if attr in variables["time"]["CF"]:
                val = variables["time"]["CF"][attr]
                if isinstance(val, datetime):
                    variables["time"]["CF"][attr] = _ensure_utc(val).timestamp()

    # Extract merged variables
    merged_vars = (
        merged_variables.get("merged_variables", {})
        if isinstance(merged_variables, dict)
        else {}
    )

    config = dict(
        globals=global_config,
        variables=variables,
        slocum=slocum_name_map,
        merged_variables=merged_vars,
        include=dict(
            flight=include_flight,
            thermo=include_thermo,
        ),
    )

    return config
