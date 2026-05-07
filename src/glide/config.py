# Functions for handling the configuration file

import copy
import logging
from datetime import datetime, timezone

from yaml import safe_load_all

_log = logging.getLogger(__name__)

# config.yml must contain exactly these top-level sections. Empty dicts are
# allowed for sections you don't use (e.g. `qc: {}`), but missing sections are
# an error so that typos and stale files fail loudly.
_REQUIRED_CONFIG_SECTIONS = (
    "trajectory",
    "netcdf_attributes",
    "include",
    "instruments",
    "qc",
    "l1_variables",
    "merged_variables",
)

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


def _load_core() -> tuple[dict, dict, dict, dict]:
    """Load core variable definitions from bundled core.yml.

    Returns
    -------
    core_variables : dict
        Core variables that are always included.
    flight_attitude : dict
        Optional flight attitude variables (heading, pitch, roll).
    derived_thermo : dict
        Optional derived thermodynamic variables.
    ngdac : dict
        IOOS NGDAC structural configuration.
    """
    from importlib import resources

    core_file = str(resources.files("glide").joinpath("assets/core.yml"))

    with open(core_file) as f:
        docs = [doc for doc in safe_load_all(f)]

    if len(docs) != 4:
        raise ValueError(
            f"Expected core.yml to contain exactly 4 YAML documents (core, "
            f"flight_attitude, derived_thermo, ngdac), but found {len(docs)}."
        )

    core = docs[0]
    flight = docs[1]
    thermo = docs[2]
    ngdac = docs[3]

    if not isinstance(ngdac, dict):
        raise ValueError(
            f"Expected NGDAC document in core.yml to be a mapping, got {type(ngdac).__name__}."
        )

    if "ngdac" in ngdac and isinstance(ngdac["ngdac"], dict):
        ngdac = ngdac["ngdac"]

    return core, flight, thermo, ngdac


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


def _parse_user_config(file: str) -> dict:
    """Read the user config file and validate it has the required sections.

    The file must be a single YAML document. Multi-document files (the
    previous glide layout) are detected and rejected with a migration hint.
    """
    with open(file) as f:
        all_docs = [d for d in safe_load_all(f)]

    if len(all_docs) > 1:
        raise ValueError(
            f"{file} is a multi-document YAML file, but glide now expects a "
            "single document with sections as top-level keys (trajectory, "
            "netcdf_attributes, include, instruments, qc, l1_variables, "
            "merged_variables). See assets/config.yml for the new layout."
        )

    parsed = all_docs[0] if all_docs else {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{file} must contain a YAML mapping at the top level, "
            f"got {type(parsed).__name__}."
        )

    missing = [s for s in _REQUIRED_CONFIG_SECTIONS if s not in parsed]
    if missing:
        raise ValueError(
            f"{file} is missing required section(s): {missing}. "
            f"Required sections are: {list(_REQUIRED_CONFIG_SECTIONS)}. "
            "Empty sections are allowed (e.g. `qc: {}`); just include the key."
        )

    return parsed


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
        - instruments: per-deployment instrument metadata
        - include: which optional suites are enabled
        - ngdac: IOOS NGDAC structural metadata
    """
    # Load core definitions
    core, flight, thermo, ngdac = _load_core()

    # Load user config
    if file is None:
        from importlib import resources

        _log.debug("No config file provided, using bundled default")
        file = str(resources.files("glide").joinpath("assets/config.yml"))

    parsed = _parse_user_config(file)

    def _section(name: str) -> dict:
        v = parsed.get(name)
        return v if isinstance(v, dict) else {}

    trajectory = _section("trajectory")
    netcdf_attributes = _section("netcdf_attributes")
    include = _section("include")
    instruments = _section("instruments")
    qc_overrides = _section("qc")
    l1_vars = _section("l1_variables")
    merged_vars = _section("merged_variables")

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
    variables = _apply_qc_overrides(variables, qc_overrides)

    # Add user L1 variables
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

    # The legacy `globals` key bundles trajectory + netcdf_attributes for
    # downstream code that still expects them under one umbrella.
    globals_block = {
        "trajectory": trajectory,
        "netcdf_attributes": netcdf_attributes,
    }

    config = dict(
        globals=globals_block,
        variables=variables,
        slocum=slocum_name_map,
        merged_variables=merged_vars,
        instruments=instruments,
        ngdac=ngdac,
        include=dict(
            flight=include_flight,
            thermo=include_thermo,
        ),
    )

    return config
