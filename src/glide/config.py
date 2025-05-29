# Functions for handling the configuration file

import logging
from datetime import datetime, timezone

from yaml import safe_load_all

_log = logging.getLogger(__name__)

# Helper functions


def _ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc)


# Public functions


def load_config(file: str | None = None) -> dict:
    """Extract variable specifications from a yaml file."""
    if file is None:
        from importlib import resources

        file = str(resources.files("glide").joinpath("assets/config.yml"))

    with open(file) as f:
        # Documents in yaml are defined by: ---
        docs = [doc for doc in safe_load_all(f)]

    global_config = docs[0]
    variable_specs = docs[1]

    # Generate mapping from Slocum source name to dataset name
    slocum_name_map = {
        source: variable_name
        for variable_name, specs in variable_specs.items()
        if "source" in specs
        for source in (
            specs["source"] if isinstance(specs["source"], list) else [specs["source"]]
        )
    }

    _log.debug("Slocum name mapping dict %s", slocum_name_map)

    # parse datetime in time valid_min and valid_max
    for v in ["valid_min", "valid_max"]:
        if v in variable_specs["time"]["CF"]:
            variable_specs["time"]["CF"][v] = _ensure_utc(
                variable_specs["time"]["CF"][v]
            ).timestamp()

    config = dict(
        globals=global_config, variables=variable_specs, slocum=slocum_name_map
    )

    return config
