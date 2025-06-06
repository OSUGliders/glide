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
        docs = [doc for doc in safe_load_all(f)]

    global_config = docs[0]
    variable_specs = docs[1]

    slocum_name_map = {
        source: variable_name
        for variable_name, specs in variable_specs.items()
        if "source" in specs
        for source in (
            specs["source"] if isinstance(specs["source"], list) else [specs["source"]]
        )
    }

    _log.debug("Slocum name mapping dict %s", slocum_name_map)

    # pyyaml loads datetime objects in local timezone as datetime.datetime objects.
    # We need to ensure that all datetime objects are in UTC timestamps for processing to work.
    for attr in ["valid_min", "valid_max"]:
        if attr in variable_specs["time"]["CF"]:
            variable_specs["time"]["CF"][attr] = _ensure_utc(
                variable_specs["time"]["CF"][attr]
            ).timestamp()

    config = dict(
        globals=global_config, variables=variable_specs, slocum=slocum_name_map
    )

    return config
