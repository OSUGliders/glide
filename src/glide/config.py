# Functions for handling the configuration file

import logging

from yaml import safe_load_all

_log = logging.getLogger(__name__)


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
    if "valid_min" in variable_specs["time"]["CF"]:
        variable_specs["time"]["CF"]["valid_min"] = variable_specs["time"]["CF"][
            "valid_min"
        ].timestamp()
    if "valid_max" in variable_specs["time"]["CF"]:
        variable_specs["time"]["CF"]["valid_max"] = variable_specs["time"]["CF"][
            "valid_max"
        ].timestamp()

    config = dict(
        globals=global_config, variables=variable_specs, slocum=slocum_name_map
    )

    return config
