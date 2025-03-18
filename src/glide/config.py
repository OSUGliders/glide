# Functions for handling the configuration file

import logging

from yaml import safe_load

_log = logging.getLogger(__name__)


def load_config(file: str | None = None) -> tuple[dict, dict]:
    """Extract variable specifications from a yaml file."""
    if file is None:
        from importlib import resources

        file = str(resources.files("glide").joinpath("assets/config.yml"))

    with open(file) as f:
        config = safe_load(f)

    # Generate mapping from Slocum source name to dataset name
    name_map = {
        sn: name
        for name, specs in config.items()
        if "source" in specs
        for sn in (
            specs["source"] if isinstance(specs["source"], list) else [specs["source"]]
        )
    }

    _log.debug("Name mapping dict %s", name_map)
    return config, name_map
