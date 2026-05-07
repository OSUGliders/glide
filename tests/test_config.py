import os
import tempfile
import textwrap

from glide import config
from glide.config import _apply_qc_overrides, _deep_merge

# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_deep_merge_nested():
    # Nested dicts are merged, not replaced — unoverridden keys survive
    base = {"cf": {"units": "m", "long_name": "Depth"}}
    override = {"cf": {"units": "dbar"}}
    result = _deep_merge(base, override)
    assert result["cf"]["units"] == "dbar"
    assert result["cf"]["long_name"] == "Depth"


def test_apply_qc_overrides_valid_min_max():
    # valid_min/max must be routed into the CF sub-dict, not the top level
    variables = {"temperature": {"CF": {"valid_min": -5.0, "valid_max": 50.0}}}
    result = _apply_qc_overrides(
        variables, {"temperature": {"valid_min": -2.0, "valid_max": 15.0}}
    )
    assert result["temperature"]["CF"]["valid_min"] == -2.0
    assert result["temperature"]["CF"]["valid_max"] == 15.0


def test_apply_qc_overrides_disallowed_key_ignored():
    # Users must not be able to overwrite CF metadata (e.g. long_name) via qc overrides
    variables = {"temperature": {"CF": {"long_name": "Temperature"}}}
    result = _apply_qc_overrides(variables, {"temperature": {"long_name": "Hacked"}})
    assert result["temperature"]["CF"]["long_name"] == "Temperature"


# ---------------------------------------------------------------------------
# Integration tests for load_config()
# ---------------------------------------------------------------------------


def test_load_config_merged_variables_present():
    # Regression: inserting an extra YAML document shifts the positional parsing
    # in load_config (docs[4]), silently dropping merged_variables entirely.
    conf = config.load_config()
    assert conf["merged_variables"], (
        "merged_variables should not be empty in default config"
    )
    assert "e_1" in conf["merged_variables"]
    assert "e_2" in conf["merged_variables"]


def test_load_config_instruments_present():
    # Regression: instruments live in the 6th YAML document. If the loader's
    # positional indexing is off by one, this dict is silently empty.
    conf = config.load_config()
    assert "instruments" in conf
    assert "instrument_ctd" in conf["instruments"], (
        "instrument_ctd should be loaded from the bundled config.yml"
    )
    ctd = conf["instruments"]["instrument_ctd"]
    assert ctd.get("make_model") == "Sea-Bird GPCTD"
    assert ctd.get("type") == "instrument"


def test_load_config_time_gps_has_anchor():
    # The anchor field drives which time points are kept in add_gps_fixes()
    conf = config.load_config()
    anchor = conf["variables"]["time_gps"].get("anchor")
    assert anchor is not None, "time_gps is missing 'anchor' field"
    assert "lat_gps" in anchor
    assert "lon_gps" in anchor


def test_load_config_time_valid_min_max_are_floats():
    # Datetime strings in core.yml must be converted to UTC timestamps
    conf = config.load_config()
    valid_min = conf["variables"]["time"]["CF"]["valid_min"]
    valid_max = conf["variables"]["time"]["CF"]["valid_max"]
    assert isinstance(valid_min, float), (
        "time valid_min should be a UTC timestamp float"
    )
    assert isinstance(valid_max, float), (
        "time valid_max should be a UTC timestamp float"
    )
    assert valid_min < valid_max


def _write_temp_config(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


def test_load_config_exclude_thermo():
    path = _write_temp_config("""\
        trajectory:
          name: test_glider
          attributes: {}
        netcdf_attributes: {}
        ---
        include:
          flight: true
          thermo: false
        ---
        qc: {}
        ---
        l1_variables: {}
        ---
        merged_variables: {}
    """)
    try:
        conf = config.load_config(path)
        assert not conf["include"]["thermo"]
        assert "salinity" not in conf["variables"]
        assert "pressure" in conf["variables"]  # core variables still present
    finally:
        os.unlink(path)


def test_load_config_qc_override_applied():
    path = _write_temp_config("""\
        trajectory:
          name: test_glider
          attributes: {}
        netcdf_attributes: {}
        ---
        include:
          flight: true
          thermo: true
        ---
        qc:
          temperature:
            valid_min: -2.0
            valid_max: 15.0
          lat:
            max_gap: 900
        ---
        l1_variables: {}
        ---
        merged_variables: {}
    """)
    try:
        conf = config.load_config(path)
        assert conf["variables"]["temperature"]["CF"]["valid_min"] == -2.0
        assert conf["variables"]["temperature"]["CF"]["valid_max"] == 15.0
        assert conf["variables"]["lat"]["max_gap"] == 900
    finally:
        os.unlink(path)


def test_load_config_companion_variable():
    path = _write_temp_config("""\
        trajectory:
          name: test_glider
          attributes: {}
        netcdf_attributes: {}
        ---
        include:
          flight: true
          thermo: true
        ---
        qc: {}
        ---
        l1_variables:
          hdop_gps:
            source: m_gps_uncertainty
            companion_dim: time_gps
            drop_from_l2: True
            dtype: f4
            CF:
              long_name: GPS horizontal dilution of precision
              units: "1"
        ---
        merged_variables: {}
    """)
    try:
        conf = config.load_config(path)
        assert "hdop_gps" in conf["variables"]
        spec = conf["variables"]["hdop_gps"]
        assert spec["companion_dim"] == "time_gps"
        assert spec["drop_from_l2"] is True
        assert conf["slocum"]["m_gps_uncertainty"] == "hdop_gps"
    finally:
        os.unlink(path)
