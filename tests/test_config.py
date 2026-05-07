import os
import tempfile
import textwrap

from glide import config
from glide.config import _apply_qc_overrides, _build_slocum_name_map, _deep_merge

# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_deep_merge_base_only():
    result = _deep_merge({"a": 1, "b": 2}, {})
    assert result == {"a": 1, "b": 2}


def test_deep_merge_override_wins():
    result = _deep_merge({"a": 1}, {"a": 99})
    assert result["a"] == 99


def test_deep_merge_nested():
    base = {"cf": {"units": "m", "long_name": "Depth"}}
    override = {"cf": {"units": "dbar"}}
    result = _deep_merge(base, override)
    assert result["cf"]["units"] == "dbar"
    assert result["cf"]["long_name"] == "Depth"


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"x": 2}}
    _deep_merge(base, override)
    assert base["a"]["x"] == 1


def test_apply_qc_overrides_valid_min_max():
    variables = {"temperature": {"CF": {"valid_min": -5.0, "valid_max": 50.0}}}
    result = _apply_qc_overrides(
        variables, {"temperature": {"valid_min": -2.0, "valid_max": 15.0}}
    )
    assert result["temperature"]["CF"]["valid_min"] == -2.0
    assert result["temperature"]["CF"]["valid_max"] == 15.0


def test_apply_qc_overrides_max_gap():
    variables = {"lat": {"max_gap": 600}}
    result = _apply_qc_overrides(variables, {"lat": {"max_gap": 900}})
    assert result["lat"]["max_gap"] == 900


def test_apply_qc_overrides_unknown_variable_ignored():
    variables = {"temperature": {"CF": {}}}
    # Should not raise; unknown variable is just warned
    result = _apply_qc_overrides(variables, {"nonexistent": {"valid_min": 0.0}})
    assert "nonexistent" not in result


def test_apply_qc_overrides_disallowed_key_ignored():
    variables = {"temperature": {"CF": {"long_name": "Temperature"}}}
    result = _apply_qc_overrides(variables, {"temperature": {"long_name": "Hacked"}})
    assert result["temperature"]["CF"]["long_name"] == "Temperature"


def test_build_slocum_name_map_single_source():
    variables = {"pressure": {"source": "sci_water_pressure"}}
    slocum = _build_slocum_name_map(variables)
    assert slocum["sci_water_pressure"] == "pressure"


def test_build_slocum_name_map_list_source():
    variables = {"time": {"source": ["m_present_time", "sci_m_present_time"]}}
    slocum = _build_slocum_name_map(variables)
    assert slocum["m_present_time"] == "time"
    assert slocum["sci_m_present_time"] == "time"


def test_build_slocum_name_map_excludes_pipeline_variables():
    variables = {
        "u": {"dtype": "f4"},  # no source — pipeline variable
        "pressure": {"source": "sci_water_pressure"},
    }
    slocum = _build_slocum_name_map(variables)
    assert "u" not in slocum.values() or "u" not in slocum
    assert "sci_water_pressure" in slocum


# ---------------------------------------------------------------------------
# Integration tests for load_config()
# ---------------------------------------------------------------------------


def test_load_config_top_level_keys():
    conf = config.load_config()
    assert set(conf.keys()) >= {
        "globals",
        "variables",
        "slocum",
        "merged_variables",
        "include",
    }


def test_load_config_core_variables_always_present():
    conf = config.load_config()
    core_vars = [
        "time",
        "lat",
        "lon",
        "lat_gps",
        "lon_gps",
        "pressure",
        "conductivity",
        "temperature",
        "pitch",
        "roll",
    ]
    for v in core_vars:
        assert v in conf["variables"], f"Core variable '{v}' missing from config"


def test_load_config_pipeline_variables_always_present():
    conf = config.load_config()
    pipeline_vars = [
        "time_gps",
        "lat_uv",
        "lon_uv",
        "u",
        "v",
        "time_uv",
        "dive_id",
        "climb_id",
        "state",
    ]
    for v in pipeline_vars:
        assert v in conf["variables"], f"Pipeline variable '{v}' missing from config"


def test_load_config_time_gps_has_anchor():
    conf = config.load_config()
    anchor = conf["variables"]["time_gps"].get("anchor")
    assert anchor is not None, "time_gps is missing 'anchor' field"
    assert "lat_gps" in anchor
    assert "lon_gps" in anchor


def test_load_config_thermo_variables_present_by_default():
    conf = config.load_config()
    thermo_vars = [
        "salinity",
        "SA",
        "CT",
        "density",
        "rho0",
        "depth",
        "z",
        "sound_speed",
        "N2",
    ]
    for v in thermo_vars:
        assert v in conf["variables"], f"Thermo variable '{v}' missing from config"


def test_load_config_thermo_variables_have_derived_from():
    conf = config.load_config()
    thermo_vars = [
        "salinity",
        "SA",
        "CT",
        "density",
        "rho0",
        "depth",
        "z",
        "sound_speed",
        "N2",
    ]
    for v in thermo_vars:
        assert "derived_from" in conf["variables"][v], (
            f"'{v}' missing 'derived_from' field"
        )


def test_load_config_track_qc_variables():
    conf = config.load_config()
    expected_qc = [
        "time",
        "lat",
        "lon",
        "pressure",
        "conductivity",
        "temperature",
        "salinity",
        "SA",
        "CT",
        "density",
        "rho0",
        "depth",
        "z",
        "sound_speed",
        "N2",
    ]
    for v in expected_qc:
        assert conf["variables"][v].get("track_qc"), (
            f"Expected track_qc: True for '{v}'"
        )


def test_load_config_slocum_key_mappings():
    conf = config.load_config()
    expected = {
        "m_present_time": "time",
        "m_lat": "lat",
        "m_lon": "lon",
        "m_gps_lat": "lat_gps",
        "m_gps_lon": "lon_gps",
        "sci_water_pressure": "pressure",
        "sci_water_cond": "conductivity",
        "sci_water_temp": "temperature",
    }
    for slocum_name, output_name in expected.items():
        assert conf["slocum"].get(slocum_name) == output_name, (
            f"Expected slocum['{slocum_name}'] == '{output_name}', "
            f"got {conf['slocum'].get(slocum_name)!r}"
        )


def test_load_config_time_valid_min_max_are_floats():
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


def test_load_config_merged_variables_present():
    # Regression: inserting extra YAML documents shifts positional parsing,
    # silently dropping merged_variables (docs[4]).
    conf = config.load_config()
    assert conf["merged_variables"], (
        "merged_variables should not be empty in default config"
    )
    assert "e_1" in conf["merged_variables"]
    assert "e_2" in conf["merged_variables"]


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
        assert "SA" not in conf["variables"]
        # Core variables must still be present
        assert "pressure" in conf["variables"]
    finally:
        os.unlink(path)


def test_load_config_exclude_flight():
    path = _write_temp_config("""\
        trajectory:
          name: test_glider
          attributes: {}
        netcdf_attributes: {}
        ---
        include:
          flight: false
          thermo: true
        ---
        qc: {}
        ---
        l1_variables: {}
        ---
        merged_variables: {}
    """)
    try:
        conf = config.load_config(path)
        assert not conf["include"]["flight"]
        assert "heading" not in conf["variables"]
        assert "battpos" not in conf["variables"]
        # Core flight variables (pitch, roll) are always present
        assert "pitch" in conf["variables"]
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
              valid_min: 0.5
              valid_max: 99.9
              observation_type: measured
        ---
        merged_variables: {}
    """)
    try:
        conf = config.load_config(path)
        assert "hdop_gps" in conf["variables"], (
            "Companion variable not loaded into variables"
        )
        spec = conf["variables"]["hdop_gps"]
        assert spec["companion_dim"] == "time_gps"
        assert spec["drop_from_l2"] is True
        assert spec["source"] == "m_gps_uncertainty"
        assert conf["slocum"]["m_gps_uncertainty"] == "hdop_gps"
    finally:
        os.unlink(path)


def test_load_config_l1_variable_conflict_does_not_overwrite_core():
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
          pressure:
            source: m_custom_pressure
            CF:
              long_name: Hacked pressure
        ---
        merged_variables: {}
    """)
    try:
        conf = config.load_config(path)
        # Core variable should not be overwritten by a conflicting l1_variable
        assert conf["variables"]["pressure"]["source"] == "sci_water_pressure"
    finally:
        os.unlink(path)
