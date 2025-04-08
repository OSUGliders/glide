from glide import config


def test_load_config() -> None:
    var_specs, source_map = config.load_config()
    assert "m_present_time" in source_map
    assert source_map["m_present_time"] == "time"
    assert var_specs["time"]["CF"]["units"] == "seconds since 1970-01-01T00:00:00Z"
