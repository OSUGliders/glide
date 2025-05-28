from glide import config


def test_load_config() -> None:
    conf = config.load_config()
    assert "m_present_time" in conf["slocum"]
    assert conf["slocum"]["m_present_time"] == "time"
    assert (
        conf["variables"]["time"]["CF"]["units"] == "seconds since 1970-01-01T00:00:00Z"
    )
    assert "globals" in conf.keys()
