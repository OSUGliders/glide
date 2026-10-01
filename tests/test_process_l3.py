"""Tests for glide.process_l3, the assimilation of external microstructure."""

import numpy as np
import pytest
import xarray as xr

import glide.process_l3 as pl3
from glide.config import load_config


def _config(*names):
    """A config declaring only the named variables as merged_variables."""
    return dict(merged_variables={n: dict(CF={}) for n in names})


# --- geometry and bin size -------------------------------------------------


def test_eps_bin_size_reads_the_coordinate_spacing():
    ds = xr.Dataset(coords=dict(depth=("depth", [1.0, 3.0, 5.0, 7.0])))
    assert pl3._eps_bin_size(ds) == 2.0


def test_eps_bin_size_rejects_uneven_spacing():
    ds = xr.Dataset(coords=dict(depth=("depth", [1.0, 3.0, 6.0])))
    with pytest.raises(ValueError, match="not uniformly spaced"):
        pl3._eps_bin_size(ds)


def test_eps_bin_size_rejects_bins_not_starting_at_the_surface():
    # centres 2, 4, 6 imply a first edge at 1 m; glide bins always start at 0.
    ds = xr.Dataset(coords=dict(depth=("depth", [2.0, 4.0, 6.0])))
    with pytest.raises(ValueError, match="do not start at the surface"):
        pl3._eps_bin_size(ds)


@pytest.mark.parametrize("bin_size", [1.0, 3.0, 4.0])
def test_check_bin_size_refuses_anything_but_a_match(bin_size):
    # Even a clean multiple is refused: the data is already depth-averaged.
    with pytest.raises(ValueError, match="microstructure file is binned at 2 m"):
        pl3._check_bin_size(bin_size, 2.0)


def test_check_bin_size_accepts_a_match():
    pl3._check_bin_size(2.0, 2.0)


# --- depth mapping ---------------------------------------------------------


def test_depth_map_matches_centres():
    l3 = np.array([1.0, 3.0, 5.0, 7.0])
    eps_idx, l3_idx = pl3._depth_map(l3, np.array([1.0, 3.0, 5.0, 7.0]))
    np.testing.assert_array_equal(eps_idx, [0, 1, 2, 3])
    np.testing.assert_array_equal(l3_idx, [0, 1, 2, 3])


def test_depth_map_drops_bins_deeper_than_the_l3_grid():
    l3 = np.array([1.0, 3.0])
    eps_idx, l3_idx = pl3._depth_map(l3, np.array([1.0, 3.0, 5.0, 7.0]))
    np.testing.assert_array_equal(eps_idx, [0, 1])
    np.testing.assert_array_equal(l3_idx, [0, 1])


def test_depth_map_survives_a_gap_in_the_l3_grid():
    # bin_l2 builds z from the bins that hold data, so a level can be absent.
    # Everything either side must still land on the right level.
    l3 = np.array([1.0, 5.0, 7.0])  # the 3 m level is missing
    eps_idx, l3_idx = pl3._depth_map(l3, np.array([1.0, 3.0, 5.0, 7.0]))
    np.testing.assert_array_equal(eps_idx, [0, 2, 3])
    np.testing.assert_array_equal(l3_idx, [0, 1, 2])


# --- time coercion ---------------------------------------------------------


def test_to_datetime64_round_trips_both_provenances():
    # bin_l2 leaves float posix seconds; reading a file back gives datetimes.
    stamp = np.datetime64("2026-08-24T00:58:47.500", "ns")
    from_float = pl3._to_datetime64(np.array([stamp.astype("f8") / 1e9]))
    from_dates = pl3._to_datetime64(np.array([stamp]))
    assert from_float[0] == from_dates[0] == stamp


def test_to_datetime64_rejects_other_types():
    with pytest.raises(TypeError):
        pl3._to_datetime64(np.array(["nonsense"]))


# --- profile alignment -----------------------------------------------------


def test_eps_profiles_land_on_the_climbs_they_overlap(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1, 3])  # the two climbs

    assigned, fraction = pl3._assign_eps_profiles(l3, eps)

    np.testing.assert_array_equal(assigned, [1, 3])
    np.testing.assert_array_equal(l3.state.values[assigned], [2, 2])
    assert np.all(fraction == 1.0)


def test_profile_time_is_not_used_for_alignment(slocum_l3_2m, make_eps_bin):
    # The fixture puts profile_time in the preceding profile, as the real files
    # do. Aligning on it would pick a dive; maximum overlap must pick the climb.
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1, 3])

    misleading = (
        np.searchsorted(
            l3.profile_time_start.values, eps.profile_time.values, side="right"
        )
        - 1
    )
    assert l3.state.values[misleading].tolist() == [1, 1], "fixture is not misleading"

    assigned, _ = pl3._assign_eps_profiles(l3, eps)
    assert l3.state.values[assigned].tolist() == [2, 2]


def test_eps_profile_in_a_dive_window_is_assigned_normally(
    slocum_l3_2m, make_eps_bin, caplog
):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[0])  # a dive

    with caplog.at_level("WARNING"):
        assigned, fraction = pl3._assign_eps_profiles(l3, eps)

    assert assigned.tolist() == [0]
    assert fraction[0] == 1.0
    assert "dive" not in caplog.text.lower()


def test_duplicate_claims_keep_the_better_overlap(slocum_l3_2m, make_eps_bin, caplog):
    l3, _ = slocum_l3_2m
    full = make_eps_bin(l3, targets=[1])
    partial = make_eps_bin(l3, targets=[1], coverage=0.5, seed=1)
    # the partial profile has some bin times outside the window
    times = partial.time.values.copy()
    times[0, 0] = l3.profile_time_start.values[0].astype("M8[ns]")
    partial["time"] = (("profile", "depth"), times)
    eps = xr.concat([full, partial], dim="profile")

    with caplog.at_level("WARNING"):
        assigned, _ = pl3._assign_eps_profiles(l3, eps)

    assert assigned.tolist() == [1, -1]
    assert "claimed by microstructure profiles" in caplog.text


def test_non_overlapping_profile_is_skipped_with_a_warning(
    slocum_l3_2m, make_eps_bin, caplog
):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1, 3])
    times = eps.time.values.copy()
    times[1] = times[1] + np.timedelta64(365, "D")
    eps["time"] = (("profile", "depth"), times)

    with caplog.at_level("WARNING"):
        assigned, _ = pl3._assign_eps_profiles(l3, eps)

    assert assigned.tolist() == [1, -1]
    assert "does not overlap any L3 profile window" in caplog.text


def test_wholly_unrelated_file_is_an_error(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1, 3])
    eps["time"] = (
        ("profile", "depth"),
        eps.time.values + np.timedelta64(365, "D"),
    )
    with pytest.raises(ValueError, match="different glider or deployment"):
        pl3._assign_eps_profiles(l3, eps)


# --- variable selection ----------------------------------------------------


def test_only_declared_variables_are_selected(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1])
    assert pl3._select_eps_variables(l3, eps, _config("eps", "chi")) == ["eps", "chi"]


def test_declaring_a_glide_variable_is_refused(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1])
    with pytest.raises(ValueError, match="already exists in the L3 dataset"):
        pl3._select_eps_variables(l3, eps, _config("temperature"))


def test_no_declared_variable_present_is_an_error(slocum_l3_2m, make_eps_bin):
    # The shipped config declares e_1/e_2 for q files; neither is in an eps file.
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1])
    with pytest.raises(ValueError, match="None of the merged_variables"):
        pl3._select_eps_variables(l3, eps, _config("e_1", "e_2"))


def test_placeholder_units_are_replaced_by_the_config(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1])
    config = dict(merged_variables=dict(W=dict(CF=dict(units="m s-1"))))

    attrs = pl3._merge_attrs(eps, "W", config)

    assert attrs["units"] == "m s-1"


def test_file_attributes_win_over_the_config(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1])
    config = dict(merged_variables=dict(eps=dict(CF=dict(units="wrong"))))

    attrs = pl3._merge_attrs(eps, "eps", config)

    assert attrs["units"] == "W kg-1"


# --- end to end ------------------------------------------------------------


def test_merge_eps_bin_places_values_on_the_right_profiles(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1, 3], n_bins=60)

    out = pl3.merge_eps_bin(l3.copy(deep=True), eps, 2.0, _config("eps"))

    assert out.eps.dims == ("z", "profile_id")
    assert out.eps.dtype == np.float32
    # only the two climbs carry data
    covered = np.isfinite(out.eps.values).any(axis=0)
    np.testing.assert_array_equal(covered, [False, True, False, True])
    np.testing.assert_array_equal(out.eps_source_profile.values, [-1, 0, -1, 1])
    # and the values are the source values, on the matching bins
    np.testing.assert_allclose(out.eps.values[:60, 1], eps.eps.values[0], rtol=1e-6)


def test_merge_eps_bin_follows_qc_companions(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1])

    out = pl3.merge_eps_bin(l3.copy(deep=True), eps, 2.0, _config("eps"))

    assert out.eps_qc.dtype == np.int8
    assert out.eps.attrs["ancillary_variables"] == "eps_qc"
    np.testing.assert_array_equal(out.eps_qc.attrs["flag_values"], [0, 1, 2, 4, 9])
    # bins no microstructure reached are flagged missing, not unknown
    assert out.eps_qc.values[:, 0].tolist() == [9] * out.sizes["z"]


def test_merge_eps_bin_records_provenance(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1, 3])

    out = pl3.merge_eps_bin(l3.copy(deep=True), eps, 2.0, _config("eps"))

    assert out.attrs["microstructure_bin_size"] == 2.0
    assert out.attrs["microstructure_profiles_assigned"] == "2/2"
    assert out.attrs["microstructure_instrument_model"] == "MR1000RDL-EM"
    assert out.attrs["microstructure_instrument_sn"] == "435"
    np.testing.assert_allclose(
        out.eps_overlap_fraction.values, [np.nan, 1.0, np.nan, 1.0]
    )


def test_merge_eps_bin_refuses_a_mismatched_grid(slocum_l3_2m, make_eps_bin):
    l3, _ = slocum_l3_2m
    eps = make_eps_bin(l3, targets=[1], bin_size=4.0)
    with pytest.raises(ValueError, match="binned at 4 m"):
        pl3.merge_eps_bin(l3.copy(deep=True), eps, 2.0, _config("eps"))


def test_shipped_config_declares_the_microstructure_variables():
    declared = load_config()["merged_variables"]
    for name in ("eps", "eps_1", "eps_2", "chi", "chi_1", "chi_2", "T1", "T2"):
        assert name in declared, f"{name} is not declared in merged_variables"
    # the q file variables are still there
    assert "e_1" in declared and "e_2" in declared


# --- bin_q, which shares the depth mapping ---------------------------------


def test_bin_q_places_shallow_data_on_the_right_bins(slocum_l3_2m):
    # A q record covering only the top 20 m of a 750 m grid. groupby_bins returns
    # one value per populated bin, so a positional write would scatter these
    # values across the wrong depths.
    l3, _ = slocum_l3_2m
    ds = l3.copy(deep=True)

    start = ds.profile_time_start.values[1]
    end = ds.profile_time_end.values[1]
    time = np.arange(start, end, np.timedelta64(10, "s"))
    depth = np.linspace(1.0, 19.0, time.size)
    pressure = depth / 1.0197  # roughly dbar for these depths

    ds_q = xr.Dataset(
        data_vars=dict(
            e_1=("time", np.full(time.size, -8.0)),
            e_2=("time", np.full(time.size, -9.0)),
            pressure=("time", pressure),
        ),
        coords=dict(time=("time", time)),
    )

    config = dict(merged_variables=dict(e_1=dict(CF={}), e_2=dict(CF={})))
    out = pl3.bin_q(ds, ds_q, 2.0, config)

    shallow = out.e_1.values[:10, 1]
    assert np.all(np.isfinite(shallow))
    np.testing.assert_allclose(shallow, 1e-8, rtol=1e-6)
    # nothing may appear below the 20 m the q record covered
    assert np.all(np.isnan(out.e_1.values[11:, 1]))
