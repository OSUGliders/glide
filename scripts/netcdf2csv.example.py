# This generates our test data!

from pathlib import Path

import xarray as xr

out = Path("../tests/data").resolve()

xr.open_dataset(Path(out, "osu684.sbd.nc")).drop_dims("j").to_pandas().to_csv(
    Path(out, "osu684.sbd.csv")
)
xr.open_dataset(Path(out, "osu684.tbd.nc")).drop_dims("j").to_pandas().to_csv(
    Path(out, "osu684.tbd.csv")
)
