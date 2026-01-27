# This generates our test data!

from pathlib import Path

import xarray as xr

out = Path("../tests/data").resolve()

# Commented out because I didn't keep the input files...
# xr.open_dataset(Path(out, "osu684.sbd.nc")).drop_dims("j").to_pandas().to_csv(
#     Path(out, "osu684.sbd.csv")
# )
# xr.open_dataset(Path(out, "osu684.tbd.nc")).drop_dims("j").to_pandas().to_csv(
#     Path(out, "osu684.tbd.csv")
# )

input_sbd = [
    "osu685-2025-056-0-27.sbd.nc",
    "osu685-2025-056-0-28.sbd.nc",
    "osu685-2025-056-0-29.sbd.nc",
    "osu685-2025-056-0-30.sbd.nc",
]
input_tbd = [
    "osu685-2025-056-0-27.tbd.nc",
    "osu685-2025-056-0-28.tbd.nc",
    "osu685-2025-056-0-29.tbd.nc",
    "osu685-2025-056-0-30.tbd.nc",
]

for f in input_sbd + input_tbd:
    xr.open_dataset(Path("../", f)).drop_dims("j").to_pandas().to_csv(
        Path(out, f.replace(".nc", ".csv"))
    )
