# glide (under construction)

Slocum underwater glider processing command line interface. 

This package produces quality controlled L2 and L3 datasets from real-time and delayed-time Slocum glider data. It can generate datasets that meet [IOOS Glider DAC](https://gliders.ioos.us/) standards. It requires that data are first converted to netCDF or csv using [`dbd2netcdf`](github.com/OSUGliders/dbd2netcdf) (or `dbd2csv`), a very fast Dinkum binary conversion tool. 

Our definitions of data processing levels are guided by [NASA](https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/data-processing-levels) and the [Spray data](https://spraydata.ucsd.edu/data-access). 

* **L0**: The binary files produced a Slocum gliders `.dbd`, `.sbd`, `.ebd`, `.tbd` or their compressed counterparts `.dcd`, ... etc. 
* **L1**: NetCDF or csv time series of flight and science data generated using `dbd2netcdf`. Usually named `glidername.dbd.nc` and `glidername.ebd.nc` or something similar.
* **L2**: Key science and flight variables are merged into a single time series. Basic quality control is applied. Some missing data are interpolated. Physical variables are derived. Other glider sensors (e.g. MicroRider, DVL) may also be assimilated. 
* **L3**: Includes gridded data products. 

## Usage

```
glide l2 osu685.sbd.nc osu684.tbd.nc --out-file=osu685.l2.nc
```

## Development

This package is developed with [`uv`](https://github.com/astral-sh/uv). 

After cloning this repository, genereate the virtual environment:
```
uv sync
```

Run tests:
```
uv run pytest -v
```

Format code:
```
uv run ruff format src tests
uv run ruff check --select I --fix
```

Type checking:
```
uv run mypy src tests
``` 

Try out glide on the test data:
```
uv run glide --log-level=debug l2 tests/data/osu684.sbd.csv tests/data/osu684.tbd.csv
```