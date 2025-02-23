# glide

Slocum underwater glider processing command line interface. 

glide produces quality controlled L2 and L3 datasets from real-time and delayed-time Slocum glider data. It can generate datasets that meet [IOOS Glider DAC](https://gliders.ioos.us/) standards. It requires that data are first converted to netCDF or csv using [`dbd2netcdf`](github.com/OSUGliders/dbd2netcdf) (or `dbd2csv`), a very fast Dinkum binary conversion tool. 

Our definitions of data processing levels are guided by [NASA](https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/data-processing-levels), the [Spray data](https://spraydata.ucsd.edu/data-access), and our own experiences working with gliders. We define the following levels:

* **L0**: Binary files produced by Slocum gliders include `.dbd`, `.sbd`, `.ebd`, `.tbd` or their compressed counterparts `.dcd`, ... etc. 
* **L1**: NetCDF or csv timeseries of flight and science data generated using `dbd2netcdf`. Usually named `glidername.dbd.nc` and `glidername.ebd.nc` or something similar. No quality control is performed. Data have the same units as in masterdata.
* **L2**: Variable units are converted to oceanographic standards. Basic quality controls are applied. Some missing data are interpolated. Dead reckoned GPS positions are adjusted using surface GPS fixes. Thermodynamic variables are derived. Profiles are identified and tagged. Science and flight variables tagged in the configuration file are merged into a single file. 
* **L3**: The L2 data are binned in depth and separated into profiles. Optionally, MicroRider data processed using [`q2netcdf`](github.com/OSUGliders/q2netcdf) may also be merged.

Additionally, we provide the following intermediate processing outputs that may be useful for debugging issues:

* **L1B**: The L1 data are parsed and basic quality control is performed but science and flight data are not merged.

## Installation

Use pip:

```bash
pip install git+https://github.com/OSUGliders/glide
```

## Usage

`glide` requires a configuration file to properly process glider data. If you do not provide a file, the [default file](src/glide/assets/config.yml) will be used. The configuration file specifies which variables to extract from the L1 data and provides flags for unit conversion and quality controls. Variables that are not listed will not be extracted.

Assuming that you have already run `dbd2netcdf` over a directory of files (e.g. `dbd2netcdf -o glider.tbd.nc *.tbd`) you can apply the l2 processing using,


```
glide l2 glidername.sbd.nc glidername.tbd.nc --out-file=glidername.l2.nc --config-file=glidername.config.yml
```

To perform level 3 processing with a specific bin size, use:

```
glide l3 glidername.l2.nc --out-file=glidername.l3.nc --config-file=glidername.config.yml --bin-size=10
```

To view the help for the package, or a specific command, use:

```
glide --help
glide l2 --help
```

## Quality controls

We currently apply the following QC during L1 -> L2 processing:

* Drop missing or repeated timestamps. 
* Check data are within `valid_min` and `valid_max` limits specified in the configuration file. 
* Linearly adjusts dead reckoned longitude and latitude estimates between surface fixes. 

We plan to implement more of the [standard IOOS QC methods](https://cdn.ioos.noaa.gov/media/2017/12/Manual-for-QC-of-Glider-Data_05_09_16.pdf) in the future.

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

By default this will produce a file `slocum.l2.nc`. 