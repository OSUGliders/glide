# Example of dbd2netdf usage. Also used to generate our test data.
alias dbd2netCDF=$HOME/Tools/dbd2netcdf/bin/dbd2netCDF
cache=/Volumes/science/data/slocum-cache-files
data=/Volumes/science/data/SFMC/osu684
out=../tests/data
dbd2netCDF -o $out/osu684.sbd.nc -C $cache $data/0135000*.scd #$data/01350005.scd #$data/01350006.scd
dbd2netCDF -o $out/osu684.tbd.nc -C $cache $data/0135000*.tcd #$data/01350005.tcd #$data/01350006.tcd