#!/usr/bin/env python
"""Generate trimmed L1 CSV test fixtures from raw Slocum NetCDF files.

Usage:
    uv run python scripts/make_test_fixtures.py <dbd_nc> <ebd_nc>

Outputs:
    tests/data/sl685.dbd.csv  — m_* source variables from core.yml
    tests/data/sl685.ebd.csv  — sci_* source variables + sci_generic_{a-l}

The time window is fixed to 2026-03-27 00:33–06:45 UTC.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).parent.parent
CORE_YML = REPO_ROOT / "src/glide/assets/core.yml"
OUT_DIR = REPO_ROOT / "tests/data"

T0 = datetime(2026, 3, 27, 0, 33, tzinfo=timezone.utc).timestamp()
T1 = datetime(2026, 3, 27, 6, 45, tzinfo=timezone.utc).timestamp()

GENERIC_KEEP = {f"sci_generic_{c}" for c in "abcdefghijkl"}


def collect_sources(d: dict) -> set[str]:
    sources: set[str] = set()
    if not isinstance(d, dict):
        return sources
    if "source" in d:
        s = d["source"]
        sources.update(s if isinstance(s, list) else [s])
    for v in d.values():
        sources |= collect_sources(v)
    return sources


def extract(nc_path: str, time_var: str, out_path: Path, keep: set[str]) -> None:
    ds = nc.Dataset(nc_path)
    t = np.array(ds.variables[time_var][:])
    mask = ~np.isnan(t) & (t >= T0) & (t <= T1)
    avail = {v for v in ds.variables if ds.variables[v].dimensions == ("i",)}
    cols = sorted(avail & keep)
    data = {v: np.array(ds.variables[v][:])[mask] for v in cols}
    ds.close()

    df = pd.DataFrame(data)

    # Use 12 significant figures
    df.to_csv(out_path, index=False, float_format="%.12g")
    sz = os.path.getsize(out_path) / 1024
    print(f"  {out_path.name}: {len(df)} rows × {len(df.columns)} cols  ({sz:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dbd_nc", help="Path to the DBD (flight) NetCDF file")
    parser.add_argument("ebd_nc", help="Path to the EBD (science) NetCDF file")
    args = parser.parse_args()

    sources = collect_sources(yaml.safe_load(CORE_YML.read_text()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting test fixtures...")
    extract(args.dbd_nc, "m_present_time", OUT_DIR / "sl685.dbd.csv", sources)
    extract(
        args.ebd_nc,
        "sci_m_present_time",
        OUT_DIR / "sl685.ebd.csv",
        sources | GENERIC_KEEP,
    )


if __name__ == "__main__":
    main()
