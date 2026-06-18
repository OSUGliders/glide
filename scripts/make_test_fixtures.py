#!/usr/bin/env python
"""Generate trimmed L1 CSV test fixtures from raw Slocum NetCDF files.

Usage:
    uv run python scripts/make_test_fixtures.py <dbd_nc> <ebd_nc> \\
        [--start "YYYY-MM-DD HH:MM"] [--end "YYYY-MM-DD HH:MM"] [--name PREFIX]

Outputs:
    tests/data/<PREFIX>.dbd.csv  — m_* source variables from core.yml
    tests/data/<PREFIX>.ebd.csv  — sci_* source variables + sci_generic_{a-l}

Times are interpreted as UTC. The defaults reproduce the sl685 fixtures
(2026-03-27 00:33–06:45 UTC).
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

DEFAULT_START = "2026-03-27 00:33"
DEFAULT_END = "2026-03-27 06:45"

GENERIC_KEEP = {f"sci_generic_{c}" for c in "abcdefghijkl"}


def parse_utc(s: str) -> float:
    """Parse an ISO-ish datetime string as UTC and return a posix timestamp."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


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


def extract(
    nc_path: str, time_var: str, out_path: Path, keep: set[str], t0: float, t1: float
) -> None:
    ds = nc.Dataset(nc_path)
    t = np.array(ds.variables[time_var][:])
    mask = ~np.isnan(t) & (t >= t0) & (t <= t1)
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
    parser.add_argument(
        "--start", default=DEFAULT_START, help='UTC start, e.g. "2026-05-14 18:00"'
    )
    parser.add_argument(
        "--end", default=DEFAULT_END, help='UTC end, e.g. "2026-05-14 19:00"'
    )
    parser.add_argument("--name", default="sl685", help="Output filename prefix")
    args = parser.parse_args()

    t0, t1 = parse_utc(args.start), parse_utc(args.end)
    sources = collect_sources(yaml.safe_load(CORE_YML.read_text()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting test fixtures ({args.start} – {args.end} UTC)...")
    extract(
        args.dbd_nc, "m_present_time", OUT_DIR / f"{args.name}.dbd.csv", sources, t0, t1
    )
    extract(
        args.ebd_nc,
        "sci_m_present_time",
        OUT_DIR / f"{args.name}.ebd.csv",
        sources | GENERIC_KEEP,
        t0,
        t1,
    )


if __name__ == "__main__":
    main()
