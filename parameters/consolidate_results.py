"""Consolidate sweep CSV results into Parquet for efficient analysis.

For each results/{task}/{optimizer}/itr_{N}/ directory, writes:
  - summary.parquet     : one row per run (config + final metrics), ~5k rows
  - trajectories.parquet: one row per (run, iteration),            ~5M rows

Usage::

    # Consolidate all results (default: results/)
    python consolidate_results.py

    # Custom results dir, 8 parallel workers
    python consolidate_results.py --results_dir /path/to/results --workers 8

    # Summary only (skip trajectory data)
    python consolidate_results.py --no_trajectories

    # Skip directories that already have parquet files
    python consolidate_results.py --skip_existing
"""

import argparse
import csv
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-file parsing
# ---------------------------------------------------------------------------

def _parse_run_csv(path: Path) -> tuple[dict, list[dict]] | None:
    """Parse a single run_N.csv file.

    Returns
    -------
    (metadata, rows) or None on error.

    metadata : dict with keys run_id, config (dict), summary (dict)
    rows     : list of dicts, one per iteration (includes run_id)
    """
    try:
        text = path.read_text()
        lines = text.split("\n")

        m = re.match(r"run_(\d+)\.csv$", path.name)
        run_id = int(m.group(1)) if m else -1

        # Line 1: JSON metadata comment
        config, summary = {}, {}
        if lines and lines[0].startswith("# "):
            try:
                meta = json.loads(lines[0][2:])
                config = meta.get("config", {})
                summary = meta.get("summary", {})
            except json.JSONDecodeError:
                pass

        # Remaining lines: CSV body (iteration-level data)
        csv_text = "\n".join(lines[1:])
        rows = []
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            parsed: dict = {"run_id": run_id}
            for k, v in row.items():
                if v == "" or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = int(v)
                    except ValueError:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
            rows.append(parsed)

        return {"run_id": run_id, "config": config, "summary": summary}, rows

    except Exception as exc:
        print(f"  WARNING: could not parse {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-directory consolidation
# ---------------------------------------------------------------------------

def consolidate_directory(
    itr_dir: Path,
    include_trajectories: bool = True,
    skip_existing: bool = False,
) -> tuple[int, int, int]:
    """Consolidate all run_*.csv files in *itr_dir* into Parquet.

    Returns
    -------
    (n_runs, n_trajectory_rows, n_skipped)
    """
    summary_path = itr_dir / "summary.parquet"
    traj_path = itr_dir / "trajectories.parquet"

    if skip_existing and summary_path.exists():
        if not include_trajectories or traj_path.exists():
            return 0, 0, 0  # already done

    run_files = sorted(
        itr_dir.glob("run_*.csv"),
        key=lambda p: int(re.search(r"(\d+)", p.name).group(1)),
    )
    if not run_files:
        return 0, 0, 0

    summary_rows: list[dict] = []
    traj_rows: list[dict] = []
    skipped = 0

    for path in run_files:
        result = _parse_run_csv(path)
        if result is None:
            skipped += 1
            continue
        metadata, rows = result

        # Flat summary row: run_id + config cols + summary cols
        summary_row = {"run_id": metadata["run_id"]}
        summary_row.update(metadata["config"])
        summary_row.update(metadata["summary"])
        summary_rows.append(summary_row)

        if include_trajectories:
            traj_rows.extend(rows)

    if summary_rows:
        tmp = summary_path.with_suffix(".parquet.tmp")
        pd.DataFrame(summary_rows).to_parquet(tmp, index=False)
        os.replace(tmp, summary_path)

    if include_trajectories and traj_rows:
        tmp = traj_path.with_suffix(".parquet.tmp")
        pd.DataFrame(traj_rows).to_parquet(tmp, index=False)
        os.replace(tmp, traj_path)

    if skipped:
        print(f"  WARNING: {skipped}/{skipped + len(summary_rows)} run files "
              f"in {itr_dir} could not be parsed")

    return len(summary_rows), len(traj_rows), skipped


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_itr_dirs(results_dir: Path) -> list[Path]:
    """Find all itr_* leaf directories under results_dir."""
    return sorted(results_dir.glob("*/*/itr_*"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate sweep CSV results into Parquet files."
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Base results directory (default: results/)",
    )
    parser.add_argument(
        "--no_trajectories", action="store_true",
        help="Skip writing trajectories.parquet (summary only)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip directories that already have parquet files",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results dir '{results_dir}' does not exist.")
        return

    itr_dirs = find_itr_dirs(results_dir)
    if not itr_dirs:
        print(f"No itr_* directories found under {results_dir}")
        return

    include_trajectories = not args.no_trajectories
    print(f"Found {len(itr_dirs)} directories")
    print(f"Writing: summary.parquet" + (", trajectories.parquet" if include_trajectories else ""))
    if args.skip_existing:
        print("Skipping directories that already have parquet files")
    print()

    total_runs = 0
    total_traj_rows = 0
    total_skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                consolidate_directory, d, include_trajectories, args.skip_existing
            ): d
            for d in itr_dirs
        }
        for i, future in enumerate(as_completed(futures), 1):
            d = futures[future]
            rel = d.relative_to(results_dir)
            try:
                n_runs, n_rows, n_skipped = future.result()
                total_runs += n_runs
                total_traj_rows += n_rows
                total_skipped += n_skipped
                if n_runs == 0 and args.skip_existing:
                    print(f"[{i}/{len(itr_dirs)}] SKIPPED  {rel}")
                else:
                    traj_str = f", {n_rows:,} traj rows" if include_trajectories else ""
                    skip_str = f" ({n_skipped} failed)" if n_skipped else ""
                    print(f"[{i}/{len(itr_dirs)}] OK       {rel}: {n_runs:,} runs{traj_str}{skip_str}")
            except Exception as exc:
                errors += 1
                print(f"[{i}/{len(itr_dirs)}] ERROR    {rel}: {exc}")

    print()
    print(f"Done. {total_runs:,} runs consolidated, {total_traj_rows:,} trajectory rows total.")
    if total_skipped:
        print(f"  {total_skipped} run files could not be parsed (see warnings above).")
    if errors:
        print(f"  {errors} directories had errors.")


if __name__ == "__main__":
    main()
