#!/usr/bin/env python3
"""Migrate sweep results from JSON (v1) to CSV (v2) format.

Usage::

    # Preview savings without writing
    python parameters/migrate_results.py --dir results/ --dry-run

    # Migrate all results, keeping original JSON files
    python parameters/migrate_results.py --dir results/ --keep-json

    # Migrate all results, deleting original JSON files
    python parameters/migrate_results.py --dir results/

    # Migrate a single task
    python parameters/migrate_results.py --dir results/mnist_mlp/
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers (duplicated from sweep_utils to keep this script standalone)
# ---------------------------------------------------------------------------

def _round_float(x, sig_figs=6):
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, sig_figs - 1 - int(math.floor(math.log10(abs(x)))))


def _format_value(v, sig_figs=6):
    if v is None:
        return ""
    if isinstance(v, float):
        return _round_float(v, sig_figs)
    return v


# Redundant per-epoch fields that can be derived from the raw metrics
REDUNDANT_KEYS = {
    "max_val_acc", "max_acc_epoch",
    "min_val_loss", "min_loss_epoch",
    "min_val_perplexity", "min_perp_epoch",
}


def migrate_file(json_path, dry_run=False, keep_json=False, sig_figs=6):
    """Convert a single JSON run file to CSV.

    Returns (old_size, new_size) in bytes, or (old_size, None) for dry-run.
    """
    json_path = Path(json_path)
    old_size = json_path.stat().st_size

    with open(json_path) as f:
        data = json.load(f)

    config = data.get("config", {})
    summary = data.get("summary", {})
    history = data.get("history", [])

    # Collect column names (excluding redundant fields), preserving order
    columns = []
    for row in history:
        for key in row:
            if key not in columns and key not in REDUNDANT_KEYS:
                columns.append(key)

    csv_path = json_path.with_suffix(".csv")

    if dry_run:
        # Estimate size: header comment + CSV header + data rows
        metadata_line = "# " + json.dumps({"config": config, "summary": summary})
        header_line = ",".join(columns)
        est_size = len(metadata_line) + 1 + len(header_line) + 1
        for row in history:
            row_vals = [str(_format_value(row.get(k))) for k in columns]
            est_size += len(",".join(row_vals)) + 1
        return old_size, est_size

    metadata = {"config": config, "summary": summary}

    with open(csv_path, "w", newline="") as f:
        f.write("# " + json.dumps(metadata) + "\n")
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({k: _format_value(row.get(k), sig_figs) for k in columns})

    new_size = csv_path.stat().st_size

    if not keep_json:
        json_path.unlink()

    return old_size, new_size


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON sweep results to CSV format")
    parser.add_argument("--dir", type=str, required=True, help="Directory to search for run_*.json files")
    parser.add_argument("--dry-run", action="store_true", help="Preview savings without writing files")
    parser.add_argument("--keep-json", action="store_true", help="Keep original JSON files after migration")
    parser.add_argument("--sig-figs", type=int, default=6, help="Significant figures for float rounding (default: 6)")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)

    json_files = sorted(root.rglob("run_*.json"))
    if not json_files:
        print(f"No run_*.json files found in {root}")
        return

    total_old = 0
    total_new = 0
    migrated = 0

    for json_file in json_files:
        # Skip if CSV already exists (already migrated)
        csv_file = json_file.with_suffix(".csv")
        if csv_file.exists():
            print(f"  SKIP {json_file.relative_to(root)} (CSV already exists)")
            continue

        old_size, new_size = migrate_file(json_file, dry_run=args.dry_run,
                                           keep_json=args.keep_json,
                                           sig_figs=args.sig_figs)
        total_old += old_size
        total_new += new_size
        migrated += 1

        reduction = (1 - new_size / old_size) * 100 if old_size > 0 else 0
        action = "WOULD MIGRATE" if args.dry_run else "MIGRATED"
        print(f"  {action} {json_file.relative_to(root)}: "
              f"{old_size:,}B -> {new_size:,}B ({reduction:.0f}% reduction)")

    print(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:")
    print(f"  Files: {migrated}/{len(json_files)}")
    if total_old > 0:
        total_reduction = (1 - total_new / total_old) * 100
        print(f"  Size:  {total_old / 1024 / 1024:.1f}MB -> {total_new / 1024 / 1024:.1f}MB "
              f"({total_reduction:.0f}% reduction)")


if __name__ == "__main__":
    main()
