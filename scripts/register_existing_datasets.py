#!/usr/bin/env python
"""Register existing datasets in the data registry.

This script scans existing data directories and registers them in the
centralized dataset registry for traceability.

Usage:
    python scripts/register_existing_datasets.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sda.data import DataRegistry
from sda.paths import get_project_root


def find_existing_datasets(data_root: Path) -> list[tuple[str, Path]]:
    """Find existing HDF5 datasets in the data directory.

    Returns:
        List of (name, path) tuples for datasets with train.h5
    """
    datasets = []

    for path in data_root.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            # Check if it has train.h5 (indicates a processed dataset)
            if (path / "train.h5").exists():
                datasets.append((path.name, path))

    return sorted(datasets, key=lambda x: x[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Register existing datasets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be registered without actually registering",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory to scan (default: project data/)",
    )
    args = parser.parse_args()

    # Get data directory
    if args.data_dir:
        data_root = args.data_dir
    else:
        data_root = get_project_root() / "data"

    print(f"Scanning: {data_root}")
    print()

    # Find existing datasets
    datasets = find_existing_datasets(data_root)

    if not datasets:
        print("No datasets found with train.h5")
        return

    print(f"Found {len(datasets)} datasets:")
    for name, path in datasets:
        h5_files = list(path.glob("*.h5"))
        print(f"  - {name}: {len(h5_files)} HDF5 files")

    if args.dry_run:
        print()
        print("Dry run - no changes made")
        return

    print()
    print("Registering datasets...")

    # Initialize registry
    registry = DataRegistry()

    for name, path in datasets:
        # Parse name to extract base name (remove any existing version suffix)
        base_name, existing_version = DataRegistry.parse_dataset_name(name)

        # Check if already registered
        try:
            existing = registry.get_latest(base_name)
            if existing.path == path:
                print(f"  {name}: Already registered as v{existing.version}")
                continue
        except KeyError:
            pass

        # Register as version 1
        try:
            dataset = registry.register_dataset(
                name=base_name,
                path=path,
                metadata={
                    "original_name": name,
                    "note": "Migrated from existing data directory",
                },
            )
            print(f"  {name}: Registered as {dataset.full_name}")
        except Exception as e:
            print(f"  {name}: Failed to register - {e}")

    print()
    print("Registration complete!")
    print(f"Registry file: {registry.datasets_file}")


if __name__ == "__main__":
    main()
