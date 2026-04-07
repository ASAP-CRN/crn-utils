#!/usr/bin/env python3
"""
Regression test for crn_utils.release_util.prep_release_metadata

This script runs prep_release_metadata twice on the same dataset:
1. with ds_path as a pathlib.Path
2. with ds_path as a str

If either call raises, or the import paths change, that’s a regression.
You are expected to edit the path placeholders below to match your setup.

Run as:
    python3 test_prep_release_metadata_regression.py

    Expected output:
    ============================= prep_release_metadata regression test =============================
    metadata_root : /path/to/asap-crn-cloud-dataset-metadata
    mapper_root   : /path/to/asap-ids   
    dataset       : lee-pmdbs-sn-rnaseq
    ...
    [1/2] Calling prep_release_metadata with ds_path as Path...
        ✓ completed without error (Path)
    [2/2] Calling prep_release_metadata with ds_path as str...
        ✓ completed without error (str)

    ✅ prep_release_metadata regression test PASSED (function accepts both Path and str for ds_path). 
    
"""

from __future__ import annotations

from pathlib import Path
import sys, os
import argparse

crn_utils_root = str(os.path.join(Path(__file__).resolve().parents[3], "crn-utils/src"))
sys.path.insert(0, str(crn_utils_root))
from crn_utils.constants import MOUSE_TABLES, PMDBS_TABLES  # noqa: F401 (imported to ensure these stay stable API)
from crn_utils.release_util import prep_release_metadata


def parse_long_dataset_name(long_dataset_name: str) -> tuple[str, str, str]:
    """
    Split a TEAM-SOURCE-DATASET[-subparts...] name into (team, source, dataset_name).
    """
    parts = long_dataset_name.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"dataset_name '{long_dataset_name}' must have at least 3 hyphen-separated parts "
            "(e.g. 'lee-pmdbs-sn-rnaseq')."
        )
    team = parts[0]
    source = parts[1]
    dataset_name = "-".join(parts[2:])
    return team, source, dataset_name


def run_prep_release_metadata_regression(
    long_dataset_name: str,
    schema_version: str,
) -> None:
    """
    Run a minimal regression test for prep_release_metadata.

    Parameters
    ----------
    long_dataset_name
        Hyphenated dataset identifier like 'lee-pmdbs-sn-rnaseq'
        or 'lee-pmdbs-bulk-rnaseq-mfg'.
    schema_version
        Schema version string, e.g. 'v3.3'.
    """
    team, source, dataset_name = parse_long_dataset_name(long_dataset_name)

    #  /path/to/asap-crn-cloud-dataset-metadata
    metadata_root = Path(
        os.path.join(
            Path(__file__).resolve().parents[3],
            "asap-crn-cloud-dataset-metadata",
        )
    )
    mapper_root = metadata_root

    # ------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------
    spatial = False
    suffix = "ids"

    # Conventional layout: <metadata_root>/datasets/<team-source-dataset>
    datasets_path = metadata_root / "datasets"
    ds_path = datasets_path / long_dataset_name

    map_path = mapper_root

    print("=== prep_release_metadata regression test ===")
    print(f"metadata_root : {metadata_root}")
    print(f"mapper_root   : {mapper_root}")
    print(f"dataset       : {long_dataset_name}")
    print(f"team          : {team}")
    print(f"source        : {source}")
    print(f"dataset_name  : {dataset_name}")
    print(f"schema        : {schema_version}")
    print(f"spatial       : {spatial}")
    print(f"suffix        : {suffix}")
    print("============================================")

    # ------------------------------------------------------------------
    # 1) Call with ds_path as a Path
    # ------------------------------------------------------------------
    print(f"\n[1/2] Calling prep_release_metadata with ds_path as Path...\n")
    prep_release_metadata(
        ds_path=ds_path,
        schema_version=schema_version,
        map_path=map_path,
        suffix=suffix,
        spatial=spatial,
        source=source,
    )
    print("    ✓ completed without error (Path)")
    print("============================================")

    # ------------------------------------------------------------------
    # 2) Call with ds_path as a str
    # ------------------------------------------------------------------
    ds_path_str = str(ds_path)
    print(f"\n[2/2] Calling prep_release_metadata with ds_path as str...\n")
    prep_release_metadata(
        ds_path=ds_path_str,
        schema_version=schema_version,
        map_path=map_path,
        suffix=suffix,
        spatial=spatial,
        source=source,
    )
    print("    ✓ completed without error (str)")
    print("============================================")

    print(
        f"\n✅ prep_release_metadata regression test PASSED for {long_dataset_name} "
        "(function accepts both Path and str for ds_path).\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_prep_release_metadata_regression",
        description=(
            "Run a regression test for crn_utils.release_util.prep_release_metadata\n\n"
            "The dataset name must be of the form team-source-dataset, e.g.:\n"
            "  lee-pmdbs-sn-rnaseq\n"
            "  lee-pmdbs-bulk-rnaseq-mfg\n"
            "It must have a gs://asap-raw-team-{dataset_name} bucket\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dataset_name",
        dest="dataset_name",
        default="lee-pmdbs-sn-rnaseq",
        help="Hyphenated dataset name, e.g. 'lee-pmdbs-sn-rnaseq' (default).",
    )

    parser.add_argument(
        "-s",
        "--schema_version",
        dest="schema_version",
        required=True,
        help="Schema version string, e.g. 'v3.3'.",
    )

    args = parser.parse_args()

    run_prep_release_metadata_regression(
        long_dataset_name=args.dataset_name,
        schema_version=args.schema_version,
    )
