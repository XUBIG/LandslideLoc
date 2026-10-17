"""Shared, lightweight input validation for the published experiment runners."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

from landslide_localization.runtime_paths import REPO_ROOT

DEFAULT_PATHS = {
    "annotations": "data/full/annotations.json",
    "candidates": "data/full/process/candidates.json",
    "city_boundary_overrides": "data/full/process/city_boundary_overrides.json",
    "image_semantics": "data/full/process/image_semantics.json",
    "direct_answers": "data/full/process/direct_answers.json",
    "direct_answer_cache": "data/full/process/direct_answer_cache",
    "development_split": "data/full/splits/development.json",
    "test_split": "data/full/splits/test.json",
    "split_summary": "data/full/splits/split_summary.json",
    "event_groups": "data/full/splits/event_groups.json",
    "susceptibility_raster": "data/external/susceptibility/n10_conus.tif",
    "rainfall_dir": "data/external/prism_rainfall",
    "osm_state_dir": "data/external/osm",
    "road_state_dir": "data/external/osm/road_only",
    "hydro_state_dir": "data/external/osm/hydro_only",
    "national_road_pbf": "data/external/osm/us-latest.osm.pbf",
    "nhd_gdb": None,
    "output_root": "outputs",
}
LOCAL_PARAMETERS = {
    "local_search_buffer_m": 2000.0,
    "local_endpoint_tolerance_m": 1.0,
    "local_seed_tolerance_m": 100.0,
    "local_memory_states": 1,
    "local_raw_policy": "diagnostic",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Write atomically; retain mtime when deterministic contents are unchanged."""
    from landslide_localization.output_paths import ensure_output_path
    path = ensure_output_path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def resolve_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_paths(config=None):
    if config:
        source = resolve_path(config)
    else:
        local = REPO_ROOT / "configs/paths.local.json"
        source = local if local.is_file() else REPO_ROOT / "configs/paths.example.json"
    values = read_json(source)
    if not isinstance(values, dict):
        raise ValueError(f"Path configuration must be an object: {source}")
    unknown = set(values) - set(DEFAULT_PATHS) - {"_comment"}
    if unknown:
        raise ValueError(f"Unknown path configuration keys: {sorted(unknown)}")
    merged = {**DEFAULT_PATHS, **values}
    paths = {key: resolve_path(merged[key]) if merged[key] is not None else None for key in DEFAULT_PATHS}
    if os.environ.get("LANDSLIDE_OUTPUT_ROOT"):
        paths["output_root"] = resolve_path(os.environ["LANDSLIDE_OUTPUT_ROOT"])
    output = paths["output_root"]
    if output is None or output == Path(output.anchor) or output == REPO_ROOT:
        raise ValueError("output_root must be a dedicated results directory")
    for key, path in paths.items():
        if key != "output_root" and path is not None and (path == output or output in path.parents or path in output.parents):
            raise ValueError(f"Output and input paths must not overlap: {key}={path}")
    os.environ["LANDSLIDE_OUTPUT_ROOT"] = str(output)
    # Fresh Python hydro workers must import the src-layout package as well.
    source_dir = str(Path(__file__).resolve().parents[1])
    python_path = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if source_dir not in python_path:
        os.environ["PYTHONPATH"] = os.pathsep.join([source_dir, *python_path])
    return paths, source


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=Path, help="Path JSON; defaults to paths.local.json, then paths.example.json.")
    parser.add_argument("--output-dir", type=Path, help="Run directory inside configured output_root.")
    parser.add_argument("--split", choices=("development", "test", "full"), default="test")
    parser.add_argument("--case-ids", nargs="+", help="Subset of the chosen split; space or comma separated IDs.")
    parser.add_argument("--workers", type=int, default=12)


def run_directory(paths, explicit, workflow):
    from landslide_localization.output_paths import ensure_output_path
    output = resolve_path(explicit) if explicit else paths["output_root"] / workflow
    return ensure_output_path(output)


def bind_split_paths(paths):
    # These constants are imported by value in the original scientific modules.
    from landslide_localization import soft_config, split_utils
    for module in (soft_config, split_utils):
        module.DEFAULT_DEVELOPMENT_SPLIT = paths["development_split"]
        module.DEFAULT_TEST_SPLIT = paths["test_split"]
        module.DEFAULT_SPLIT_SUMMARY = paths["split_summary"]
    if paths["event_groups"] != paths["split_summary"].parent / "event_groups.json":
        raise ValueError("event_groups must be event_groups.json beside split_summary")


def index_records(rows, label):
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must be a JSON list of objects")
    indexed = {}
    for row in rows:
        case_id = str(row.get("tweets_id", ""))
        if not case_id or case_id == "None" or case_id in indexed:
            raise ValueError(f"Missing or duplicate tweets_id in {label}: {case_id}")
        indexed[case_id] = row
    return indexed


def validate_inputs(paths, workflow):
    file_keys = ["annotations", "image_semantics", "development_split", "test_split", "split_summary", "event_groups", "susceptibility_raster"]
    directory_keys = ["rainfall_dir", "osm_state_dir", "road_state_dir", "hydro_state_dir"]
    if workflow == "full":
        file_keys += ["candidates", "city_boundary_overrides"]
    elif workflow == "direct":
        file_keys += ["direct_answers"]
        directory_keys += ["direct_answer_cache"]
    else:
        raise ValueError(f"Unknown workflow: {workflow}")
    if paths["national_road_pbf"] is not None:
        file_keys.append("national_road_pbf")
    if paths["nhd_gdb"] is not None:
        directory_keys.append("nhd_gdb")
    problems = []
    for key in file_keys:
        if paths[key] is None or not paths[key].is_file():
            problems.append(f"{key}: expected a file, got {paths[key]}")
    for key in directory_keys:
        if paths[key] is None or not paths[key].is_dir():
            problems.append(f"{key}: expected a directory, got {paths[key]}")
    if not problems:
        if not any(paths["rainfall_dir"].glob("*.tif")) and not any(paths["rainfall_dir"].rglob("*.bil")):
            problems.append("rainfall_dir has no rainfall .tif or .bil rasters")
        for key in ("osm_state_dir", "road_state_dir", "hydro_state_dir"):
            if not any(paths[key].glob("*.pbf")):
                problems.append(f"{key}: no .pbf files")
        if workflow == "direct" and not any(paths["direct_answer_cache"].glob("*.json")):
            problems.append("direct_answer_cache contains no response JSON files")
    if problems:
        raise ValueError("Input validation failed:\n  " + "\n  ".join(problems))


def load_population(paths, split="test", case_ids=None):
    bind_split_paths(paths)
    from landslide_localization.split_utils import load_split_document
    gt = index_records(read_json(paths["annotations"]), "annotations")
    for case_id, row in gt.items():
        coordinates = row.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            raise ValueError(f"Invalid reference coordinates: {case_id}")
        lon, lat = coordinates
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in coordinates) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError(f"Invalid reference coordinates: {case_id}")
    partitions, split_rows = {}, {}
    for name in ("development", "test"):
        path = paths[f"{name}_split"]
        _, ids = load_split_document(path)
        document = read_json(path)
        records = index_records(document.get("records"), f"{name} split records")
        if set(records) != set(ids):
            raise ValueError(f"{name} split records must match its case IDs")
        partitions[name] = ids
        split_rows.update(records)
    if set(partitions["development"]) & set(partitions["test"]) or set(split_rows) != set(gt):
        raise ValueError("Frozen development/test splits must be disjoint and partition all annotations")
    ids = list(gt) if split == "full" else partitions[split]
    if case_ids:
        requested = {piece.strip() for token in case_ids for piece in token.split(",") if piece.strip()}
        if not requested or not requested <= set(ids):
            raise ValueError(f"--case-ids must belong to {split}; unknown IDs: {sorted(requested - set(ids))}")
        ids = [case_id for case_id in ids if case_id in requested]
    return {
        "gt": gt, "ids": ids, "cases": [copy.deepcopy(split_rows[cid]) for cid in ids],
        "gt_subset": [gt[cid] for cid in ids],
        "summary": {"population_count": len(gt), "development_count": len(partitions["development"]),
                    "test_count": len(partitions["test"]), "split": split, "selected_count": len(ids),
                    "case_ids": sorted(ids), "subset": bool(case_ids)},
    }


