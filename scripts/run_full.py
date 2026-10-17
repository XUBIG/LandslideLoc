#!/usr/bin/env python3
"""Recompute Full and five ablations from frozen process outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landslide_localization.runtime_paths import configure_numerical_threads

configure_numerical_threads()



def main():
    from landslide_localization.experiment_config import (
        add_common_arguments, bind_split_paths, index_records,
        load_paths, load_population, read_json, run_directory,
        validate_inputs, write_json,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--experiment", default="all", help="all (six ablations), full/E5, E1-E6, or comma-separated E1,E5.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    paths, _ = load_paths(args.config)
    validate_inputs(paths, "full")
    population = load_population(paths, args.split, args.case_ids)
    candidates = read_json(paths["candidates"])
    candidate_index = index_records(candidates, "candidates")
    if not set(population["gt"]) <= set(candidate_index):
        raise ValueError("Frozen candidates do not cover every released annotation")
    candidates = [row for row in candidates if str(row["tweets_id"]) in population["gt"]]
    print(f"Full inputs valid: {population['summary']['selected_count']} {args.split} cases", flush=True)

    bind_split_paths(paths)
    from landslide_localization import full_pipeline as full
    from landslide_localization.city_boundaries import apply_candidate_boundary_overrides
    from landslide_localization.spatial import load_candidate_dataset
    output = run_directory(paths, args.output_dir, "full")
    # Reject completed/partially exported results before touching input evidence.
    full.ensure_fresh_results(output)
    if (output / "COMPLETED.json").exists():
        raise FileExistsError(f"Full results already exist; choose a new --output-dir: {output}")
    # Boundary overrides validate against the entire released candidate set.
    candidate_path = output / "inputs/candidates.json"
    write_json(candidate_path, candidates)
    cases = load_candidate_dataset(str(candidate_path), str(paths["annotations"]))
    cases = apply_candidate_boundary_overrides(cases, paths["city_boundary_overrides"])
    full.selected_experiments(args.experiment)
    cache = output / "cache"
    argv = ["run_full.py", "--experiment", args.experiment,
            "--candidate-file", str(candidate_path), "--gt-file", str(paths["annotations"]),
            "--candidate-boundary-overrides", str(paths["city_boundary_overrides"]),
            "--sus-raster", str(paths["susceptibility_raster"]), "--rainfall-dir", str(paths["rainfall_dir"]),
            "--image-semantics", str(paths["image_semantics"]), "--cache-dir", str(cache),
            "--results-dir", str(output), "--workers", str(args.workers),
            "--w-s", "0.8", "--w-r", "1", "--rain-sigmoid-scale", "0.6",
            "--image-aggregation", "noisy_or", "--id-trigger-threshold-mode", "formula",
            "--hydro-timeout-seconds", "180", "--road-pbf-timeout-seconds", "3600",
            "--local-memory-states", "1", "--local-search-buffer-m", "2000",
            "--local-endpoint-tolerance-m", "1", "--local-seed-tolerance-m", "100",
            "--local-raw-policy", "diagnostic", "--local-text-cache-dir", str(cache / "local_text_reconstruction"),
            "--no-allow-cross-type-fallback-with-text-fine", "--enable-fine-fallback",
            "--gdb-path", str(paths["nhd_gdb"] or output / "disabled_nhd.gdb"),
            "--road-pbf", str(paths["national_road_pbf"] or output / "disabled_national.osm.pbf")]
    for key in ("osm_state_dir", "road_state_dir", "hydro_state_dir"):
        argv.extend(["--" + key.replace("_", "-"), str(paths[key])])
    if args.case_ids:
        subset = output / "inputs/selected_case_ids.json"
        write_json(subset, population["ids"])
        argv.extend(["--split-file", str(subset), "--run-name", args.split + "_subset"])
    else:
        argv.extend(["--split", args.split])
    result = full.main(argv[1:])
    from landslide_localization.soft_utils import CacheManager, configure_reference_modules
    from landslide_localization.stage1_reporting import full_stage1_records
    config = read_json(output / "run_config.json")
    full_cache = CacheManager(cache, config["inputs"])
    promap, _ = configure_reference_modules(cache)
    by_id = {str(row["tweets_id"]): row for row in cases}
    stage1 = full_stage1_records(population["gt_subset"], by_id, full_cache, promap)
    write_json(output / "stage1_test.json", stage1)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
