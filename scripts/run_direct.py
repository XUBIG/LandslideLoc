#!/usr/bin/env python3
"""Recompute independent Direct localization from frozen answers and responses."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landslide_localization.runtime_paths import configure_numerical_threads

configure_numerical_threads()


def frozen_answers(paths, selected):
    """Read complete query responses without writing to the supplied cache."""
    from landslide_localization.answer_resolution import answer_key, normalized_answer
    from landslide_localization.experiment_config import read_json
    responses = {}
    for path in sorted(paths["direct_answer_cache"].glob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Frozen answer response must be an object: {path}")
        query = payload.get("query")
        if not isinstance(query, str) or not isinstance(payload.get("results"), list):
            raise ValueError(f"Invalid frozen answer response: {path}")
        key = answer_key(query)
        if key in responses and responses[key] != payload["results"]:
            raise ValueError(f"Conflicting frozen responses for {query!r}")
        responses[key] = payload["results"]
    required = {answer_key(answer) for row in selected if (answer := normalized_answer(row.get("answer")))}
    if required - set(responses):
        raise ValueError(f"Missing frozen responses for {len(required - set(responses))} selected answers")

    class FrozenAnswers:
        def __init__(self):
            self.stats = Counter()

        def search(self, query):
            self.stats["file_cache"] += 1
            return responses[answer_key(query)], "frozen_response"

    return FrozenAnswers()


def pipeline_inputs(paths, output):
    """Configure the spatial inputs and paper parameters."""
    from landslide_localization import preprocessing as core
    from landslide_localization.soft_utils import path_signature
    from landslide_localization.experiment_config import LOCAL_PARAMETERS
    disabled_gdb = output / "disabled_nhd.gdb"
    if paths["nhd_gdb"] is None and disabled_gdb.exists():
        raise ValueError(f"Disabled NHD sentinel unexpectedly exists: {disabled_gdb}")
    return {
        "gt_file": path_signature(paths["annotations"]),
        "candidate_file": None,
        "sus_raster": path_signature(paths["susceptibility_raster"]),
        "rainfall_dir": str(paths["rainfall_dir"]),
        "image_semantics": path_signature(paths["image_semantics"]),
        "gdb_path": path_signature(paths["nhd_gdb"]) if paths["nhd_gdb"] else str(disabled_gdb),
        "hydro_cache_only": False,
        "id_trigger_threshold_mode": "formula",
        "id_trigger_thresholds_mm_per_day": core.id_trigger_rainfall_configuration("formula", 0.6)["thresholds_mm_per_day"],
        "rain_sigmoid_scale": 0.6, "hydro_timeout_seconds": 180,
        "fine_reconstruction": {
            "road_pbf": path_signature(paths["national_road_pbf"]) if paths["national_road_pbf"] else {"missing": True},
            "pbf_directories": {key: core._pbf_directory_signature(paths[key]) for key in ("osm_state_dir", "road_state_dir", "hydro_state_dir")},
            "allow_cross_type_fallback_with_text_fine": False,
            "enable_fine_fallback": True, "source_policy": "text-first", "seed_tolerance_m": 50.0,
            "road_pbf_timeout_seconds": 3600, "local_parameters": dict(LOCAL_PARAMETERS),
        },
    }


def main():
    from landslide_localization.experiment_config import (
        add_common_arguments, index_records, load_paths, load_population,
        read_json, run_directory, validate_inputs, write_json,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    paths, _ = load_paths(args.config)
    validate_inputs(paths, "direct")
    population = load_population(paths, args.split, args.case_ids)
    answers = index_records(read_json(paths["direct_answers"]), "Direct answers")
    if set(answers) != set(population["gt"]):
        raise ValueError("Direct answer IDs must match the complete released annotation population")
    # The annotation release owns every field except the model's answer string.
    selected = []
    for case_id in population["ids"]:
        record = copy.deepcopy(population["gt"][case_id])
        answer = answers[case_id]
        if "coordinates" in answer and answer["coordinates"] != record["coordinates"]:
            raise ValueError(f"Direct answers contain conflicting reference coordinates: {case_id}")
        record["answer"] = answer.get("answer")
        selected.append(record)
    print(f"Direct inputs valid: {population['summary']['selected_count']} {args.split} cases", flush=True)
    if args.workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        parser.error("Parallel Direct execution requires Linux fork; use --workers 1 here")

    import rasterio
    from landslide_localization import direct_pipeline as direct
    from landslide_localization.soft_utils import CacheManager, configure_reference_modules
    output = run_directory(paths, args.output_dir, "direct")
    direct_dir, cache_root = output / "direct_only", output / "cache"
    client = frozen_answers(paths, selected)
    promap, fine_ref = configure_reference_modules(cache_root)
    resolved, _ = direct.resolve_direct_answers(selected, client, promap, 50)
    failures = [value["failure_reason"] for value in resolved.values()
                if str(value.get("failure_reason", "")).startswith("query_or_geometry_error")]
    if failures:
        raise ValueError(f"Frozen answer geometry failed: {failures[:3]}")
    cases, gt = population["cases"], population["gt"]
    by_id = {str(row["tweets_id"]): row for row in cases}
    stage1 = direct.direct_stage1_records(selected, by_id, resolved, promap)
    inputs = pipeline_inputs(paths, output)
    identity = {
        "selection": population["summary"], "inputs": inputs,
        "answer_geometry_key": direct.resolution_signature(stage1),
        "image_aggregation": "noisy_or", "w_s": 0.8, "w_r": 1.0,
        "raster_mask_policy": direct.RASTER_MASK_POLICY,
    }
    # JSON round-trip prevents integer threshold keys changing an identity on resume.
    identity = json.loads(json.dumps(identity))
    existing = direct.reusable_direct_result(direct_dir, identity, cases, gt)
    if existing is None and (output / "direct_run.pending.json").is_file():
        pending = read_json(output / "direct_run.pending.json")
        if pending.get("identity") != identity:
            raise ValueError("Existing partial Direct run has different input paths/settings; choose a new --output-dir")
    cache = CacheManager(cache_root, {**direct.fixed_manifest(), **identity})
    context = direct.prepare_pipeline(inputs, cache_root, promap, "noisy_or", 0.8, 1.0)
    context.update(direct_cache=cache, promap=promap, fine_ref=fine_ref, w_s=0.8, w_r=1.0, progress_every=10,
                   checkpoint_root=direct_dir / "case_metrics", checkpoint_identity=identity)
    public_stage1 = [direct.public_stage1_record(row) for row in stage1]
    write_json(output / "direct_run.pending.json", {"identity": identity})
    write_json(direct_dir / "stage1_test.json", public_stage1)
    rows = existing
    if rows is None:
        with rasterio.open(context["sus_path"]) as raster:
            low, high, count = direct.global_sus_range(raster, cache, context["sus_path"])
            if not high > low or count < 1:
                raise ValueError("Degenerate global susceptibility range")
            context.update(sus_min=low, sus_max=high)
            if args.workers == 1:
                rows = direct.direct_stage2_rows(cases, {row["tweets_id"]: row for row in stage1}, cache, promap, fine_ref, raster,
                    context["image_semantics"], context["pipeline_args"], low, high, context["rainfall_dir"], context["gdb_path"],
                    0.8, 1.0, 10, checkpoint_context=context)
        if args.workers > 1:
            rows = direct.direct_stage2_parallel_rows(cases, {row["tweets_id"]: row for row in stage1}, context, args.workers, args.split)
        rows = direct.freeze_metric_population(rows, cases, gt, direct.DIRECT_MODEL)
    stage1_by_id = {row["tweets_id"]: row for row in stage1}
    for row in rows:
        source = stage1_by_id[str(row["tweets_id"])]
        if row["stage1_evaluable"] is not source["evaluable"] or row["gt_inside_region"] is not source["in_region"]:
            raise ValueError(f"Stage I/II containment disagrees: {row['tweets_id']}")
    if existing is None:
        write_json(direct_dir / "stage2_test.json", rows)
        write_json(direct_dir / "completion.json", {"identity": identity, "n_test": len(cases)})
    print(f"Direct evaluation complete: {direct_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
