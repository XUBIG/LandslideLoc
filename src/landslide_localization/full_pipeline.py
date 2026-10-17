"""Full localization and five ablation experiments."""

from __future__ import annotations
from landslide_localization import preprocessing
from landslide_localization.preprocessing import (
    _locked_local_osm_load,
    add_fine_preprocessing_arguments,
    global_sus_range,
    id_trigger_rainfall_configuration,
    positive_finite_cli_float,
    preprocess_case,
    resolved_input,
    validate_fine_preprocessing_args,
)

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Optional,
)
import numpy as np
import rasterio
from tqdm import tqdm
from landslide_localization.soft_config import (
    CANDIDATE_REGION_POLICY_VERSION,
    NUMERICAL_PRECISION,
    RASTER_MASK_POLICY,
    DEFAULT_CANDIDATE_FILE,
    DEFAULT_GDB_PATH,
    DEFAULT_GT_FILE,
    DEFAULT_ID_TRIGGER_THRESHOLD_MODE,
    DEFAULT_IMAGE_SEMANTICS,
    DEFAULT_RAIN_SIGMOID_SCALE,
    DEFAULT_RAINFALL_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SUS_RASTER,
    DEFAULT_W_R,
    DEFAULT_W_S,
    EXPERIMENT_ORDER,
    LINE_BUFFER_M,
    MAX_CANDIDATE_AREA_KM2,
    POINT_BUFFER_M,
    PROCESSING_VERSION,
    TOP_K_VALUES,
    fixed_manifest,
)
from landslide_localization.soft_utils import (
    CacheManager,
    configure_reference_modules,
    evaluate_score,
    experiment_scores,
    fusion_result_metadata,
    missing_metric_row,
    path_signature,
    require_unit_interval,
    selected_experiments as original_selected_experiments,
    validate_soft_or_weights,
    weighted_soft_or,
    write_json,
)
from landslide_localization.split_utils import safe_run_name, select_cases
from landslide_localization.city_boundaries import (
    apply_candidate_boundary_overrides,
    boundary_override_configuration,
)
from landslide_localization import fine_area
from landslide_localization.fine_area import _pbf_directory_signature


RECONSTRUCTION_VERSION = "soft_fine_local_textfirst_v2"
PROCESSING_VERSION = f"{PROCESSING_VERSION}_{RECONSTRUCTION_VERSION}"
RESULT_FILENAMES = [
    "ablation_case_metrics.json", "case_processing_failures.json",
    "run_config.json", "COMPLETED.json",
]


def selected_experiments(value: str) -> list[str]:
    selected = {
        experiment
        for token in value.split(",")
        for experiment in original_selected_experiments(token)
    }
    return [experiment for experiment in EXPERIMENT_ORDER if experiment in selected]


def unit_interval_cli_weight(value: str) -> float:
    try:
        parsed = float(value)
        validate_soft_or_weights(parsed, DEFAULT_W_R)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Soft-OR weights must be finite values in [0, 1]; got {value!r}"
        ) from exc
    return parsed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Soft-OR ablations with validated local text-first fine reconstruction."
    )
    parser.add_argument(
        "--experiment",
        default="all",
        help="full/E5, E1-E6, all, or a comma-separated list (E4,E5,E6)",
    )
    parser.add_argument("--candidate-file", default=str(DEFAULT_CANDIDATE_FILE))
    parser.add_argument(
        "--candidate-boundary-overrides",
        default=None,
        help="City-boundary overrides for the released candidate regions.",
    )
    parser.add_argument("--gt-file", default=str(DEFAULT_GT_FILE))
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split",
        choices=("development", "test", "full"),
        default="test",
        help="Evaluation population (default: frozen test).",
    )
    split_group.add_argument(
        "--split-file",
        help="JSON file containing custom case IDs; overrides --split.",
    )
    parser.add_argument(
        "--run-name",
        help="Safe output/metadata name for a custom split (default: split filename stem).",
    )
    parser.add_argument("--sus-raster", default=str(DEFAULT_SUS_RASTER))
    parser.add_argument("--rainfall-dir", default=str(DEFAULT_RAINFALL_DIR))
    parser.add_argument("--image-semantics", default=str(DEFAULT_IMAGE_SEMANTICS))
    parser.add_argument(
        "--cache-dir",
        help="Writable cache directory for this run.",
    )
    parser.add_argument("--gdb-path", default=str(DEFAULT_GDB_PATH))
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Output directory; "
            "existing results are never overwritten."
        ),
    )
    parser.add_argument(
        "--w-s",
        type=unit_interval_cli_weight,
        default=DEFAULT_W_S,
        help="Susceptibility weight used in weighted Soft-OR (default: %(default)s).",
    )
    parser.add_argument(
        "--w-r",
        type=unit_interval_cli_weight,
        default=DEFAULT_W_R,
        help="Rainfall weight used in weighted Soft-OR (default: %(default)s).",
    )
    parser.add_argument(
        "--id-trigger-threshold-mode",
        "--id-threshold-mode",
        dest="id_trigger_threshold_mode",
        choices=("formula", "fixed"),
        default=DEFAULT_ID_TRIGGER_THRESHOLD_MODE,
        help=(
            "Choose formula thresholds 13.04*D^-0.44 (default) or the historical "
            "fixed ID_TRIGGER_THRESHOLDS."
        ),
    )
    parser.add_argument(
        "--rain-sigmoid-scale",
        type=positive_finite_cli_float,
        default=DEFAULT_RAIN_SIGMOID_SCALE,
        help="Positive scale a in R=sigmoid((E-1)/a) (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Case worker processes for all area tiers, including states (default: 1).",
    )
    parser.add_argument(
        "--max-cases", type=int, default=-1, help="-1 runs the full configured dataset"
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--hydro-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--hydro-cache-only",
        action="store_true",
        help="Use validated OSMnx cache hits but fail fast on network cache misses.",
    )
    add_fine_preprocessing_arguments(parser)
    args = parser.parse_args(argv)
    args.workflow = "full"
    if args.workers < 1:
        parser.error("--workers must be positive")
    validate_fine_preprocessing_args(parser, args)
    return args


def assert_under_soft(path: Path) -> Path:
    from landslide_localization.output_paths import ensure_output_path

    return ensure_output_path(path)


def ensure_fresh_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    collisions = [
        results_dir / name for name in RESULT_FILENAMES if (results_dir / name).exists()
    ]
    if collisions:
        listing = "\n".join(str(path) for path in collisions)
        raise FileExistsError(
            "Refusing to overwrite existing experiment results. Choose a new --results-dir.\n"
            + listing
        )


def range_accumulator() -> dict[str, dict[str, Optional[float]]]:
    return {
        name: {"min": None, "max": None}
        for name in (
            "s_norm",
            "r_id_trigger",
            "weighted_s_norm",
            "weighted_r_id_trigger",
            "h_soft_or",
            "F_fine",
        )
    }


def update_range(
    accumulator: dict[str, dict[str, Optional[float]]], name: str, array: np.ndarray
) -> None:
    valid = np.isfinite(array)
    if not np.any(valid):
        return
    minimum = float(np.min(array[valid]))
    maximum = float(np.max(array[valid]))
    current = accumulator[name]
    current["min"] = (
        minimum if current["min"] is None else min(float(current["min"]), minimum)
    )
    current["max"] = (
        maximum if current["max"] is None else max(float(current["max"]), maximum)
    )


def process_ablation_case(
    item,
    src,
    promap,
    fine_ref,
    image_semantics,
    cache,
    args,
    sus_min,
    sus_max,
    rainfall_dir,
    gdb_path,
    legacy_cache,
    experiments,
):
    """Run one case locally; return only small evaluation records, never rasters."""
    result = {"metric_rows": [], "failures": []}
    metric_rows, failures = result["metric_rows"], result["failures"]
    observed = result["observed"] = range_accumulator()
    uses_weighted_fusion = any(
        e in {"E3_sus_rain_softor", "E5_full_softor_fine"} for e in experiments
    )
    tweets_id = str(item.get("tweets_id"))
    try:
        context, case_text_rows, error = preprocess_case(
            item,
            src,
            promap,
            fine_ref,
            image_semantics,
            cache,
            args,
            sus_min,
            sus_max,
            rainfall_dir,
            gdb_path,
            legacy_cache,
        )
        if error or context is None:
            reason = error or "unknown_preprocessing_failure"
            failures.append({"tweets_id": tweets_id, "reason": reason})
            metric_rows.extend(
                missing_metric_row(item, experiment, reason)
                for experiment in experiments
            )
            return result

        s_norm = context["s_norm"]
        rain = context["rain"]
        fine = context["fine"]
        update_range(observed, "s_norm", s_norm)
        update_range(observed, "r_id_trigger", rain)
        update_range(observed, "F_fine", np.where(context["valid_mask"], fine, np.nan))
        if uses_weighted_fusion:
            weighted_s = args.w_s * s_norm
            weighted_r = args.w_r * rain
            sr = weighted_soft_or(s_norm, rain, args.w_s, args.w_r)
            if np.any(np.isfinite(sr)):
                require_unit_interval("w_s*S_norm", weighted_s)
                require_unit_interval("w_r*R_id_trigger", weighted_r)
                require_unit_interval("H_weighted_softOR", sr)
                update_range(observed, "weighted_s_norm", weighted_s)
                update_range(observed, "weighted_r_id_trigger", weighted_r)
                update_range(observed, "h_soft_or", sr)

        scores = experiment_scores(
            s_norm,
            rain,
            fine,
            experiments,
            w_s=args.w_s,
            w_r=args.w_r,
        )
        base = {
            "tweets_id": tweets_id,
            "candidate_area_tier": context.get("candidate_area_tier"),
            "candidate_area_km2": context.get("candidate_area_km2"),
            "gt_inside_region": context.get("gt_inside_region"),
        }
        for experiment, score in scores.items():
            row, _ = evaluate_score(
                experiment,
                score,
                context["transform"],
                context.get("gt_lon"),
                context.get("gt_lat"),
                base,
                promap,
            )
            metric_rows.append(row)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        failures.append(
            {
                "tweets_id": tweets_id,
                "reason": reason,
                "traceback": traceback.format_exc(),
            }
        )
        metric_rows.extend(
            missing_metric_row(item, experiment, reason) for experiment in experiments
        )
    return result


_WORKER_STATE = None


def _initialize_ablation_worker(config, local_cache_lock):
    global _WORKER_STATE
    args = argparse.Namespace(**config["args"])
    cache = CacheManager(
        Path(config["cache_root"]), config["compatibility"], rebuild=args.rebuild_cache,
    )
    promap, fine_ref = configure_reference_modules(cache.root)
    preprocessing._LOCAL_CACHE_LOCK = local_cache_lock
    fine_area.local_reconstruction.LocalOSMStore.load_local_osm_for_state = (
        _locked_local_osm_load
    )
    _WORKER_STATE = (config, args, cache, promap, fine_ref)


def _ablation_worker(task):
    item, per_images = task
    config, args, cache, promap, fine_ref = _WORKER_STATE
    hits, misses = cache.hits, cache.misses
    started = time.perf_counter()
    case_id = str(item.get("tweets_id"))
    print(f"[{case_id}] worker={os.getpid()} started", flush=True)
    # Rasterio/GDAL handles remain process-local and close after every case.
    with rasterio.open(config["sus_path"]) as src:
        result = process_ablation_case(
            item,
            src,
            promap,
            fine_ref,
            {case_id: per_images},
            cache,
            args,
            config["sus_min"],
            config["sus_max"],
            Path(config["rainfall_dir"]),
            Path(config["gdb_path"]) if config["gdb_path"] else None,
            None,
            config["experiments"],
        )
    result.update(
        cache_hits=cache.hits - hits,
        cache_misses=cache.misses - misses,
    )
    print(
        f"[{case_id}] worker={os.getpid()} completed in {time.perf_counter() - started:.1f}s; "
        f"failures={len(result['failures'])}",
        flush=True,
    )
    return result


def _case_scheduling_keys(item):
    # Cases sharing an administrative AOI may read its data concurrently.
    # Keep duplicate IDs separate because they write to the same case cache.
    return {("case", str(item.get("tweets_id")))}


def _parallel_case_results(cases, image_semantics, executor, workers):
    """Bound in-flight tasks and yield completed cases without head-of-line blocking."""
    pending = [
        (index, item, _case_scheduling_keys(item)) for index, item in enumerate(cases)
    ]
    running = {}
    while pending or running:
        active = (
            set().union(*(entry[2] for entry in running.values())) if running else set()
        )
        index = 0
        while index < len(pending) and len(running) < workers:
            entry = pending[index]
            if entry[2] & active:
                index += 1
                continue
            pending.pop(index)
            _, item, keys = entry
            future = executor.submit(
                _ablation_worker,
                (item, image_semantics.get(str(item.get("tweets_id")), [])),
            )
            running[future] = entry
            active.update(keys)
        finished, _ = wait(running, return_when=FIRST_COMPLETED)
        for future in finished:
            index, item, _ = running.pop(future)
            # A process crash must fail the run visibly; completed case caches survive.
            yield index, future.result()


def run_parallel_cases(cases, image_semantics, config, workers):
    context = multiprocessing.get_context("spawn")
    local_cache_lock = context.Lock()
    results = [None] * len(cases)
    if not cases:
        return results
    with ProcessPoolExecutor(
        max_workers=min(workers, len(cases)),
        mp_context=context,
        initializer=_initialize_ablation_worker,
        initargs=(config, local_cache_lock),
    ) as executor:
        completed = _parallel_case_results(cases, image_semantics, executor, workers)
        for index, result in tqdm(
            completed,
            total=len(cases),
            desc=f"Soft fine-grained preprocessing/evaluation ({workers} workers)",
        ):
            results[index] = result
    # Preserve the original dataset order in all output tables.
    return results


def build_preprocessing_compatibility(
    args,
    cache_dir,
    candidate_file,
    gt_file,
    sus_path,
    rainfall_dir,
    image_path,
    gdb_path,
    legacy_cache,
    *,
    isolated_cache=False,
):
    """Record the scientific inputs used to name this run's cached arrays."""
    active_id_thresholds = id_trigger_rainfall_configuration(
        args.id_trigger_threshold_mode,
        args.rain_sigmoid_scale,
    )["thresholds_mm_per_day"]
    upstream_compatibility = {
        **fixed_manifest(),
        "candidate_region_policy_version": CANDIDATE_REGION_POLICY_VERSION,
        "id_trigger_threshold_mode": args.id_trigger_threshold_mode,
        "id_trigger_thresholds_mm_per_day": active_id_thresholds,
        "id_trigger_threshold_formula": (
            "13.04*D^-0.44" if args.id_trigger_threshold_mode == "formula" else None
        ),
        "rain_sigmoid_scale": args.rain_sigmoid_scale,
        "candidate_file": path_signature(candidate_file),
        "gt_file": path_signature(gt_file),
        "gt_coordinate_policy": "metadata_coordinates_v1",
        "sus_raster": path_signature(sus_path),
        "rainfall_dir": str(rainfall_dir),
        "image_semantics": path_signature(image_path),
        "gdb_path": path_signature(gdb_path) if gdb_path else None,
        "point_buffer_m": POINT_BUFFER_M,
        "line_buffer_m": LINE_BUFFER_M,
        "max_candidate_area_km2": MAX_CANDIDATE_AREA_KM2,
        "hydro_timeout_seconds": args.hydro_timeout_seconds,
        "hydro_cache_only": bool(args.hydro_cache_only),
    }
    if getattr(args, "candidate_boundary_overrides", None):
        args.candidate_boundary_overrides = str(
            resolved_input(args.candidate_boundary_overrides)
        )
        upstream_compatibility["candidate_boundary_resolution"] = (
            boundary_override_configuration(
                args.candidate_boundary_overrides,
            )
        )
    reconstruction_configuration = {
        "version": RECONSTRUCTION_VERSION,
        "category_mapping": {"road": "road", "water": "river"},
        "failure_policy": "unusable_text_excluded_then_per_category_fallback",
        "scope": "all_eligible_text_fine_features_regardless_of_candidate_area_tier",
    }
    args.cache_dir = str(assert_under_soft(Path(cache_dir)))
    for name in (
        "road_pbf",
        "osm_state_dir",
        "road_state_dir",
        "hydro_state_dir",
        "local_text_cache_dir",
    ):
        setattr(args, name, str(resolved_input(getattr(args, name), must_exist=False)))
    args.local_text_cache_dir = str(cache_dir / "local_text_cache")
    reconstruction_configuration.update(
        source_policy="text-first",
        allow_cross_type_fallback_with_text_fine=args.allow_cross_type_fallback_with_text_fine,
        enable_fine_fallback=args.enable_fine_fallback,
        local_parameters={
            name: getattr(args, name)
            for name in (
                "local_search_buffer_m",
                "local_endpoint_tolerance_m",
                "local_seed_tolerance_m",
                "local_memory_states",
                "local_raw_policy",
                "local_text_cache_dir",
            )
        },
        road_pbf=(path_signature(Path(args.road_pbf)) if Path(args.road_pbf).is_file()
                  else {"path": str(args.road_pbf), "missing": True}),
        road_pbf_timeout_seconds=args.road_pbf_timeout_seconds,
        pbf_directories={
            name: _pbf_directory_signature(Path(getattr(args, name)))
            for name in ("osm_state_dir", "road_state_dir", "hydro_state_dir")
        },
    )
    upstream_compatibility = {
        **upstream_compatibility,
        "fine_reconstruction": reconstruction_configuration,
    }
    upstream_compatibility["image_aggregation"] = args.image_aggregation
    return upstream_compatibility


def initialize_preprocessing_caches(args, cache_dir, upstream_compatibility, legacy_cache=None):
    cache = CacheManager(cache_dir, upstream_compatibility, rebuild=args.rebuild_cache)
    return cache, None


def initialize_preprocessing_modules(args, cache):
    return configure_reference_modules(cache.root)


def main(argv=None) -> None:
    started = time.perf_counter()
    args = parse_args(argv)
    args.w_s, args.w_r = validate_soft_or_weights(args.w_s, args.w_r)
    experiments = selected_experiments(args.experiment)
    rainfall_configuration = id_trigger_rainfall_configuration(
        args.id_trigger_threshold_mode, args.rain_sigmoid_scale,
    )
    split_name = safe_run_name(args.run_name or args.split)
    results_dir = assert_under_soft(resolved_input(
        args.results_dir or str(DEFAULT_RESULTS_DIR / split_name), must_exist=False,
    ))
    ensure_fresh_results(results_dir)
    cache_dir = assert_under_soft(resolved_input(
        args.cache_dir or str(results_dir / "cache"), must_exist=False,
    ))
    candidate_file = resolved_input(args.candidate_file)
    gt_file = resolved_input(args.gt_file)
    sus_path = resolved_input(args.sus_raster)
    rainfall_dir = resolved_input(args.rainfall_dir)
    image_path = resolved_input(args.image_semantics)
    gdb_candidate = resolved_input(args.gdb_path, must_exist=False)
    gdb_path = gdb_candidate if gdb_candidate.exists() else None
    compatibility = build_preprocessing_compatibility(
        args, cache_dir, candidate_file, gt_file, sus_path, rainfall_dir,
        image_path, gdb_path, None,
    )
    cache, _ = initialize_preprocessing_caches(args, cache_dir, compatibility)
    promap, fine_ref = initialize_preprocessing_modules(args, cache)
    all_cases = promap.load_candidate_dataset(str(candidate_file), str(gt_file))
    if args.candidate_boundary_overrides:
        all_cases = apply_candidate_boundary_overrides(all_cases, args.candidate_boundary_overrides)
    cases, selection = select_cases(
        all_cases, split=args.split,
        split_file=resolved_input(args.split_file) if args.split_file else None,
        run_name=args.run_name,
    )
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    print(f"Full: {len(cases)} {selection.name} cases; {len(experiments)} experiments; {args.workers} workers", flush=True)
    image_semantics = promap.load_image_semantics(str(image_path))
    with rasterio.open(sus_path) as src:
        sus_min, sus_max, sus_count = global_sus_range(src, cache, sus_path)
        if not sus_max > sus_min:
            raise RuntimeError(f"Degenerate global SUS range: {sus_min}, {sus_max}")
        if args.workers == 1:
            results = [
                process_ablation_case(
                    item, src, promap, fine_ref, image_semantics, cache, args,
                    sus_min, sus_max, rainfall_dir, gdb_path, None, experiments,
                )
                for item in tqdm(cases, desc="Full and ablations")
            ]
        else:
            config = {
                "args": vars(args),
                "cache_root": str(cache.root), "compatibility": cache.compatibility,
                "sus_path": str(sus_path), "sus_min": sus_min, "sus_max": sus_max,
                "rainfall_dir": str(rainfall_dir), "gdb_path": str(gdb_path) if gdb_path else None,
                "experiments": experiments,
            }
            results = run_parallel_cases(cases, image_semantics, config, args.workers)
    observed = range_accumulator()
    metric_rows, failures = [], []
    for result in results:
        metric_rows.extend(result["metric_rows"])
        failures.extend(result["failures"])
        for name, limits in result["observed"].items():
            for bound, operation in (("min", min), ("max", max)):
                value = limits[bound]
                if value is not None:
                    previous = observed[name][bound]
                    observed[name][bound] = value if previous is None else operation(previous, value)
    if observed["s_norm"]["min"] is None:
        raise RuntimeError("No cases produced valid globally normalized susceptibility values")
    needs_rain = any(e in {"E2_rain", "E3_sus_rain_softor", "E5_full_softor_fine", "E6_rain_fine"} for e in experiments)
    if needs_rain and observed["r_id_trigger"]["min"] is None:
        raise RuntimeError("No valid rainfall values were evaluated")
    for name in ("s_norm", "r_id_trigger", "h_soft_or", "weighted_s_norm", "weighted_r_id_trigger"):
        if observed[name]["min"] is not None:
            require_unit_interval(name, np.asarray([observed[name]["min"], observed[name]["max"]], dtype=float))
    for row in metric_rows:
        row.update(fusion_result_metadata(row["experiment_id"], args.w_s, args.w_r))
        row["image_aggregation"] = args.image_aggregation
    write_json(results_dir / "ablation_case_metrics.json", metric_rows)
    if failures:
        write_json(results_dir / "case_processing_failures.json", failures)
    metadata = {
        "split": selection.name, "split_case_ids": [str(case["tweets_id"]) for case in cases],
        "experiments": experiments, "workers": args.workers,
        "w_s": args.w_s, "w_r": args.w_r, "rainfall": rainfall_configuration,
        "image_aggregation": args.image_aggregation, "top_k_values": TOP_K_VALUES,
        "numerical_precision": NUMERICAL_PRECISION, "raster_mask_policy": RASTER_MASK_POLICY,
        "inputs": compatibility,
        "cache": {"directory": str(cache.root)},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(results_dir / "run_config.json", metadata)
    write_json(results_dir / "COMPLETED.json", {
        "case_count": len(cases), "experiments": experiments,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Full results: {results_dir}", flush=True)
