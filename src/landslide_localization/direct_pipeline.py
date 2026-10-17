"""Independent Direct-answer localization using the shared spatial preprocessor."""

from __future__ import annotations
import copy
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import rasterio
from shapely.geometry import Point, mapping
from landslide_localization.answer_resolution import (
    answer_key,
    choose_ranked_polygon,
    normalized_answer,
    result_metadata,
)
from landslide_localization.preprocessing import (
    get_gt,
    global_sus_range,
    id_trigger_rainfall_configuration,
    default_preprocessing_arguments,
    preprocess_case,
    resolved_input,
)
from landslide_localization import preprocessing
from landslide_localization.soft_config import (
    DEFAULT_GDB_PATH,
    DEFAULT_ID_TRIGGER_THRESHOLD_MODE,
    DEFAULT_IMAGE_SEMANTICS,
    DEFAULT_RAIN_SIGMOID_SCALE,
    DEFAULT_RAINFALL_DIR,
    DEFAULT_SUS_RASTER,
    TIER_DISPLAY_NAMES,
    fixed_manifest,
)
from landslide_localization.soft_utils import (
    CacheManager,
    build_candidate_region,
    candidate_geojson,
    evaluate_score,
    experiment_scores,
    fusion_result_metadata,
    missing_metric_row,
    stable_hash,
    valid_geometry,
    write_json,
)


FULL_EXPERIMENT = "E5_full_softor_fine"


DIRECT_MODEL = "direct_llm"


DIRECT_SPATIAL_POLICY = "answer_polygon_no_full_text_or_spatial_cache_v1"


DIRECT_SOURCE_FIELDS = (
    "tweets_id",
    "answer",
    "coordinates",
    "coordinates_confidence",
    "date",
    "tweetFulltext",
    "text",
)


RASTER_MASK_POLICY = "all_touched_source_mask_v1"


STAGE1_CASE_FIELDS = [
    "model",
    "tweets_id",
    "tier",
    "answer",
    "status",
    "evaluable",
    "in_region",
    "failure_reason",
    "cache_source",
    "matched_display_name",
    "matched_addresstype",
    "matched_osm_type",
    "matched_osm_id",
    "direct_candidate_area_tier",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def model_tier(case: Mapping[str, Any]) -> str:
    area = case.get("candidate_area") or {}
    value = case.get("candidate_area_tier") or area.get("tier")
    return TIER_DISPLAY_NAMES.get(str(value), "other")


def unique_records(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in records:
        case_id = str(row.get("tweets_id"))
        if not case_id or case_id == "None":
            raise ValueError(f"{label} contains a missing tweets_id")
        if case_id in indexed:
            raise ValueError(f"{label} contains duplicate tweets_id={case_id}")
        indexed[case_id] = row
    return indexed


def normalize_answer_polygon(geometry: Any, promap) -> Any:
    work = valid_geometry(promap.reproject_geom(geometry, "EPSG:4326", promap.WORK_CRS))
    if work is None:
        raise ValueError("invalid_projected_answer_geometry")
    candidate = valid_geometry(promap.build_analysis_polygon(work, geometry.geom_type))
    if candidate is None:
        raise ValueError("empty_answer_candidate_region")
    return candidate


def resolve_direct_answers(
    raw_records: Sequence[Mapping[str, Any]],
    client,
    promap,
    progress_every: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    unique_answers: dict[str, str] = {}
    for row in raw_records:
        answer = normalized_answer(row.get("answer"))
        if answer:
            unique_answers.setdefault(answer_key(answer), answer)

    resolved_by_answer: dict[str, dict[str, Any]] = {}
    for number, (key, answer) in enumerate(unique_answers.items(), start=1):
        try:
            results, cache_source = client.search(answer)
            selected = choose_ranked_polygon(results, answer)
            if selected is None:
                resolved_by_answer[key] = {
                    "failure_reason": "no_matching_polygon",
                    "cache_source": cache_source,
                }
            else:
                rank, result, geometry = selected
                work_geometry = normalize_answer_polygon(geometry, promap)
                resolved_by_answer[key] = {
                    "geometry_work": work_geometry,
                    "geometry_wgs": promap.reproject_geom(
                        work_geometry, promap.WORK_CRS, "EPSG:4326"
                    ),
                    "matched_result": result_metadata(result, rank, geometry),
                    "cache_source": cache_source,
                }
        except Exception as exc:
            resolved_by_answer[key] = {
                "failure_reason": f"query_or_geometry_error:{type(exc).__name__}:{exc}",
                "cache_source": "",
            }
        if number == 1 or number % progress_every == 0 or number == len(unique_answers):
            print(
                f"Direct answer resolution: {number}/{len(unique_answers)} unique answers",
                flush=True,
            )
    return resolved_by_answer, {
        "unique_nonempty_answers": len(unique_answers),
        "query_sources": dict(sorted(client.stats.items())),
    }


def direct_stage1_records(
    raw_records: Sequence[Mapping[str, Any]],
    full_by_id: Mapping[str, Mapping[str, Any]],
    resolved_by_answer: Mapping[str, Mapping[str, Any]],
    promap,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in raw_records:
        case_id = str(row["tweets_id"])
        answer = normalized_answer(row.get("answer"))
        base = {
            "model": DIRECT_MODEL,
            "tweets_id": case_id,
            "tier": model_tier(full_by_id[case_id]),
            "answer": row.get("answer"),
            "status": "failed",
            "evaluable": False,
            "in_region": False,
            "failure_reason": "",
            "cache_source": "",
            "matched_display_name": "",
            "matched_addresstype": "",
            "matched_osm_type": "",
            "matched_osm_id": "",
            "_geometry_work": None,
            "_geometry_wgs": None,
            "_matched_result": {},
            "_source_record": {
                key: copy.deepcopy(row[key])
                for key in DIRECT_SOURCE_FIELDS
                if key in row
            },
            "direct_candidate_area_tier": "unknown",
        }
        if not answer:
            base["failure_reason"] = "missing_answer"
            output.append(base)
            continue
        resolution = resolved_by_answer[answer_key(answer)]
        if resolution.get("failure_reason"):
            base["failure_reason"] = str(resolution["failure_reason"])
            base["cache_source"] = str(resolution.get("cache_source") or "")
            output.append(base)
            continue
        lon, lat = get_gt(dict(row))
        if lon is None or lat is None:
            base["failure_reason"] = "invalid_reference_coordinates"
            output.append(base)
            continue
        point_work = promap.reproject_geom(
            Point(lon, lat), "EPSG:4326", promap.WORK_CRS
        )
        geometry_work = resolution["geometry_work"]
        inside = bool(geometry_work.covers(point_work))
        matched = dict(resolution.get("matched_result") or {})
        base.update(
            {
                "status": "inside" if inside else "outside",
                "evaluable": True,
                "in_region": inside,
                "cache_source": resolution.get("cache_source") or "",
                "matched_display_name": matched.get("display_name") or "",
                "matched_addresstype": matched.get("addresstype") or "",
                "matched_osm_type": matched.get("osm_type") or "",
                "matched_osm_id": matched.get("osm_id") or "",
                "_geometry_work": geometry_work,
                "_geometry_wgs": resolution["geometry_wgs"],
                "_matched_result": matched,
                "direct_candidate_area_tier": direct_administrative_metadata(matched)[
                    "tier"
                ],
            }
        )
        output.append(base)
    return output


def resolution_signature(records: Sequence[Mapping[str, Any]]) -> str:
    values = []
    for row in records:
        geometry = row.get("_geometry_wgs")
        values.append(
            {
                "tweets_id": row["tweets_id"],
                "answer": row.get("answer"),
                "status": row["status"],
                "geometry_sha256": (
                    hashlib.sha256(geometry.wkb).hexdigest()
                    if geometry is not None
                    else None
                ),
                "failure_reason": row.get("failure_reason"),
            }
        )
    return stable_hash(values, 64)


def direct_administrative_metadata(matched: Mapping[str, Any]) -> dict[str, Any]:
    """Computational administrative hints come only from the Direct match."""
    address = matched.get("address") or {}
    address_type = str(matched.get("addresstype") or "").casefold()
    tier = {
        "country": "broad_area",
        "state": "state_area",
        "province": "state_area",
        "county": "county_area",
        "state_district": "county_area",
    }.get(address_type, "fine_area" if address_type else "unknown")
    state = address.get("state") or address.get("province") or ""
    if not state and tier == "state_area":
        state = matched.get("name") or ""
    return {
        "tier": tier,
        "admin1": state,
        "admin2": address.get("county") or address.get("state_district") or "",
        "country": address.get("country") or "",
        "address": copy.deepcopy(dict(address)),
    }


def direct_source_item(direct_stage1: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only original tweet metadata; never inherit the Full candidate item."""
    source = direct_stage1["_source_record"]
    item = {
        key: copy.deepcopy(source[key]) for key in DIRECT_SOURCE_FIELDS if key in source
    }
    item["tweets_id"] = str(direct_stage1["tweets_id"])
    item["answer"] = direct_stage1.get("answer")
    item["weight_evidence"] = []
    item["parsed_answer"] = {}
    item["candidate_area_tier"] = direct_stage1.get(
        "direct_candidate_area_tier", "unknown"
    )
    item["direct_spatial_policy"] = DIRECT_SPATIAL_POLICY
    return item


def direct_candidate_item(direct_stage1: Mapping[str, Any]) -> dict[str, Any]:
    item = direct_source_item(direct_stage1)
    matched = direct_stage1.get("_matched_result") or {}
    administrative = direct_administrative_metadata(matched)
    item["candidate_area_tier"] = administrative["tier"]
    item["candidate_area"] = {
        **administrative,
        "geojson": mapping(direct_stage1["_geometry_wgs"]),
        "geojson_available": True,
        "display_name": matched.get("display_name") or direct_stage1.get("answer"),
        "name": matched.get("name") or direct_stage1.get("answer"),
        "geometry_type": direct_stage1["_geometry_wgs"].geom_type,
        "osm_type": matched.get("osm_type"),
        "osm_id": matched.get("osm_id"),
        "source": "direct_llm_answer_nominatim",
    }
    item["direct_llm_answer"] = direct_stage1.get("answer")
    return item


def prepare_direct_candidate_cache(
    item: dict[str, Any],
    case_id: str,
    direct_cache: CacheManager,
    promap,
) -> None:
    if item.get("weight_evidence") or (item.get("parsed_answer") or {}).get(
        "local_features"
    ):
        raise ValueError(
            "Independent Direct cannot contain Full structured text evidence"
        )
    text_records, direct_text_path = direct_cache.load_json("seed_text_features", case_id)
    if text_records is not None and text_records != []:
        raise ValueError("Independent Direct cache contains incompatible text evidence")
    if text_records is None:
        direct_cache.save_json(direct_text_path, [])

    candidate_path = direct_cache.path("candidate_regions", case_id, ".geojson")
    if candidate_path.exists():
        return
    administrative, unions, _, _ = build_candidate_region(item, [], promap)
    document = candidate_geojson(
        case_id, administrative, unions, administrative, promap
    )
    for feature in document.get("features", []):
        properties = feature.get("properties") or {}
        if properties.get("role") == "final_candidate_region":
            properties["definition"] = (
                "Direct LLM answer polygon; no structured text evidence"
            )
            properties["direct_spatial_policy"] = DIRECT_SPATIAL_POLICY
    write_json(candidate_path, document)


def metric_metadata(w_s: float, w_r: float) -> dict[str, Any]:
    return fusion_result_metadata(FULL_EXPERIMENT, w_s, w_r)


def direct_stage2_rows(
    test_cases: Sequence[Mapping[str, Any]],
    direct_stage1_by_id: Mapping[str, Mapping[str, Any]],
    direct_cache: CacheManager,
    promap,
    fine_ref,
    source_raster,
    image_semantics: Mapping[str, Any],
    pipeline_args: SimpleNamespace,
    sus_min: float,
    sus_max: float,
    rainfall_dir: Path,
    gdb_path: Path | None,
    w_s: float,
    w_r: float,
    progress_every: int,
    report_progress: bool = True,
    checkpoint_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metadata = metric_metadata(w_s, w_r)
    checkpoints = (
        load_direct_checkpoints(test_cases, checkpoint_context)
        if checkpoint_context
        else {}
    )
    for number, full_item_source in enumerate(test_cases, start=1):
        if number - 1 in checkpoints:
            output.append(checkpoints[number - 1])
            continue
        case_id = str(full_item_source["tweets_id"])
        stage1 = direct_stage1_by_id[case_id]
        source_item = direct_source_item(stage1)
        if not stage1["evaluable"]:
            reason = f"direct_stage1_failure:{stage1['failure_reason']}"
            row = missing_metric_row(source_item, DIRECT_MODEL, reason)
            row["stage1_evaluable"] = False
        elif not stage1["in_region"]:
            row = missing_metric_row(
                source_item, DIRECT_MODEL, "ground_truth_outside_candidate_region"
            )
            row["stage1_evaluable"] = True
        else:
            item = direct_candidate_item(stage1)
            try:
                prepare_direct_candidate_cache(item, case_id, direct_cache, promap)
                context, _, error = preprocess_case(
                    item,
                    source_raster,
                    promap,
                    fine_ref,
                    dict(image_semantics),
                    direct_cache,
                    pipeline_args,
                    sus_min,
                    sus_max,
                    rainfall_dir,
                    gdb_path,
                    None,
                )
                if error or context is None:
                    reason = error or "unknown_preprocessing_failure"
                    row = missing_metric_row(item, DIRECT_MODEL, reason)
                    row["gt_inside_region"] = True
                else:
                    if context.get("gt_inside_region") is not True:
                        raise AssertionError(
                            "Direct Stage-I containment disagrees with Stage-II candidate context"
                        )
                    score = experiment_scores(
                        context["s_norm"],
                        context["rain"],
                        context["fine"],
                        [FULL_EXPERIMENT],
                        w_s=w_s,
                        w_r=w_r,
                    )[FULL_EXPERIMENT]
                    base = {
                        "tweets_id": case_id,
                        "candidate_area_tier": context.get("candidate_area_tier"),
                        "candidate_area_km2": context.get("candidate_area_km2"),
                        "gt_inside_region": True,
                    }
                    row, _ = evaluate_score(
                        FULL_EXPERIMENT,
                        score,
                        context["transform"],
                        context.get("gt_lon"),
                        context.get("gt_lat"),
                        base,
                        promap,
                    )
                    row["experiment_id"] = DIRECT_MODEL
                row["stage1_evaluable"] = True
            except Exception as exc:
                row = missing_metric_row(
                    item,
                    DIRECT_MODEL,
                    f"{type(exc).__name__}:{exc}",
                )
                row["gt_inside_region"] = True
                row["stage1_evaluable"] = True
        row.update(metadata)
        row["direct_spatial_policy"] = DIRECT_SPATIAL_POLICY
        row["direct_candidate_area_tier"] = row.get("candidate_area_tier")
        if checkpoint_context:
            save_direct_checkpoint(checkpoint_context, full_item_source, row)
        output.append(row)
        if report_progress and (
            number == 1 or number % progress_every == 0 or number == len(test_cases)
        ):
            print(f"Direct Stage II: {number}/{len(test_cases)} test cases", flush=True)
    return output


def _init_direct_worker(sus_path: str, local_cache_lock) -> None:
    global _DIRECT_WORKER_RASTER
    _DIRECT_WORKER_RASTER = rasterio.open(sus_path)
    # Local OSM parquet creation is shared across cases and is not atomic.
    preprocessing._LOCAL_CACHE_LOCK = local_cache_lock
    preprocessing.fine_area.local_reconstruction.LocalOSMStore.load_local_osm_for_state = (
        preprocessing._locked_local_osm_load
    )


def _direct_worker(task):
    index, item, stage1 = task
    c = _DIRECT_WORKER_CONTEXT
    case_id = str(item["tweets_id"])
    hits_before = c["direct_cache"].hits
    misses_before = c["direct_cache"].misses
    print(
        f"Direct case {case_id} worker={multiprocessing.current_process().pid} started",
        flush=True,
    )
    row = direct_stage2_rows(
        [item],
        {case_id: stage1},
        c["direct_cache"],
        c["promap"],
        c["fine_ref"],
        _DIRECT_WORKER_RASTER,
        c["image_semantics"],
        c["pipeline_args"],
        c["sus_min"],
        c["sus_max"],
        c["rainfall_dir"],
        c["gdb_path"],
        c["w_s"],
        c["w_r"],
        c["progress_every"],
        report_progress=False,
    )[0]
    save_direct_checkpoint(c, item, row)
    print(
        f"Direct case {case_id} worker={multiprocessing.current_process().pid} completed",
        flush=True,
    )
    return (
        index,
        row,
        c["direct_cache"].hits - hits_before,
        c["direct_cache"].misses - misses_before,
    )


def _checkpoint_path(context, case_id):
    if not str(case_id).isdigit():
        raise ValueError("Direct checkpoint case ID must be numeric")
    return (
        Path(context["checkpoint_root"])
        / f"{case_id}_{stable_hash(context['checkpoint_identity'])}.json"
    )


def save_direct_checkpoint(context, item, row):
    """Commit each returned metric independently, including failed/outside cases."""
    if not context.get("checkpoint_root"):
        return
    case_id = str(item["tweets_id"])
    if str(row.get("tweets_id")) != case_id or row.get("experiment_id") != DIRECT_MODEL:
        raise ValueError("Direct checkpoint metric identity mismatch")
    atomic_json(
        _checkpoint_path(context, case_id),
        {
            "identity": context["checkpoint_identity"],
            "case_id": case_id,
            "row": row,
        },
    )


def load_direct_checkpoints(cases, context):
    if not context or not context.get("checkpoint_root"):
        return {}
    result = {}
    for index, item in enumerate(cases):
        case_id = str(item["tweets_id"])
        path = _checkpoint_path(context, case_id)
        if not path.is_file():
            continue
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid Direct case checkpoint: {path}")
        row = data.get("row")
        if (
            data.get("identity") != context["checkpoint_identity"]
            or data.get("case_id") != case_id
            or not isinstance(row, dict)
            or str(row.get("tweets_id")) != case_id
            or row.get("experiment_id") != DIRECT_MODEL
        ):
            raise ValueError(f"Invalid Direct case checkpoint: {path}")
        result[index] = row
    return result


def direct_stage2_parallel_rows(
    cases: Sequence[Mapping[str, Any]],
    stage1_by_id: Mapping[str, Mapping[str, Any]],
    context: dict[str, Any],
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    global _DIRECT_WORKER_CONTEXT
    _DIRECT_WORKER_CONTEXT = context
    results: list[dict[str, Any] | None] = [None] * len(cases)
    if not cases:
        return []
    checkpoints = load_direct_checkpoints(cases, context)
    for index, row in checkpoints.items():
        results[index] = row
    if checkpoints:
        print(
            f"Resuming {len(checkpoints)}/{len(cases)} saved Direct case metrics",
            flush=True,
        )
    if len(checkpoints) == len(cases):
        _DIRECT_WORKER_CONTEXT = None
        return results
    fork_context = multiprocessing.get_context("fork")
    local_cache_lock = fork_context.Lock()
    with ProcessPoolExecutor(
        max_workers=min(workers, len(cases)),
        mp_context=fork_context,
        initializer=_init_direct_worker,
        initargs=(str(context["sus_path"]), local_cache_lock),
    ) as executor:
        futures = [
            executor.submit(
                _direct_worker, (index, item, stage1_by_id[str(item["tweets_id"])])
            )
            for index, item in enumerate(cases)
            if index not in checkpoints
        ]
        for completed, future in enumerate(
            as_completed(futures), start=1 + len(checkpoints)
        ):
            index, row, hits, misses = future.result()
            results[index] = row
            save_direct_checkpoint(context, cases[index], row)
            context["direct_cache"].hits += hits
            context["direct_cache"].misses += misses
            if (
                completed == 1
                or completed % context["progress_every"] == 0
                or completed == len(cases)
            ):
                print(
                    f"Direct Stage II {label}: {completed}/{len(cases)} cases",
                    flush=True,
                )
    _DIRECT_WORKER_CONTEXT = None
    return results


def public_stage1_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in STAGE1_CASE_FIELDS}


_DIRECT_WORKER_CONTEXT = None


_DIRECT_WORKER_RASTER = None


def prepare_pipeline(
    full_inputs,
    direct_cache_root,
    promap,
    image_aggregation,
    w_s,
    w_r,
):
    sus_path = Path(
        str((full_inputs.get("sus_raster") or {}).get("path") or DEFAULT_SUS_RASTER)
    ).resolve()
    rainfall_dir = Path(
        str(full_inputs.get("rainfall_dir") or DEFAULT_RAINFALL_DIR)
    ).resolve()
    image_path = Path(
        str(
            (full_inputs.get("image_semantics") or {}).get("path")
            or DEFAULT_IMAGE_SEMANTICS
        )
    ).resolve()
    gdb_value = (
        (full_inputs.get("gdb_path") or {}).get("path")
        if isinstance(full_inputs.get("gdb_path"), dict)
        else full_inputs.get("gdb_path")
    )
    gdb_candidate = Path(str(gdb_value or DEFAULT_GDB_PATH)).resolve()
    gdb_path = gdb_candidate if gdb_candidate.exists() else None
    for path, label in (
        (sus_path, "susceptibility raster"),
        (rainfall_dir, "rainfall directory"),
        (image_path, "image semantics"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")

    threshold_mode = str(
        full_inputs.get("id_trigger_threshold_mode")
        or DEFAULT_ID_TRIGGER_THRESHOLD_MODE
    )
    rain_sigmoid_scale = float(
        full_inputs.get("rain_sigmoid_scale", DEFAULT_RAIN_SIGMOID_SCALE)
    )
    rainfall_configuration = id_trigger_rainfall_configuration(
        threshold_mode, rain_sigmoid_scale
    )
    pipeline_args = default_preprocessing_arguments(workflow="direct")
    pipeline_args.id_trigger_threshold_mode = threshold_mode
    pipeline_args.image_aggregation = image_aggregation
    pipeline_args.rain_sigmoid_scale = rain_sigmoid_scale
    pipeline_args.hydro_timeout_seconds = int(
        full_inputs.get("hydro_timeout_seconds", 180)
    )
    pipeline_args.hydro_cache_only = bool(full_inputs.get("hydro_cache_only", False))
    pipeline_args.cache_dir = str(direct_cache_root)
    fine_settings = full_inputs.get("fine_reconstruction") or {}
    frozen_road_pbf = fine_settings.get("road_pbf") or {}
    pipeline_args.disable_national_road_pbf = bool(frozen_road_pbf.get("missing"))
    if frozen_road_pbf.get("path"):
        pipeline_args.road_pbf = str(frozen_road_pbf["path"])
    for name, signature in (fine_settings.get("pbf_directories") or {}).items():
        if name in {"osm_state_dir", "road_state_dir", "hydro_state_dir"}:
            path = Path(signature["path"])
            setattr(pipeline_args, name, str(path))
    pipeline_args.allow_cross_type_fallback_with_text_fine = bool(
        fine_settings.get(
            "allow_cross_type_fallback_with_text_fine",
            pipeline_args.allow_cross_type_fallback_with_text_fine,
        )
    )
    pipeline_args.enable_fine_fallback = bool(
        fine_settings.get("enable_fine_fallback", pipeline_args.enable_fine_fallback)
    )
    pipeline_args.road_pbf_timeout_seconds = int(
        fine_settings.get(
            "road_pbf_timeout_seconds", pipeline_args.road_pbf_timeout_seconds
        )
    )
    if pipeline_args.road_pbf_timeout_seconds < 0:
        raise ValueError("PBF timeout must be nonnegative")
    for name, value in (fine_settings.get("local_parameters") or {}).items():
        if hasattr(pipeline_args, name):
            setattr(pipeline_args, name, value)
    pipeline_args.local_text_cache_dir = str(
        direct_cache_root / "local_text_reconstruction"
    )
    pipeline_args.w_s = w_s
    pipeline_args.w_r = w_r
    image_semantics = promap.load_image_semantics(str(image_path))

    return dict(
        sus_path=sus_path,
        rainfall_dir=rainfall_dir,
        gdb_path=gdb_path,
        pipeline_args=pipeline_args,
        image_semantics=image_semantics,
        rainfall_configuration=rainfall_configuration,
    )


def isolated_path(value: str | Path) -> Path:
    """Require the package or the explicitly configured external output tree."""
    from landslide_localization.output_paths import ensure_output_path

    path = resolved_input(str(value), must_exist=False)
    return ensure_output_path(path)


def atomic_json(path: Path, value: Any) -> None:
    path = isolated_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def freeze_metric_population(rows, cases, raw_by_id, model):
    """Join labels/GT confidence by exact ID, preserving each model's own validity."""
    indexed = unique_records(rows, f"{model} metrics")
    ids = [str(case["tweets_id"]) for case in cases]
    if set(indexed) != set(ids):
        raise ValueError(
            f"{model} metrics do not cover exactly the active frozen test IDs"
        )
    result = []
    for case in cases:
        case_id = str(case["tweets_id"])
        row = dict(indexed[case_id])
        row["experiment_id"] = model
        row["candidate_area_tier"] = case.get("candidate_area_tier") or (
            case.get("candidate_area") or {}
        ).get("tier")
        row["coordinates_confidence"] = raw_by_id[case_id].get("coordinates_confidence")
        if (
            row.get("gt_in_valid_raster") is True
            and row.get("gt_inside_region") is not True
        ):
            raise ValueError(
                f"{model}/{case_id}: valid raster point outside candidate polygon"
            )
        result.append(row)
    return result


def reusable_direct_result(
    direct_dir: Path, identity: Mapping[str, Any], cases, raw_by_id
):
    marker = direct_dir / "completion.json"
    if not marker.is_file():
        return None
    saved = read_json(marker)
    if saved.get("identity") != identity:
        raise ValueError(
            "Saved Direct results have different inputs/settings; choose a new --output-dir"
        )
    path = direct_dir / "stage2_test.json"
    if not path.is_file():
        raise ValueError("Saved Direct results are missing")
    rows = freeze_metric_population(read_json(path), cases, raw_by_id, DIRECT_MODEL)
    print(f"Reusing {len(rows)} completed Direct metrics: {path}", flush=True)
    return rows
