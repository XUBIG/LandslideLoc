"""Shared spatial preprocessing used by the Full and Direct workflows."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
import numpy as np
from affine import Affine
from rasterio.mask import mask as raster_mask
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union
from landslide_localization.runtime_paths import REPO_ROOT as ROOT
from landslide_localization.soft_config import (
    NUMERICAL_PRECISION,
    RASTER_MASK_POLICY,
    DEFAULT_RAIN_SIGMOID_SCALE,
    ID_TRIGGER_FORMULA_COEFFICIENT,
    ID_TRIGGER_FORMULA_EXPONENT,
    ID_TRIGGER_THRESHOLDS,
    LANDSLIDE_GATE_THRESHOLD,
    MAX_CANDIDATE_AREA_KM2,
    OTHER_BUFFER_M,
    ROAD_BUFFER_M,
    TEXT_MULTIPLIER,
    WATER_BUFFER_M,
)
from landslide_localization.soft_utils import (
    CacheManager,
    build_candidate_region,
    cacheable_text_records,
    candidate_geojson,
    combine_multiplier_arrays,
    extract_hydro_buffers,
    parse_candidate_geojson,
    path_signature,
    rasterize_multiplier,
    require_unit_interval,
    resolve_text_feature_records,
    semantic_effective_multipliers,
    stable_hash,
    write_json,
)
from landslide_localization import fine_area
from landslide_localization.fine_area import (
    _prepare_local_text_records,
    _corrected_text_unions,
    _case_pbf_candidates as _unfrozen_case_pbf_candidates,
    _fallback_reason,
    _pbf_directory_signature,
)


def _case_pbf_candidates(item, args, component):
    candidates = _unfrozen_case_pbf_candidates(item, args, component)
    if getattr(args, "disable_national_road_pbf", False):
        candidates = [
            entry for entry in candidates if entry["label"] != "national_full"
        ]
    return candidates


ALLOW_CROSS_TYPE_FALLBACK_WITH_TEXT_FINE = False


ENABLE_FINE_FALLBACK = True


def crop_candidate_susceptibility(src, geometry):
    """Include intersecting candidate pixels while preserving the source mask."""
    crop, transform = raster_mask(
        src,
        [mapping(geometry)],
        crop=True,
        filled=False,
        all_touched=True,
    )
    values = crop[0].astype(float).filled(np.nan)
    valid = np.isfinite(values)
    if src.nodata is not None:
        valid &= values != src.nodata
    return values, valid, transform


def positive_finite_cli_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number; got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"Value must be finite and greater than zero; got {value!r}"
        )
    return parsed


def add_fine_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Spatial preprocessing options used by Full localization."""
    parser.add_argument(
        "--image-aggregation",
        choices=("noisy_or_weighted", "noisy_or", "weighted_avg"),
        default="noisy_or",
        help="Across-image road/water score aggregation; default: noisy_or.",
    )
    parser.add_argument(
        "--allow-cross-type-fallback-with-text-fine",
        action=argparse.BooleanOptionalAction,
        default=ALLOW_CROSS_TYPE_FALLBACK_WITH_TEXT_FINE,
    )
    parser.add_argument(
        "--enable-fine-fallback",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_FINE_FALLBACK,
    )
    parser.add_argument(
        "--road-pbf", default=str(ROOT / "data/tif_data/road/us-latest.osm.pbf")
    )
    parser.add_argument("--osm-state-dir", default=str(ROOT / "data/tif_data/osm_data"))
    parser.add_argument(
        "--road-state-dir", default=str(ROOT / "data/tif_data/osm_data/road_only")
    )
    parser.add_argument(
        "--hydro-state-dir", default=str(ROOT / "data/tif_data/osm_data/hydro_only")
    )
    parser.add_argument(
        "--road-pbf-timeout-seconds",
        type=int,
        default=3600,
        help="Local road/water PBF worker limit in seconds; 0 disables this limit.",
    )
    parser.add_argument("--local-search-buffer-m", type=float, default=2000.0)
    parser.add_argument("--local-endpoint-tolerance-m", type=float, default=1.0)
    parser.add_argument("--local-seed-tolerance-m", type=float, default=100.0)
    parser.add_argument("--local-memory-states", type=int, default=1)
    parser.add_argument(
        "--local-raw-policy",
        choices=("diagnostic", "all", "none"),
        default="diagnostic",
    )
    parser.add_argument(
        "--local-text-cache-dir",
        default=str(ROOT / "outputs/local_fine_reconstruction/cache"),
    )


def validate_fine_preprocessing_args(parser, args) -> None:
    if (
        not math.isfinite(args.local_search_buffer_m)
        or args.local_search_buffer_m < 0
        or not math.isfinite(args.local_endpoint_tolerance_m)
        or args.local_endpoint_tolerance_m <= 0
        or not math.isfinite(args.local_seed_tolerance_m)
        or args.local_seed_tolerance_m < 0
        or args.local_memory_states < 1
        or args.road_pbf_timeout_seconds < 0
    ):
        parser.error(
            "Invalid local reconstruction distance, memory limit, or PBF timeout"
        )


def resolved_input(path: str, must_exist: bool = True) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    value = value.resolve()
    if must_exist and not value.exists():
        raise FileNotFoundError(value)
    return value


def global_sus_range(
    src, cache: CacheManager, sus_path: Path
) -> tuple[float, float, int]:
    key = "sus_minmax_" + stable_hash(path_signature(sus_path), 16)
    cached, path = cache.load_json("global", key)
    if cached is not None:
        return (
            float(cached["min"]),
            float(cached["max"]),
            int(cached["valid_pixel_count"]),
        )
    from landslide_localization.rainfall import global_sus_min_max

    minimum, maximum, count = global_sus_min_max(src)
    cache.save_json(
        path,
        {
            "min": minimum,
            "max": maximum,
            "valid_pixel_count": count,
            "source": str(sus_path),
        },
    )
    return minimum, maximum, count


def id_trigger_thresholds(mode: str) -> dict[int, float]:
    """Resolve the four I-D thresholds while preserving the historical set."""
    if mode == "formula":
        return {
            days: ID_TRIGGER_FORMULA_COEFFICIENT * days**ID_TRIGGER_FORMULA_EXPONENT
            for days in ID_TRIGGER_THRESHOLDS
        }
    if mode == "fixed":
        return {
            days: float(threshold) for days, threshold in ID_TRIGGER_THRESHOLDS.items()
        }
    raise ValueError(f"Unknown ID-trigger threshold mode: {mode!r}")


def id_trigger_rainfall_configuration(
    threshold_mode: str,
    sigmoid_scale: float,
) -> dict[str, Any]:
    """Return the shared, serializable ID-trigger rainfall configuration."""
    if not math.isfinite(sigmoid_scale) or sigmoid_scale <= 0.0:
        raise ValueError(
            f"Rain sigmoid scale must be finite and greater than zero; got {sigmoid_scale!r}"
        )
    thresholds = id_trigger_thresholds(threshold_mode)
    return {
        "raw_exceedance_formula": (
            "E_i(x)=max_D I_i,D(x)/(13.04*D^-0.44)"
            if threshold_mode == "formula"
            else "E_i(x)=max_D I_i,D(x)/ID_TRIGGER_THRESHOLDS[D]"
        ),
        "threshold_mode": threshold_mode,
        "thresholds_mm_per_day": thresholds,
        "sigmoid_formula": "R_id_trigger=sigmoid((E_i-1)/a)",
        "sigmoid_scale": float(sigmoid_scale),
    }


def rain_args(rainfall_dir: Path, args: argparse.Namespace) -> SimpleNamespace:
    # Callers predating this interface retain their historical fixed-threshold behavior.
    mode = getattr(args, "id_trigger_threshold_mode", "fixed")
    configuration = id_trigger_rainfall_configuration(
        mode,
        getattr(args, "rain_sigmoid_scale", DEFAULT_RAIN_SIGMOID_SCALE),
    )
    thresholds = configuration["thresholds_mm_per_day"]
    return SimpleNamespace(
        rainfall_dir=rainfall_dir,
        id_threshold_1=thresholds[1],
        id_threshold_3=thresholds[3],
        id_threshold_5=thresholds[5],
        id_threshold_7=thresholds[7],
        id_sigmoid_scale=configuration["sigmoid_scale"],
        # Unused by R_id_trigger but retained for the authoritative function signature.
        rain_scale_mm=100.0,
        rain_cap=3.0,
        tau_mm=60.0,
        api_decay=0.8,
    )


def get_gt(item: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    coords = item.get("coordinates") or [None, None]
    try:
        lon, lat = float(coords[0]), float(coords[1])
    except (TypeError, ValueError, IndexError):
        return None, None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None, None
    return lon, lat


def infer_time(item: dict[str, Any], promap) -> tuple[Optional[str], Optional[str]]:
    tweet_date = item.get("date")
    if not tweet_date:
        return None, None
    tweet_text = item.get("tweetFulltext") or item.get("text") or ""
    inferred, _, _ = promap.infer_event_date_from_text(tweet_text, tweet_date)
    if inferred:
        return inferred, "text"
    return tweet_date, "tweet_date"


def validate_cached_gt(item, context):
    expected_gt = get_gt(item)
    cached_gt = (context.get("gt_lon"), context.get("gt_lat"))
    if cached_gt != expected_gt:
        raise ValueError(
            f"Cached GT differs from authoritative metadata for {item.get('tweets_id')}: "
            f"{cached_gt} != {expected_gt}"
        )


def context_from_cache(data: dict[str, Any], arrays: dict[str, np.ndarray]):
    if data.get("raster_mask_policy") != RASTER_MASK_POLICY:
        raise ValueError("Incompatible candidate raster mask policy in cached context")
    for key in ("s_norm", "rain", "fine"):
        if arrays[key].dtype != np.dtype(float):
            raise ValueError("Cached array has an incompatible numeric type: " + key)
    return {
        **data,
        "transform": Affine(*[float(value) for value in data["transform"]]),
        "s_norm": arrays["s_norm"].astype(float),
        "rain": arrays["rain"].astype(float),
        "fine": arrays["fine"].astype(float),
        "valid_mask": arrays["valid_mask"].astype(bool),
    }


def fine_source_selection(records, text_unions, args):
    """Select fallback only for types without usable, buffered text evidence."""
    allow_cross = getattr(
        args,
        "allow_cross_type_fallback_with_text_fine",
        ALLOW_CROSS_TYPE_FALLBACK_WITH_TEXT_FINE,
    )
    enabled = getattr(args, "enable_fine_fallback", ENABLE_FINE_FALLBACK)
    result = {
        "allow_cross_type_fallback_with_text_fine": allow_cross,
        "enable_fine_fallback": enabled,
    }
    for category in ("road", "water", "other"):
        rows = [row for row in records if row.get("fine_category") == category]
        geom = text_unions.get(category)
        count = sum(bool(row.get("final_text_usable")) for row in rows)
        result[f"text_{category}_total"] = len(rows)
        result[f"n_text_{category}"] = (
            count if geom is not None and not geom.is_empty else 0
        )
        result[f"{category}_source"] = (
            "text" if result[f"n_text_{category}"] else "none"
        )
    any_text = any(result[f"n_text_{c}"] for c in ("road", "water", "other"))
    for category in ("road", "water"):
        missing = not result[f"n_text_{category}"]
        suppressed = bool(missing and any_text and not allow_cross)
        requested = bool(enabled and missing and not suppressed)
        result[f"{category}_cross_type_fallback_suppressed"] = suppressed
        result[f"{category}_fallback_requested"] = requested
        result[f"{category}_fallback_reason"] = (
            _fallback_reason(category, records)
            if requested
            else (
                "GLOBAL_FALLBACK_DISABLED"
                if missing and not enabled
                else "CROSS_TYPE_FALLBACK_SUPPRESSED" if suppressed else ""
            )
        )
    return result


def log_fine_sources(case_id, diagnostic, cached=False):
    text = ", ".join(
        f"{c}={diagnostic.get(f'n_text_{c}', 0)}/{diagnostic.get(f'text_{c}_total', 0)}"
        for c in ("road", "water", "other")
    )
    fallback = ", ".join(
        f"{c}:requested={diagnostic.get(f'{c}_fallback_requested', False)},"
        f"success={diagnostic.get(f'{c}_fallback_success', False)},"
        f"cross_suppressed={diagnostic.get(f'{c}_cross_type_fallback_suppressed', False)}"
        for c in ("road", "water")
    )
    sources = ", ".join(
        f"{c}_source={diagnostic.get(f'{c}_source', 'none')}"
        for c in ("road", "water", "other")
    )
    print(
        f"[{case_id}] fine{' (cache)' if cached else ''}: text resolved/available {text}; "
        f"fallback {fallback}; {sources}",
        flush=True,
    )


def preprocess_case(
    item: dict[str, Any],
    src,
    promap,
    fine_ref,
    image_semantics: dict[str, Any],
    cache: CacheManager,
    args: argparse.Namespace,
    sus_min: float,
    sus_max: float,
    rainfall_dir: Path,
    gdb_path: Optional[Path],
    legacy_cache: Optional[Path],
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]], Optional[str]]:
    from landslide_localization.rainfall import build_method_score

    tweets_id = str(item.get("tweets_id"))
    case_key = tweets_id
    image_aggregation = getattr(args, "image_aggregation", "noisy_or_weighted")
    text_records, text_cache_path = cache.load_json("text_features", case_key)
    context_cached, context_path = cache.load_json("contexts", case_key)
    arrays_cached, arrays_path = cache.load_npz("arrays", case_key)
    if (
        context_cached is not None
        and context_cached.get("image_aggregation", "noisy_or_weighted")
        != image_aggregation
    ):
        raise ValueError(f"Image aggregation cache mismatch for {tweets_id}")
    if (
        text_records is not None
        and context_cached is not None
        and arrays_cached is not None
    ):
        if getattr(args, "workflow", "full") == "full":
            validate_cached_gt(item, context_cached)
        context_cached["diagnostic"]["image_aggregation"] = image_aggregation
        log_fine_sources(tweets_id, context_cached["diagnostic"], cached=True)
        return context_from_cache(context_cached, arrays_cached), text_records, None

    candidate_path = cache.path("candidate_regions", case_key, ".geojson")
    cached_candidate_wgs = None
    try:
        if text_records is not None and cache.existing(candidate_path):
            candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
            administrative, text_unions, text_union, candidate = (
                parse_candidate_geojson(candidate_data, promap)
            )
            # Use the original cached WGS84 coordinates for geometry cache keys;
            # a WGS84 -> WORK_CRS -> WGS84 round trip changes WKB bits.
            cached_candidate_wgs = next(
                (
                    shape(feature["geometry"])
                    for feature in candidate_data.get("features", [])
                    if (feature.get("properties") or {}).get("role")
                    == "final_candidate_region"
                ),
                None,
            )
        else:
            initial_records, seed_cache_path = cache.load_json(
                "seed_text_features", case_key
            )
            if initial_records is None:
                initial_records = resolve_text_feature_records(item, promap, fine_ref)
                cache.save_json(
                    seed_cache_path, cacheable_text_records(initial_records)
                )
            # Resolve the candidate area, then reconstruct and clip text features.
            administrative, _, _, candidate = build_candidate_region(
                item, initial_records, promap
            )
            text_records, recovery_details = _prepare_local_text_records(
                item,
                initial_records,
                administrative,
                candidate,
                promap,
                args,
            )
            for record, detail in zip(text_records, recovery_details):
                record["local_reconstruction_diagnostic"] = detail
            text_unions = _corrected_text_unions(text_records, candidate, promap)
            parts = [geom for geom in text_unions.values() if geom is not None]
            text_union = unary_union(parts) if parts else None
            write_json(
                candidate_path,
                candidate_geojson(
                    tweets_id, administrative, text_unions, candidate, promap
                ),
            )
            cache.save_json(text_cache_path, cacheable_text_records(text_records))
    except Exception as exc:
        return (
            None,
            text_records or [],
            f"candidate_region_failed:{type(exc).__name__}:{exc}",
        )

    administrative_area = float(administrative.area / 1e6)
    text_area = float(text_union.area / 1e6) if text_union is not None else 0.0
    candidate_area = float(candidate.area / 1e6)
    if administrative_area > MAX_CANDIDATE_AREA_KM2:
        return (
            None,
            text_records,
            f"administrative_area_too_large:{administrative_area:.6f}",
        )
    if candidate_area > MAX_CANDIDATE_AREA_KM2:
        return None, text_records, f"candidate_area_too_large:{candidate_area:.6f}"

    selection = fine_source_selection(text_records, text_unions, args)
    text_counts = {
        category: selection[f"n_text_{category}"]
        for category in ("road", "water", "other")
    }
    use_road_fallback = selection["road_fallback_requested"]
    use_water_fallback = selection["water_fallback_requested"]

    semantic_cached, semantic_path = cache.load_json("semantics", case_key)
    if semantic_cached is None:
        per_images = image_semantics.get(tweets_id, [])
        keep, p_landslide, p_road, p_water, semantic_debug = (
            promap.aggregate_probs_per_tweet(
                per_images,
                landslide_gate_thr=LANDSLIDE_GATE_THRESHOLD,
                mode=image_aggregation,
            )
        )
        road_multiplier, water_multiplier, multiplier_debug = (
            semantic_effective_multipliers(
                p_road,
                p_water,
                use_road_fallback,
                use_water_fallback,
            )
        )
        semantic_cached = {
            "image_aggregation": image_aggregation,
            "keep_tweet": bool(keep),
            "image_landslide_score": float(p_landslide),
            "image_road_score": float(p_road),
            "image_water_score": float(p_water),
            "road_image_multiplier": road_multiplier,
            "water_image_multiplier": water_multiplier,
            "semantic_debug": semantic_debug,
            "multiplier_debug": multiplier_debug,
            "use_road_fallback": use_road_fallback,
            "use_water_fallback": use_water_fallback,
        }
        cache.save_json(semantic_path, semantic_cached)
    road_multiplier = semantic_cached.get("road_image_multiplier")
    water_multiplier = semantic_cached.get("water_image_multiplier")

    hydro_aoi_wgs = (
        cached_candidate_wgs
        if cached_candidate_wgs is not None
        else promap.reproject_geom(candidate, promap.WORK_CRS, "EPSG:4326")
    )

    road_geom, water_geom, hydro_debug = extract_hydro_buffers(
        cache,
        administrative,
        hydro_aoi_wgs,
        use_road_fallback,
        use_water_fallback,
        promap,
        gdb_path,
        args.hydro_timeout_seconds,
        legacy_cache,
        args.hydro_cache_only,
        (
            None
            if getattr(args, "disable_national_road_pbf", False)
            else Path(args.road_pbf)
        ),
        args.road_pbf_timeout_seconds,
        _case_pbf_candidates(item, args, "road"),
        _case_pbf_candidates(item, args, "water"),
    )
    # One source per category; an unrequested/empty/outside buffer cannot enter F.
    fallback_geometries = {}
    for category, geom, multiplier in (
        ("road", road_geom, road_multiplier),
        ("water", water_geom, water_multiplier),
    ):
        if not selection[f"{category}_fallback_requested"] or multiplier is None:
            geom = None
        if geom is not None:
            geom = geom.intersection(candidate)
            if geom.is_empty:
                geom = None
        fallback_geometries[category] = geom
        selection[f"{category}_fallback_success"] = geom is not None
        if selection[f"{category}_fallback_requested"] and geom is None:
            reason = (
                hydro_debug.get(f"{category}_reason")
                or "FALLBACK_ZERO_RESULT_OR_OUTSIDE_CANDIDATE"
            )
            selection[f"{category}_fallback_reason"] += f"; {reason}"
        selection[f"{category}_source"] = (
            "text"
            if text_counts[category]
            else "fallback" if geom is not None else "none"
        )
    road_geom, water_geom = fallback_geometries["road"], fallback_geometries["water"]
    road_source, water_source, other_source = (
        selection[f"{c}_source"] for c in ("road", "water", "other")
    )
    log_fine_sources(tweets_id, selection)

    candidate_in_raster_crs = promap.reproject_geom(
        candidate, promap.WORK_CRS, src.crs
    )
    try:
        sus_raw, valid_mask, transform_ = crop_candidate_susceptibility(
            src, candidate_in_raster_crs
        )
    except Exception as exc:
        return (
            None,
            text_records,
            f"susceptibility_crop_failed:{type(exc).__name__}:{exc}",
        )
    if not np.any(valid_mask):
        return None, text_records, "no_valid_susceptibility_pixels"
    s_norm = np.full(sus_raw.shape, np.nan, dtype=float)
    s_norm[valid_mask] = (
        (sus_raw[valid_mask] - sus_min) / (sus_max - sus_min)
    ).astype(float)
    del sus_raw
    require_unit_interval("S_norm", s_norm, valid_mask)

    multiplier_layers: list[np.ndarray] = []
    active_geometries = []
    for category in ("road", "water", "other"):
        geom = text_unions.get(category)
        if geom is not None:
            layer = rasterize_multiplier(
                geom, TEXT_MULTIPLIER, s_norm.shape, transform_, src.crs, promap
            )
            # Every text-buffer cell uses the unchanged configured TEXT_MULTIPLIER.
            burned = layer > 1.0
            if np.any(burned) and not np.allclose(layer[burned], TEXT_MULTIPLIER):
                raise AssertionError(f"Unexpected text {category} multiplier")
            multiplier_layers.append(layer)
            active_geometries.append(geom)
    if road_geom is not None and road_multiplier is not None:
        multiplier_layers.append(
            rasterize_multiplier(
                road_geom,
                float(road_multiplier),
                s_norm.shape,
                transform_,
                src.crs,
                promap,
            )
        )
        if float(road_multiplier) > 1.0:
            active_geometries.append(road_geom)
    if water_geom is not None and water_multiplier is not None:
        multiplier_layers.append(
            rasterize_multiplier(
                water_geom,
                float(water_multiplier),
                s_norm.shape,
                transform_,
                src.crs,
                promap,
            )
        )
        if float(water_multiplier) > 1.0:
            active_geometries.append(water_geom)
    fine = combine_multiplier_arrays(multiplier_layers, s_norm.shape)
    fine[~valid_mask] = 1.0

    inferred_time, time_source = infer_time(item, promap)
    rain, rain_raw, rain_debug = build_method_score(
        promap,
        "id_trigger",
        inferred_time,
        rainfall_dir,
        s_norm.shape,
        transform_,
        src.crs,
        rain_args(rainfall_dir, args),
    )
    if rain is None:
        rain = np.full(s_norm.shape, np.nan, dtype=float)
    else:
        rain = rain.astype(float)
        rain[~valid_mask] = np.nan
        rain[~np.isfinite(rain)] = np.nan
        require_unit_interval("R_id_trigger", rain)

    gt_lon, gt_lat = get_gt(item)
    gt_inside = False
    if gt_lon is not None and gt_lat is not None:
        candidate_wgs = promap.reproject_geom(candidate, promap.WORK_CRS, "EPSG:4326")
        gt = Point(gt_lon, gt_lat)
        gt_inside = bool(candidate_wgs.contains(gt) or candidate_wgs.touches(gt))

    active_union = (
        unary_union(active_geometries).intersection(candidate)
        if active_geometries
        else None
    )
    fine_area = (
        float(active_union.area / 1e6)
        if active_union is not None and not active_union.is_empty
        else 0.0
    )
    fine_valid_values = fine[valid_mask]
    area = item.get("candidate_area") or {}
    candidate_name = str(
        area.get("display_name") or area.get("name") or item.get("answer") or ""
    )
    tier = item.get("candidate_area_tier") or area.get("tier")
    diagnostic = {
        **selection,
        "image_aggregation": image_aggregation,
        "tweets_id": tweets_id,
        "candidate_name": candidate_name,
        "candidate_area_tier": tier,
        "n_text_road": text_counts["road"],
        "n_text_water": text_counts["water"],
        "n_text_other": text_counts["other"],
        "road_source": road_source,
        "water_source": water_source,
        "other_source": other_source,
        "road_buffer_m": ROAD_BUFFER_M,
        "water_buffer_m": WATER_BUFFER_M,
        "other_buffer_m": OTHER_BUFFER_M,
        "text_multiplier": TEXT_MULTIPLIER,
        "image_landslide_score": semantic_cached.get("image_landslide_score"),
        "image_road_score": semantic_cached.get("image_road_score"),
        "image_water_score": semantic_cached.get("image_water_score"),
        "road_image_multiplier": road_multiplier,
        "water_image_multiplier": water_multiplier,
        "administrative_area_km2": administrative_area,
        "text_buffer_union_area_km2": text_area,
        "final_candidate_area_km2": candidate_area,
        "fine_area_gt1_km2": fine_area,
        "fraction_candidate_with_F_gt1": (
            fine_area / candidate_area if candidate_area else 0.0
        ),
        "F_min": float(np.min(fine_valid_values)),
        "F_max": float(np.max(fine_valid_values)),
        "F_mean": float(np.mean(fine_valid_values)),
        "candidate_region_cache": str(candidate_path),
        "array_cache": str(arrays_path),
        "failure_reason": "",
    }
    context_data = {
        "numerical_precision": NUMERICAL_PRECISION,
        "raster_mask_policy": RASTER_MASK_POLICY,
        "image_aggregation": image_aggregation,
        "tweets_id": tweets_id,
        "candidate_name": candidate_name,
        "candidate_area_tier": tier,
        "candidate_area_km2": candidate_area,
        "gt_lon": gt_lon,
        "gt_lat": gt_lat,
        "gt_inside_region": gt_inside,
        "inferred_time": inferred_time,
        "time_source": time_source,
        "transform": list(transform_)[:6],
        "crs": str(src.crs),
        "diagnostic": diagnostic,
        "rain_debug": rain_debug,
        "rainfall_grid_tag": "800mD2",
        "hydro_debug": hydro_debug,
        "candidate_region_cache": str(candidate_path),
    }
    cache.save_npz(
        arrays_path,
        s_norm=s_norm.astype(float),
        rain=rain.astype(float),
        fine=fine.astype(float),
        valid_mask=valid_mask.astype("uint8"),
    )
    cache.save_json(context_path, context_data)
    return (
        context_from_cache(
            context_data,
            {"s_norm": s_norm, "rain": rain, "fine": fine, "valid_mask": valid_mask},
        ),
        text_records,
        None,
    )


_LOCAL_CACHE_LOCK = None


_LOCAL_OSM_LOAD = fine_area.local_reconstruction.LocalOSMStore.load_local_osm_for_state


def _locked_local_osm_load(store, *args, **kwargs):
    # Parquet creation is not atomic in the validated local reconstruction helper.
    # Serialize just this shared cache access, not geometry reconstruction/evaluation.
    with _LOCAL_CACHE_LOCK:
        return _LOCAL_OSM_LOAD(store, *args, **kwargs)


def default_preprocessing_arguments(*, workflow="direct"):
    """Create preprocessing settings without invoking an experiment CLI."""
    if workflow not in {"full", "direct"}:
        raise ValueError(f"Unknown workflow: {workflow}")
    parser = argparse.ArgumentParser(add_help=False)
    add_fine_preprocessing_arguments(parser)
    args = parser.parse_args([])
    args.workflow = workflow
    args.hydro_timeout_seconds = 180
    args.hydro_cache_only = False
    args.rebuild_cache = False
    args.cache_dir = None
    return args
