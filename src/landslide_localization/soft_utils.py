from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import rasterio
import rasterio.features as rfeatures
from affine import Affine
from geopy.distance import geodesic
from shapely import wkb
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

from landslide_localization.output_paths import ensure_output_path
from landslide_localization.soft_config import (
    DEFAULT_W_R,
    DEFAULT_W_S,
    EXPERIMENT_FORMULAS,
    EXPERIMENT_ORDER,
    FULL_WEIGHTED_SOFT_OR_FORMULA,
    FUSION_METHOD,
    IMAGE_BASE_ROAD_WEIGHT,
    IMAGE_BASE_WATER_WEIGHT,
    IMAGE_JOINT_CAP,
    IMAGE_K_ROAD,
    IMAGE_K_WATER,
    IMAGE_MULTIPLIER_HI,
    IMAGE_MULTIPLIER_LO,
    OTHER_BUFFER_M,
    ROAD_BUFFER_M,
    TOP_K_VALUES,
    WATER_BUFFER_M,
    WEIGHTED_SOFT_OR_FORMULA,
)


UNIT_TOLERANCE = 1e-6


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Affine):
        return list(value)[:6]
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_hash(value: Any, length: int = 20) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def path_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class CacheManager:
    def __init__(self, root: Path, compatibility: Mapping[str, Any], rebuild: bool = False):
        self.root = ensure_output_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rebuild = bool(rebuild)
        self.hits = 0
        self.misses = 0
        self.compatibility = dict(compatibility)
        self.signature = stable_hash(self.compatibility)

    def path(self, category: str, key: str, suffix: str) -> Path:
        directory = self.root / category
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}_{self.signature}{suffix}"

    def existing(self, path: Path) -> bool:
        if path.exists() and not self.rebuild:
            self.hits += 1
            return True
        self.misses += 1
        return False

    def load_json(self, category: str, key: str) -> tuple[Optional[Any], Path]:
        path = self.path(category, key, ".json")
        if self.existing(path):
            try:
                return json.loads(path.read_text(encoding="utf-8")), path
            except Exception:
                self.hits -= 1
                self.misses += 1
        return None, path

    def save_json(self, path: Path, value: Any) -> None:
        write_json(path, value)

    def load_wkb(self, category: str, key: str) -> tuple[Optional[Any], Optional[str], Path]:
        path = self.path(category, key, ".wkb")
        missing = path.with_suffix(".missing.json")
        if not self.rebuild and path.exists():
            self.hits += 1
            return wkb.loads(path.read_bytes()), None, path
        if not self.rebuild and missing.exists():
            self.hits += 1
            data = json.loads(missing.read_text(encoding="utf-8"))
            return None, str(data.get("reason") or "cached_missing"), path
        self.misses += 1
        return None, None, path

    def save_wkb(self, path: Path, geom: Optional[Any], reason: str = "no_features") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        missing = path.with_suffix(".missing.json")
        if geom is None or geom.is_empty:
            write_json(missing, {"reason": reason})
            return
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(geom.wkb)
        temporary.replace(path)

    def load_npz(self, category: str, key: str) -> tuple[Optional[dict[str, np.ndarray]], Path]:
        path = self.path(category, key, ".npz")
        if self.existing(path):
            try:
                with np.load(path, allow_pickle=False) as data:
                    return {name: data[name] for name in data.files}, path
            except Exception:
                self.hits -= 1
                self.misses += 1
        return None, path

    def save_npz(self, path: Path, **arrays: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)


def configure_reference_modules(cache_dir: Path):
    """Load the shared spatial helpers and keep Nominatim caches in this run."""
    promap = importlib.import_module("landslide_localization.spatial")
    nominatim_dir = cache_dir / "nominatim"
    nominatim_dir.mkdir(parents=True, exist_ok=True)
    promap.NOMINATIM_CACHE_DIR = str(nominatim_dir)
    promap.NOMINATIM_CACHE_FILE = str(nominatim_dir / "nominatim_cache.json")
    promap._nominatim_cache = None
    fine_ref = importlib.import_module("landslide_localization.text_features")
    fine_ref.promap = promap
    return promap, fine_ref


def require_unit_interval(name: str, array: np.ndarray, valid: Optional[np.ndarray] = None) -> tuple[float, float]:
    mask = np.isfinite(array)
    if valid is not None:
        mask &= valid
    if not np.any(mask):
        raise RuntimeError(f"{name} has no finite evaluated values")
    minimum = float(np.min(array[mask]))
    maximum = float(np.max(array[mask]))
    if minimum < -UNIT_TOLERANCE or maximum > 1.0 + UNIT_TOLERANCE:
        raise RuntimeError(f"{name} outside [0, 1]: min={minimum}, max={maximum}")
    return minimum, maximum


def soft_or(s_norm: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """Return the original unweighted Soft-OR (kept for reference experiments)."""
    return s_norm + rain - s_norm * rain


def validate_soft_or_weights(w_s: float, w_r: float) -> tuple[float, float]:
    """Validate fusion weights without clipping or otherwise changing them."""
    weights = {"w_s": float(w_s), "w_r": float(w_r)}
    for name, value in weights.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a finite value in [0, 1]; got {value!r}")
    return weights["w_s"], weights["w_r"]


def weighted_soft_or(
    s_norm: np.ndarray,
    rain: np.ndarray,
    w_s: float = DEFAULT_W_S,
    w_r: float = DEFAULT_W_R,
) -> np.ndarray:
    """Return 1-(1-w_s*S_norm)*(1-w_r*R_id_trigger), without clipping."""
    w_s, w_r = validate_soft_or_weights(w_s, w_r)
    s_array = np.asarray(s_norm)
    r_array = np.asarray(rain)
    if np.any(np.isfinite(s_array)):
        require_unit_interval("S_norm", s_array)
    if np.any(np.isfinite(r_array)):
        require_unit_interval("R_id_trigger", r_array)
    weighted_s = w_s * s_array
    weighted_r = w_r * r_array
    result = weighted_s + weighted_r - weighted_s * weighted_r
    if np.any(np.isfinite(weighted_s)):
        require_unit_interval("w_s*S_norm", weighted_s)
    if np.any(np.isfinite(weighted_r)):
        require_unit_interval("w_r*R_id_trigger", weighted_r)
    if np.any(np.isfinite(result)):
        require_unit_interval("H_weighted_softOR", result)
    return result


def fusion_result_metadata(experiment: str, w_s: float, w_r: float) -> dict[str, Any]:
    """Metadata shared by per-case, overall, scale, and curve outputs."""
    w_s, w_r = validate_soft_or_weights(w_s, w_r)
    uses_weights = experiment in {"E3_sus_rain_softor", "E5_full_softor_fine"}
    return {
        "fusion_method": FUSION_METHOD,
        "w_s": w_s,
        "w_r": w_r,
        "formula": WEIGHTED_SOFT_OR_FORMULA,
        "full_formula": (
            FULL_WEIGHTED_SOFT_OR_FORMULA
            if experiment == "E5_full_softor_fine"
            else ""
        ),
        "experiment_formula": EXPERIMENT_FORMULAS[experiment],
        "fusion_weights_applied": uses_weights,
    }


def combine_multiplier_arrays(arrays: Iterable[np.ndarray], shape_: tuple[int, int]) -> np.ndarray:
    """Reference overlap rule: cellwise maximum over baseline one."""
    result = np.ones(shape_, dtype=float)
    for array in arrays:
        result = np.maximum(result, array.astype(float, copy=False))
    return result


def semantic_effective_multipliers(
    p_road: float,
    p_water: float,
    use_road_fallback: bool,
    use_water_fallback: bool,
) -> tuple[Optional[float], Optional[float], dict[str, float]]:
    """Exact reference semantic mapping plus its base weights and joint cap.

    The joint correction is relevant only when both image layers are active;
    a text layer is never indirectly modulated by same-type image evidence.
    """
    road_sem = float(np.clip(1.0 + IMAGE_K_ROAD * np.clip(p_road, 0.0, 1.0), IMAGE_MULTIPLIER_LO, IMAGE_MULTIPLIER_HI))
    water_sem = float(np.clip(1.0 + IMAGE_K_WATER * np.clip(p_water, 0.0, 1.0), IMAGE_MULTIPLIER_LO, IMAGE_MULTIPLIER_HI))
    road = IMAGE_BASE_ROAD_WEIGHT * road_sem if use_road_fallback else None
    water = IMAGE_BASE_WATER_WEIGHT * water_sem if use_water_fallback else None
    joint_scale = 1.0
    if road is not None and water is not None and road * water > IMAGE_JOINT_CAP:
        joint_scale = IMAGE_JOINT_CAP / (road * water)
        road *= joint_scale
        water *= joint_scale
    return road, water, {
        "road_semantic_multiplier": road_sem,
        "water_semantic_multiplier": water_sem,
        "joint_scale": float(joint_scale),
    }


def feature_buffer_distance(category: str) -> float:
    return {"road": ROAD_BUFFER_M, "water": WATER_BUFFER_M, "other": OTHER_BUFFER_M}[category]


def valid_geometry(geom: Any) -> Optional[Any]:
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None
    return None if geom.is_empty else geom


def is_focus_area_evidence(evidence: Mapping[str, Any]) -> bool:
    """Identify the source's regional evidence, independent of feature naming."""
    return str(evidence.get("category") or "").strip().casefold() == "focusarea"


def record_is_focus_area(
    item: Mapping[str, Any], record: Mapping[str, Any], index: Optional[int] = None,
) -> bool:
    """Recognize regional records, including old positional seed-cache rows.

    Stored ``evidence_index`` is one-based; ``index`` is the zero-based position
    in a complete seed-record list. Legacy positional inference also checks the
    feature name, so an already-filtered final record is not mistaken for the
    FocusArea that previously occupied that position.
    """
    if "evidence_category" in record:
        return is_focus_area_evidence({"category": record["evidence_category"]})
    if any(str(record.get(key) or "").strip().casefold() == "focusarea"
           for key in ("category", "feature_type")):
        return True
    raw = list(item.get("weight_evidence") or [])
    stored_index = record.get("evidence_index")
    if isinstance(stored_index, int) and not isinstance(stored_index, bool):
        position = stored_index - 1
        return 0 <= position < len(raw) and is_focus_area_evidence(raw[position])
    if index is None or not 0 <= index < len(raw):
        return False
    evidence = raw[index]
    if not is_focus_area_evidence(evidence):
        return False
    name = (evidence.get("token") or evidence.get("normalized_token")
            or evidence.get("resolved_name") or evidence.get("name")
            or f"text_{index + 1}")
    return str(record.get("raw_feature_name") or "") == str(name)


def sanitize_focus_area_records(
    item: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove only FocusArea local buffers while preserving seed row alignment.

    Returns copies and is safe to call on historical seed caches during an
    explicit migration. Geometry of the administrative candidate remains in
    ``item``; a FocusArea row is retained only as an excluded audit record.
    """
    aligned = len(records) == len(item.get("weight_evidence") or [])
    cleaned = []
    for index, source in enumerate(records):
        record = dict(source)
        if record_is_focus_area(item, record, index if aligned else None):
            record.update(
                evidence_category="FocusArea", fine_category="focus_area",
                resolution_success=False, buffer_distance_m=0.0,
                buffer_area_km2=0.0, included_in_candidate_union=False,
                used_as_fine_evidence=False,
                failure_reason="focus_area_not_local_feature",
            )
            if aligned:
                record.setdefault("evidence_index", index + 1)
            record.pop("buffer_geojson", None)
            record.pop("feature_id", None)
        cleaned.append(record)
    return cleaned


def resolve_text_feature_records(item: Mapping[str, Any], promap, fine_ref) -> list[dict[str, Any]]:
    """Apply the reference text eligibility, resolution, and typing rules."""
    raw = list(item.get("weight_evidence") or [])
    usable_ids = {id(evidence) for evidence in promap.local_text_evidence(dict(item))}
    rows: list[dict[str, Any]] = []
    category_indexes: Counter[str] = Counter()
    for index, evidence in enumerate(raw, start=1):
        original, reference_category = fine_ref.classify_original_feature_type(evidence)
        category = "water" if reference_category == "river" else reference_category
        raw_name = fine_ref._feature_name(evidence, f"text_{index}")
        record: dict[str, Any] = {
            "tweets_id": str(item.get("tweets_id")),
            "evidence_index": index,
            "evidence_category": str(evidence.get("category") or ""),
            "raw_feature_name": raw_name,
            "resolved_feature_name": str(evidence.get("resolved_name") or ""),
            "feature_type": original,
            "fine_category": category,
            "geometry_type": str(evidence.get("geometry_type") or ""),
            "resolution_success": False,
            "buffer_distance_m": feature_buffer_distance(category),
            "buffer_area_km2": 0.0,
            "included_in_candidate_union": False,
            "used_as_fine_evidence": False,
            "failure_reason": "",
        }
        if is_focus_area_evidence(evidence):
            # Keep one placeholder per source evidence for fine_area's index
            # lookup, but never resolve or buffer regional evidence as other.
            record.update(
                fine_category="focus_area", buffer_distance_m=0.0,
                failure_reason="focus_area_not_local_feature",
            )
            rows.append(record)
            continue
        if id(evidence) not in usable_ids:
            record["failure_reason"] = "text_evidence_not_eligible"
            rows.append(record)
            continue
        try:
            geom_wgs = fine_ref._resolve_text_geometry(evidence)
        except Exception as exc:
            geom_wgs = None
            record["failure_reason"] = f"geometry_resolution_failed:{type(exc).__name__}"
        if geom_wgs is None:
            record["failure_reason"] = record["failure_reason"] or "geometry_resolution_failed"
            rows.append(record)
            continue
        try:
            geom_work = valid_geometry(promap.reproject_geom(geom_wgs, "EPSG:4326", promap.WORK_CRS))
        except Exception as exc:
            geom_work = None
            record["failure_reason"] = f"projection_failed:{type(exc).__name__}"
        if geom_work is None:
            record["failure_reason"] = record["failure_reason"] or "invalid_feature_geometry"
            rows.append(record)
            continue

        category_indexes[category] += 1
        buffered = valid_geometry(geom_work.buffer(feature_buffer_distance(category)))
        if buffered is None:
            record["failure_reason"] = "buffer_failed"
            rows.append(record)
            continue
        record.update(
            {
                "feature_id": f"{item.get('tweets_id')}:{category}:text:{category_indexes[category]:04d}",
                "geometry_type": geom_work.geom_type,
                "resolution_success": True,
                "buffer_area_km2": float(buffered.area / 1e6),
                "included_in_candidate_union": True,
                "used_as_fine_evidence": True,
                "failure_reason": "",
                "geometry_geojson": mapping(geom_wgs),
                "buffer_geojson": mapping(promap.reproject_geom(buffered, promap.WORK_CRS, "EPSG:4326")),
            }
        )
        rows.append(record)
    return rows


def cacheable_text_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def build_candidate_region(item: Mapping[str, Any], records: Sequence[Mapping[str, Any]], promap):
    base_geojson, _ = promap.candidate_geometry(dict(item))
    if not isinstance(base_geojson, dict):
        raise ValueError("candidate_geometry_unresolved")
    base_wgs = valid_geometry(shape(base_geojson))
    if base_wgs is None:
        raise ValueError("invalid_administrative_geometry")
    base_work = valid_geometry(promap.reproject_geom(base_wgs, "EPSG:4326", promap.WORK_CRS))
    if base_work is None:
        raise ValueError("invalid_projected_administrative_geometry")
    administrative = valid_geometry(
        promap.build_analysis_polygon(base_work, base_wgs.geom_type)
    )
    if administrative is None:
        raise ValueError("empty_administrative_region")

    category_buffers: dict[str, list[Any]] = {"road": [], "water": [], "other": []}
    aligned = len(records) == len(item.get("weight_evidence") or [])
    for index, record in enumerate(records):
        if record_is_focus_area(item, record, index if aligned else None):
            continue
        if not record.get("resolution_success") or not isinstance(record.get("buffer_geojson"), dict):
            continue
        buffered_wgs = shape(record["buffer_geojson"])
        buffered_work = valid_geometry(promap.reproject_geom(buffered_wgs, "EPSG:4326", promap.WORK_CRS))
        if buffered_work is not None:
            category_buffers[str(record["fine_category"])].append(buffered_work)
    unions = {
        category: (valid_geometry(unary_union(parts)) if parts else None)
        for category, parts in category_buffers.items()
    }
    text_parts = [geom for geom in unions.values() if geom is not None]
    text_union = valid_geometry(unary_union(text_parts)) if text_parts else None
    final = valid_geometry(unary_union([administrative] + text_parts))
    if final is None:
        raise ValueError("empty_final_candidate_region")

    # Contract: only text buffers can enlarge the administrative region.
    if administrative.difference(final).area > 1e-3:
        raise AssertionError("CandidateRegion does not contain AdministrativeRegion")
    for category, geom in unions.items():
        if geom is not None and geom.difference(final).area > 1e-3:
            raise AssertionError(f"CandidateRegion does not contain text {category} buffers")
    return administrative, unions, text_union, final


def candidate_geojson(
    tweets_id: str,
    administrative: Any,
    unions: Mapping[str, Any],
    final: Any,
    promap,
) -> dict[str, Any]:
    features = []
    for role, geom in [("administrative", administrative)]:
        features.append(
            {
                "type": "Feature",
                "properties": {"tweets_id": tweets_id, "role": role},
                "geometry": mapping(promap.reproject_geom(geom, promap.WORK_CRS, "EPSG:4326")),
            }
        )
    for category in ("road", "water", "other"):
        geom = unions.get(category)
        if geom is not None:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "tweets_id": tweets_id,
                        "role": "text_feature_buffers",
                        "fine_category": category,
                        "buffer_distance_m": feature_buffer_distance(category),
                    },
                    "geometry": mapping(promap.reproject_geom(geom, promap.WORK_CRS, "EPSG:4326")),
                }
            )
    features.append(
        {
            "type": "Feature",
            "properties": {
                "tweets_id": tweets_id,
                "role": "final_candidate_region",
                "definition": "AdministrativeRegion UNION textual fine-feature buffers only",
            },
            "geometry": mapping(promap.reproject_geom(final, promap.WORK_CRS, "EPSG:4326")),
        }
    )
    return {"type": "FeatureCollection", "features": features}


def parse_candidate_geojson(data: Mapping[str, Any], promap):
    administrative = None
    unions: dict[str, Any] = {"road": None, "water": None, "other": None}
    final = None
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        geom = promap.reproject_geom(shape(feature["geometry"]), "EPSG:4326", promap.WORK_CRS)
        if props.get("role") == "administrative":
            administrative = geom
        elif props.get("role") == "text_feature_buffers":
            unions[str(props.get("fine_category"))] = geom
        elif props.get("role") == "final_candidate_region":
            final = geom
    if administrative is None or final is None:
        raise ValueError("Incomplete cached candidate GeoJSON")
    parts = [geom for geom in unions.values() if geom is not None]
    text_union = unary_union(parts) if parts else None
    return administrative, unions, text_union, final


def rasterize_multiplier(geom_work: Any, multiplier: float, shape_: tuple[int, int], transform_: Affine, crs, promap) -> np.ndarray:
    geom_raster = promap.reproject_geom(geom_work, promap.WORK_CRS, crs)
    return rfeatures.rasterize(
        [(geom_raster, float(multiplier))],
        out_shape=shape_,
        transform=transform_,
        fill=1.0,
        dtype=float,
    )


def extract_hydro_buffers(
    cache: CacheManager,
    administrative: Any,
    hydro_aoi_wgs: Any,
    need_road: bool,
    need_water: bool,
    promap,
    gdb_path: Optional[Path],
    timeout_seconds: int,
    legacy_osmnx_cache: Optional[Path],
    hydro_cache_only: bool,
    road_pbf: Optional[Path] = None,
    road_pbf_timeout_seconds: Optional[int] = None,
    road_pbf_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    water_pbf_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[Optional[Any], Optional[Any], dict[str, Any]]:
    # Use the exact source GeoJSON coordinates, like spatial,
    # so validated OSMnx response-cache keys remain reusable. Buffers returned by
    # the worker are still internal evidence and never enter CandidateRegion.
    geom_hash = hashlib.sha256(hydro_aoi_wgs.wkb).hexdigest()[:24]
    road_pbf = road_pbf.resolve() if road_pbf is not None else None

    def prepare_pbf_candidates(
        component: str,
        candidates: Optional[Sequence[Mapping[str, Any]]],
        legacy_path: Optional[Path] = None,
    ) -> tuple[list[dict[str, Any]], str]:
        values = [dict(value) for value in (candidates or [])]
        if not values and legacy_path is not None:
            values = [{"label": "national_full", "path": str(legacy_path)}]
        prepared = []
        signatures = []
        for value in values:
            source_path = Path(str(value["path"])).resolve()
            signature = path_signature(source_path)
            source_key = stable_hash(signature, length=16)
            prepared.append({
                "label": str(value.get("label") or "local_pbf"),
                "path": str(source_path),
                "raw_cache": str(
                    cache.root
                    / f"{component}_pbf_raw"
                    / f"{component}_pbf_{source_key}_{geom_hash}.wkb"
                ),
            })
            signatures.append({"label": value.get("label"), "signature": signature})
        chain_key = (
            f"pbfchain_{stable_hash(signatures, length=16)}"
            if signatures
            else "overpass"
        )
        return prepared, chain_key

    prepared_road_candidates, road_source_key = prepare_pbf_candidates(
        "road", road_pbf_candidates, road_pbf
    )
    prepared_water_candidates, water_source_key = prepare_pbf_candidates(
        "water", water_pbf_candidates
    )
    road_key = f"road_{road_source_key}_{geom_hash}_{int(ROAD_BUFFER_M)}m"
    water_key = f"water_{water_source_key}_{geom_hash}_{int(WATER_BUFFER_M)}m"
    road = water = None
    road_reason = water_reason = None
    road_path = water_path = None
    road_needs_compute = water_needs_compute = False
    if need_road:
        road, road_reason, road_path = cache.load_wkb("hydro", road_key)
        road_needs_compute = road is None and road_reason is None
    if need_water:
        water, water_reason, water_path = cache.load_wkb("hydro", water_key)
        water_needs_compute = water is None and water_reason is None
    if not road_needs_compute and not water_needs_compute:
        return road, water, {
            "road_reason": road_reason,
            "water_reason": water_reason,
            "road_source": "pbf_chain_buffer_cache" if prepared_road_candidates else "overpass",
            "water_source": "pbf_chain_buffer_cache" if prepared_water_candidates else "overpass",
            "road_pbf_candidates": prepared_road_candidates,
            "water_pbf_candidates": prepared_water_candidates,
            "road_buffer_cache_hit": bool(need_road),
            "water_buffer_cache_hit": bool(need_water),
            "worker_used": False,
        }

    worker = Path(__file__).resolve().parent / "hydro_worker.py"
    temporary_root = cache.root / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    admin_geojson = mapping(hydro_aoi_wgs)

    def run_component(component: str, directory: Path) -> tuple[Optional[Any], Optional[str], dict[str, Any]]:
        input_path = directory / f"{component}_input.json"
        output_path = directory / f"{component}_output.json"
        input_payload = {
            "geometry": admin_geojson,
            "work_crs": promap.WORK_CRS,
            "gdb_path": str(gdb_path) if gdb_path else None,
            "road_buffer_m": ROAD_BUFFER_M,
            "water_buffer_m": WATER_BUFFER_M,
            "need_road": component == "road",
            "need_water": component == "water",
            "soft_osmnx_cache": str(cache.root / "osmnx"),
            "hydro_cache_only": bool(hydro_cache_only),
            "road_pbf_candidates": prepared_road_candidates,
            "water_pbf_candidates": prepared_water_candidates,
        }
        write_json(input_path, input_payload)
        try:
            component_timeout = (
                int(road_pbf_timeout_seconds)
                if (
                    road_pbf_timeout_seconds is not None
                    and (
                        (component == "road" and prepared_road_candidates)
                        or (component == "water" and prepared_water_candidates)
                    )
                )
                else int(timeout_seconds)
            )
            # Zero explicitly disables the local PBF component time limit.
            if component_timeout == 0:
                component_timeout = None
            completed = subprocess.run(
                [sys.executable, str(worker), str(input_path), str(output_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=component_timeout,
                check=False,
            )
            if completed.returncode != 0 or not output_path.exists():
                detail = (completed.stderr or completed.stdout or f"exit_code={completed.returncode}")[-2000:]
                raise RuntimeError(detail)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            value = result.get(f"{component}_union_wkb")
            geom = wkb.loads(value, hex=True) if value else None
            reason = (result.get("errors") or {}).get(component) or (None if geom is not None else f"no_{component}_features")
            return geom, reason, result
        except Exception as exc:
            return None, f"hydro_worker_failed:{type(exc).__name__}:{exc}", {}

    with tempfile.TemporaryDirectory(prefix="hydro_", dir=temporary_root) as directory_name:
        directory = Path(directory_name)
        components = []
        if road_needs_compute:
            components.append("road")
        if water_needs_compute:
            components.append("water")
        with ThreadPoolExecutor(max_workers=len(components)) as executor:
            futures = {component: executor.submit(run_component, component, directory) for component in components}
            results = {component: future.result() for component, future in futures.items()}

    road_worker_debug: dict[str, Any] = {}
    water_worker_debug: dict[str, Any] = {}
    if road_needs_compute:
        road, road_reason, road_worker_debug = results["road"]
        if road is not None or road_reason == "no_road_features":
            cache.save_wkb(road_path, road, road_reason or "no_road_features")
    if water_needs_compute:
        water, water_reason, water_worker_debug = results["water"]
        if water is not None or water_reason == "no_water_features":
            cache.save_wkb(water_path, water, water_reason or "no_water_features")
    return road, water, {
        "road_reason": road_reason,
        "water_reason": water_reason,
        "road_source": road_worker_debug.get("road_source") or ("local_pbf" if prepared_road_candidates else "overpass"),
        "road_raw_cache": road_worker_debug.get("road_raw_cache"),
        "road_raw_cache_hit": road_worker_debug.get("road_raw_cache_hit"),
        "road_buffer_cache_hit": False if road_needs_compute else bool(need_road),
        "water_source": water_worker_debug.get("water_source"),
        "water_raw_cache": water_worker_debug.get("water_raw_cache"),
        "water_raw_cache_hit": water_worker_debug.get("water_raw_cache_hit"),
        "water_buffer_cache_hit": False if water_needs_compute else bool(need_water),
        "road_pbf_attempts": road_worker_debug.get("road_pbf_attempts") or [],
        "water_pbf_attempts": water_worker_debug.get("water_pbf_attempts") or [],
        "worker_used": True,
    }


def experiment_scores(
    s_norm: np.ndarray,
    rain: np.ndarray,
    fine: np.ndarray,
    selected: Sequence[str],
    w_s: float = DEFAULT_W_S,
    w_r: float = DEFAULT_W_R,
) -> dict[str, np.ndarray]:
    w_s, w_r = validate_soft_or_weights(w_s, w_r)
    result: dict[str, np.ndarray] = {}
    sr = None
    for experiment in selected:
        if experiment == "E1_sus":
            result[experiment] = s_norm
        elif experiment == "E2_rain":
            result[experiment] = rain
        elif experiment == "E3_sus_rain_softor":
            sr = weighted_soft_or(s_norm, rain, w_s, w_r) if sr is None else sr
            result[experiment] = sr
        elif experiment == "E4_sus_fine":
            result[experiment] = s_norm * fine
        elif experiment == "E6_rain_fine":
            result[experiment] = rain * fine
        elif experiment == "E5_full_softor_fine":
            sr = weighted_soft_or(s_norm, rain, w_s, w_r) if sr is None else sr
            result[experiment] = sr * fine
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
    return result


def evaluate_score(
    experiment: str,
    score: np.ndarray,
    transform_: Affine,
    gt_lon: Optional[float],
    gt_lat: Optional[float],
    base: Mapping[str, Any],
    promap,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    valid = np.isfinite(score)
    row: dict[str, Any] = {
        "tweets_id": base["tweets_id"],
        "experiment_id": experiment,
        "candidate_area_tier": base.get("candidate_area_tier"),
        "candidate_area_km2": base.get("candidate_area_km2"),
        "gt_inside_region": bool(base.get("gt_inside_region")),
        "gt_in_valid_raster": False,
        "has_prediction": bool(np.any(valid)),
        "score_at_gt": None,
        "PR": 0.0,
        "point_error_km": None,
        "failure_reason": base.get("failure_reason") or "",
    }
    for k in TOP_K_VALUES:
        row[f"HitTop{k}"] = False
        # Area@K is undefined when Stage II cannot be evaluated.  In
        # particular, do not manufacture zero-area values for out-of-region
        # cases; aggregate Area@K only over the Stage-II-valid population.
        row[f"AreaTop{k}_km2"] = None
        row[f"AreaTop{k}_percent"] = None
    if not np.any(valid):
        return row, None

    prediction = None
    valid_indices = np.where(valid)
    flat_index = int(np.nanargmax(score[valid]))
    best_row = int(valid_indices[0][flat_index])
    best_col = int(valid_indices[1][flat_index])
    best_x, best_y = rasterio.transform.xy(transform_, best_row, best_col)
    if gt_lon is not None and gt_lat is not None:
        metrics = promap.evaluate_probmap_case_from_array(
            score,
            transform_,
            float(gt_lon),
            float(gt_lat),
            top_percents=TOP_K_VALUES,
        )
        row["raw_gt_pixel_finite"] = bool(metrics["gt_in_valid_raster"])
        row["gt_in_valid_raster"] = bool(row["gt_inside_region"] and metrics["gt_in_valid_raster"])
        if row["gt_in_valid_raster"]:
            row["score_at_gt"] = metrics.get("gt_pixel_prob")
            row["PR"] = float(metrics.get("pr") or 0.0)
            for k in TOP_K_VALUES:
                row[f"HitTop{k}"] = bool(metrics[f"hit_top{k}"])
                row[f"AreaTop{k}_km2"] = float(metrics[f"area_top{k}_km2"])
                row[f"AreaTop{k}_percent"] = float(metrics[f"area_top{k}_percent"])
        else:
            detail = ("ground_truth_outside_candidate_geometry" if not row["gt_inside_region"]
                      else "ground_truth_outside_finite_stage2_raster")
            row["failure_reason"] = ";".join(
                value for value in (str(row.get("failure_reason") or ""), detail) if value
            )
        row["point_error_km"] = float(
            geodesic((float(best_y), float(best_x)), (float(gt_lat), float(gt_lon))).km
        )
    prediction = {
        "type": "Feature",
        "geometry": mapping(Point(float(best_x), float(best_y))),
        "properties": {
            "tweets_id": base["tweets_id"],
            "experiment_id": experiment,
            "best_score": float(score[best_row, best_col]),
            "candidate_area_tier": base.get("candidate_area_tier"),
        },
    }
    return row, prediction


def missing_metric_row(item: Mapping[str, Any], experiment: str, reason: str) -> dict[str, Any]:
    area = item.get("candidate_area") or {}
    row: dict[str, Any] = {
        "tweets_id": str(item.get("tweets_id")),
        "experiment_id": experiment,
        "candidate_area_tier": item.get("candidate_area_tier") or area.get("tier"),
        "candidate_area_km2": None,
        "gt_inside_region": False,
        "gt_in_valid_raster": False,
        "has_prediction": False,
        "score_at_gt": None,
        "PR": 0.0,
        "point_error_km": None,
        "failure_reason": reason,
    }
    for k in TOP_K_VALUES:
        row[f"HitTop{k}"] = False
        row[f"AreaTop{k}_km2"] = None
    return row


def selected_experiments(value: str) -> list[str]:
    normalized = value.strip().casefold()
    if normalized == "all":
        return list(EXPERIMENT_ORDER)
    aliases = {
        "full": "E5_full_softor_fine",
        "e5": "E5_full_softor_fine",
        "e1": "E1_sus",
        "e2": "E2_rain",
        "e3": "E3_sus_rain_softor",
        "e4": "E4_sus_fine",
        "e6": "E6_rain_fine",
        "fine+rain": "E6_rain_fine",
        "rain+fine": "E6_rain_fine",
        "sus+rain": "E3_sus_rain_softor",
        "rain+sus": "E3_sus_rain_softor",
    }
    if normalized in aliases:
        return [aliases[normalized]]
    for experiment in EXPERIMENT_ORDER:
        if value == experiment or normalized == experiment.casefold():
            return [experiment]
    raise ValueError(f"Unknown experiment selection {value!r}; use all, full, or E1-E6")
