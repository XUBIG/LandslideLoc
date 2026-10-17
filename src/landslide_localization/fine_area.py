"""Reconstruct and buffer fine features shared by Full and Direct localization."""
from __future__ import annotations
import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from landslide_localization.soft_config import ROAD_BUFFER_M, WATER_BUFFER_M, OTHER_BUFFER_M
from landslide_localization import local_fine_reconstruction as local_reconstruction

from landslide_localization.runtime_paths import REPO_ROOT as ROOT

def resolved_path(value: str | Path, must_exist: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _state_file_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")



@lru_cache(maxsize=8)
def _state_pbf_index(directory: Path) -> dict[str, Path]:
    return {
        _state_file_key(path.name[:-8]): path.resolve()
        for path in directory.glob("*.osm.pbf")
        if path.is_file()
    }


def _case_state_name(item: Mapping[str, Any]) -> str:
    source = item.get("candidate_area") or {}
    state = source.get("admin1") or item.get("admin1")
    tier = str(item.get("candidate_area_tier") or source.get("tier") or "").casefold()
    if not state and "state" in tier:
        state = source.get("name") or source.get("display_name")
    return str(state or "").strip()


def _case_pbf_candidates(
    item: Mapping[str, Any], args: argparse.Namespace, component: str
) -> list[dict[str, str]]:
    state_key = _state_file_key(_case_state_name(item))
    specialized_dir = Path(
        args.road_state_dir if component == "road" else args.hydro_state_dir
    )
    full_state_dir = Path(args.osm_state_dir)
    candidates: list[dict[str, str]] = []
    if state_key:
        specialized = _state_pbf_index(specialized_dir).get(state_key)
        complete = _state_pbf_index(full_state_dir).get(state_key)
        if specialized is not None:
            candidates.append({
                "label": "state_road_only" if component == "road" else "state_hydro_only",
                "path": str(specialized),
            })
        if complete is not None:
            candidates.append({"label": "state_full", "path": str(complete)})
    national = Path(args.road_pbf).resolve()
    if national.is_file():
        candidates.append({"label": "national_full", "path": str(national)})
    seen: set[str] = set()
    return [
        entry for entry in candidates
        if not (entry["path"] in seen or seen.add(entry["path"]))
    ]


def _pbf_directory_signature(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("*.osm.pbf"), key=lambda value: value.name.casefold())
    return {
        "path": str(directory.resolve()),
        "files": [
            {
                "name": path.name,
                "size": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
            for path in files
        ],
    }


def _usable_geometry_type(category: str, geom_type: str) -> bool:
    if category == "road":
        return geom_type in {"LineString", "MultiLineString"}
    if category == "water":
        return geom_type in {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}
    return True


def _fallback_reason(category: str, records: Sequence[Mapping[str, Any]]) -> str:
    selected = [record for record in records if record.get("fine_category") == category]
    if not selected:
        return "NO_TEXT_FEATURE"
    if any(record.get("initial_geometry_type") in {"Point", "MultiPoint"} for record in selected):
        return "TEXT_POINT_RECOVERY_FAILED"
    if any(not record.get("initial_resolution_success") for record in selected):
        return "TEXT_RESOLUTION_FAILED"
    return "INVALID_TEXT_GEOMETRY"


def _prepare_local_text_records(item, initial_records, administrative, candidate, promap, args):
    """Match local features within the administrative area, then buffer and clip.

    Preserve connected geometry before clipping to the final candidate area.
    """
    local = local_reconstruction
    case_id = str(item.get("tweets_id"))
    area = item.get("candidate_area") or {}
    state = str(area.get("admin1") or "")
    region = promap.reproject_geom(administrative, promap.WORK_CRS, "EPSG:4326")
    inventory, seen = [], set()
    for index, evidence in enumerate(item.get("weight_evidence") or []):
        if evidence.get("category") == "FocusArea":
            continue
        name = local.fine_ref._feature_name(evidence, f"text_{index + 1}")
        seen.add(local.plain_name(name))
        kind, category = local.classify(evidence)
        record = dict(initial_records[index]) if index < len(initial_records) else {}
        inventory.append({
            "original_fine_name": name, "fine_type": kind, "category": category,
            "resolved_name": evidence.get("resolved_name", ""),
            "seed_geometry": evidence.get("geojson"), "legacy_record": record,
            "inventory_source": "weight_evidence",
        })
    for name in (item.get("parsed_answer") or {}).get("local_features", []):
        normalized = local.plain_name(name)
        if not normalized or normalized in seen or normalized in {"none", "null", "n a"}:
            continue
        seen.add(normalized)
        kind, category = local.classify({"token": name})
        inventory.append({
            "original_fine_name": name, "fine_type": kind, "category": category,
            "resolved_name": "", "seed_geometry": None, "legacy_record": {},
            "inventory_source": "unresolved_local_mention",
        })
    for index, row in enumerate(inventory, 1):
        row.update(case_id=case_id, state=state,
                   candidate_area_tier=item.get("candidate_area_tier"),
                   record_id=f"{case_id}_{index:03d}")
    local_args = argparse.Namespace(
        osm_dir=resolved_path(args.osm_state_dir),
        road_state_dir=resolved_path(args.road_state_dir),
        hydro_state_dir=resolved_path(args.hydro_state_dir),
        output=resolved_path(args.cache_dir, must_exist=False) / "local_text_reconstruction",
        local_cache_dir=resolved_path(getattr(args, "local_text_cache_dir", None)
                                     or ROOT / "outputs/local_fine_reconstruction/cache", must_exist=False),
        search_buffer_m=args.local_search_buffer_m,
        endpoint_tolerance_m=args.local_endpoint_tolerance_m,
        seed_tolerance_m=args.local_seed_tolerance_m,
        memory_states=args.local_memory_states, raw_policy=args.local_raw_policy,
    )
    local_args.output.mkdir(parents=True, exist_ok=True)
    case = {"region": region, "region_error": ""}
    store = None
    if any(row["category"] != "OTHER" for row in inventory):
        print(f"[{case_id}] Local text reconstruction: {len(inventory)} feature(s)", flush=True)
        polygons = {state: region} if item.get("candidate_area_tier") == "state_area" else {}
        store = local.LocalOSMStore(local_args, inventory, {case_id: case}, polygons)
    corrected, diagnostics = [], []
    for row in inventory:
        record = dict(row["legacy_record"])
        category = {"ROAD": "road", "HYDRO": "water", "OTHER": "other"}[row["category"]]
        initial_geom = local.safe_shape(record.get("geometry_geojson") or row["seed_geometry"])
        initial_type = initial_geom.geom_type if initial_geom is not None else str(record.get("geometry_type") or "")
        is_point = initial_type in {"Point", "MultiPoint"} and category != "other"
        buffer_m = {"road": ROAD_BUFFER_M, "water": WATER_BUFFER_M, "other": OTHER_BUFFER_M}[category]
        record.update(
            tweets_id=case_id, feature_id=f"{case_id}:{category}:text:{row['record_id']}",
            raw_feature_name=row["original_fine_name"], resolved_feature_name=row["resolved_name"],
            feature_type=row["fine_type"], fine_category=category, buffer_distance_m=buffer_m,
            initial_resolution_success=bool(record.get("resolution_success")),
            initial_geometry_type=initial_type, point_recovery_attempted=is_point,
            point_recovery_success=False, final_text_usable=False, used_as_fine_evidence=False,
            buffer_geojson=None, buffer_area_km2=0.0,
        )
        detail = {
            "feature": row["original_fine_name"], "category": category,
            "query_name": row["original_fine_name"], "inventory_source": row["inventory_source"],
            "initial_resolution_success": record["initial_resolution_success"],
            "initial_geometry_type": initial_type, "point_recovery_attempted": is_point,
            "point_recovery_success": False, "anchor_point_used": row["seed_geometry"] is not None,
            "final_usable": False, "final_geometry_type": None,
            "failure_type": "", "failure_reason": "", "recovery_cache_source": "local_osm",
            "lookup_aoi": "administrative Candidate Area plus local search buffer",
        }
        try:
            result = local.evaluate_reconstruction(row, case, store, local_args)
            # Strip live Shapely objects and LocalFeature instances from diagnostics.
            detail["local_reconstruction"] = {
                key: value for key, value in result.items()
                if key not in {"geometry", "clipped", "selected", "candidates"}
            }
            detail["recovery_method"] = result.get("match_method")
            detail["recovery_notes"] = result.get("notes", [])
            detail["candidate_matches"] = result.get("candidate_matches", 0)
            detail["selected_match"] = {"osm_way_ids": [f.osm_id for f in result["selected"]]}
            geom = result["geometry"]
            if not result["status"].startswith("SUCCESS") or geom.is_empty:
                detail["failure_type"] = "TEXT_NAME_MATCH_FAILED" if result["status"] == "AMBIGUOUS" else "TEXT_FEATURE_NOT_FOUND"
                detail["failure_reason"] = "; ".join(result.get("notes") or [result["status"]])
            elif category != "other" and not _usable_geometry_type(category, geom.geom_type):
                detail["failure_type"] = "TEXT_INVALID_GEOMETRY"
                detail["failure_reason"] = f"{category} geometry type {geom.geom_type} is not usable"
            else:
                work = promap.reproject_geom(geom, "EPSG:4326", promap.WORK_CRS)
                buffered = work.buffer(buffer_m).intersection(candidate)
                if buffered.is_empty or not buffered.is_valid:
                    raise ValueError("empty/invalid buffer after Candidate Area clipping")
                record.update(
                    geometry_geojson=mapping(geom), geometry_type=geom.geom_type,
                    buffer_geojson=mapping(promap.reproject_geom(buffered, promap.WORK_CRS, "EPSG:4326")),
                    buffer_area_km2=float(buffered.area / 1e6), resolution_success=True,
                    final_text_usable=True, used_as_fine_evidence=True,
                    point_recovery_success=is_point, failure_reason="",
                )
                detail.update(final_usable=True, final_geometry_type=geom.geom_type,
                              point_recovery_success=is_point,
                              final_buffer_geometry_type=buffered.geom_type,
                              final_buffer_area_km2=record["buffer_area_km2"])
        except Exception as exc:
            detail["failure_type"] = "TEXT_RESOLUTION_FAILED"
            detail["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if detail["failure_type"]:
            record["failure_reason"] = detail["failure_reason"]
        corrected.append(record)
        diagnostics.append(detail)
    return corrected, diagnostics


def _corrected_text_unions(
    records: Sequence[Mapping[str, Any]], candidate: Any, promap
) -> dict[str, Any]:
    parts: dict[str, list[Any]] = {"road": [], "water": [], "other": []}
    for record in records:
        if not record.get("final_text_usable") or not isinstance(record.get("buffer_geojson"), dict):
            continue
        try:
            buffered = promap.reproject_geom(
                shape(record["buffer_geojson"]), "EPSG:4326", promap.WORK_CRS
            ).intersection(candidate)
        except Exception:
            continue
        if not buffered.is_empty:
            parts[str(record.get("fine_category") or "other")].append(buffered)
    return {
        category: (unary_union(geometries) if geometries else None)
        for category, geometries in parts.items()
    }

