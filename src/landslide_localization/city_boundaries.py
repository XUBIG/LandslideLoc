"""Apply explicitly verified city boundaries without using ground-truth locations.

This module only reads a versioned manifest. It does not geocode or select a
boundary using a case's coordinates. Entity identity and valid, intersecting
geometries are checked before any replacement is returned.
"""
from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from shapely.geometry import shape


SCHEMA_VERSION = "verified_city_boundaries_v1"
POLICY_VERSION = "verified_city_boundary_overrides_v1"
CITY_TYPES = {"city", "town", "municipality", "municipal_district", "borough", "independent_city"}
BROAD_TYPES = {"state", "province", "region", "country", "continent", "nation", "federal_state"}


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _entity_name(value: Any) -> str:
    return re.sub(r"^(?:city|town|municipality|borough) of\s+", "", _normalized(value))


def _validate_coordinates(value: Any) -> None:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("Boundary geometry has missing or malformed coordinates")
    if isinstance(value[0], (int, float)):
        if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) for v in value):
            raise ValueError("Boundary positions must be finite two-dimensional coordinates")
        if not (-180 <= value[0] <= 180 and -90 <= value[1] <= 90):
            raise ValueError("Boundary coordinates must be WGS84 longitude/latitude")
        return
    for child in value:
        _validate_coordinates(child)


def _polygon(geometry: Any, *, allow_bbox: bool = False):
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Boundary must be a Polygon or MultiPolygon")
    _validate_coordinates(geometry.get("coordinates"))
    polygon = shape(geometry)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Boundary must be nonempty and geometrically valid")
    if not allow_bbox and polygon.equals(polygon.envelope):
        raise ValueError("A rectangular bbox is not a verified city boundary")
    return polygon


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required boundary manifest field: {key}")
    return value.strip()


def _validate_entry(entry: Any) -> None:
    if not isinstance(entry, dict):
        raise ValueError("Every boundary manifest entry must be an object")
    _required_text(entry, "case_id")
    if isinstance(entry.get("expected_geoname_id"), bool) or not re.fullmatch(
            r"[1-9][0-9]*", str(entry.get("expected_geoname_id", ""))):
        raise ValueError("expected_geoname_id must be a positive GeoNames identifier")
    boundary = entry.get("boundary")
    if not isinstance(boundary, dict):
        raise ValueError("Boundary manifest entry must contain a boundary object")
    _polygon(boundary.get("geometry"))
    if boundary.get("source") not in {"nominatim", "osm"}:
        raise ValueError("Verified boundary source must be nominatim or osm")
    source_url = urlparse(_required_text(boundary, "source_url"))
    hostname = (source_url.hostname or "").lower()
    if source_url.scheme != "https" or not (hostname == "openstreetmap.org" or
                                             hostname.endswith(".openstreetmap.org")):
        raise ValueError("Boundary provenance must link to an HTTPS OpenStreetMap source")
    timestamp = _required_text(boundary, "retrieved_utc")
    try:
        parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("retrieved_utc must be an ISO8601 timestamp") from exc
    if parsed_time.utcoffset() is None or parsed_time.utcoffset().total_seconds() != 0:
        raise ValueError("retrieved_utc must include a UTC timezone")
    if boundary.get("osm_type") not in {"relation", "way"}:
        raise ValueError("City boundary must identify an OSM relation or way")
    if isinstance(boundary.get("osm_id"), bool) or not re.fullmatch(r"[1-9][0-9]*", str(boundary.get("osm_id", ""))):
        raise ValueError("City boundary must include a positive OSM identifier")
    _required_text(boundary, "display_name")
    address = boundary.get("address")
    extratags = boundary.get("extratags") or {}
    if not isinstance(address, dict) or not isinstance(extratags, dict):
        raise ValueError("Boundary address and extratags must be objects")
    # Inspect the object's own classification. address.state is a parent, not
    # the object's type. US independent cities may legitimately have level 6.
    own_types = {_normalized(boundary.get(k)) for k in ("addresstype", "type", "place")}
    own_types.update(_normalized(extratags.get(k)) for k in ("place", "border_type", "designation"))
    if own_types & BROAD_TYPES:
        raise ValueError("A state/country/region object cannot replace a city boundary")
    if not own_types & CITY_TYPES:
        raise ValueError("Boundary object is not independently classified as a city/municipality")
    if _normalized(boundary.get("category")) not in {"boundary", "place"}:
        raise ValueError("City boundary category must be boundary or place")
    if _normalized(address.get("country_code")) != _normalized(_required_text(entry, "expected_country_code")):
        raise ValueError("City boundary country does not match the expected country")
    if _normalized(address.get("state")) != _normalized(_required_text(entry, "expected_state")):
        raise ValueError("City boundary state does not match the expected state")
    expected_name = _entity_name(_required_text(entry, "expected_name"))
    names = {_entity_name(boundary.get("name")), _entity_name(extratags.get("name:en")),
             _entity_name(extratags.get("official_name"))}
    if expected_name not in names:
        raise ValueError("City boundary object's own name does not match the expected city")


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Boundary manifest must use schema {SCHEMA_VERSION}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Boundary manifest entries must be a nonempty list")
    seen = set()
    for entry in entries:
        _validate_entry(entry)
        if entry["case_id"] in seen:
            raise ValueError(f"Duplicate case in boundary manifest: {entry['case_id']}")
        seen.add(entry["case_id"])
        entry.pop("expected_geometry_sha256", None)
    return resolved, manifest


def boundary_override_configuration(path: str | Path) -> dict[str, Any]:
    resolved, manifest = _load_manifest(path)
    return {"policy": POLICY_VERSION, "manifest_path": str(resolved),
            "entries": manifest["entries"]}


def apply_candidate_boundary_overrides(cases: list[dict[str, Any]], path: str | Path) -> list[dict[str, Any]]:
    resolved, manifest = _load_manifest(path)
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("Candidate cases must be a list of objects")
    index: dict[str, list[int]] = {}
    for position, case in enumerate(cases):
        index.setdefault(str(case.get("tweets_id")), []).append(position)
    result = copy.deepcopy(cases)
    for entry in manifest["entries"]:
        case_id = entry["case_id"]
        positions = index.get(case_id, [])
        if len(positions) != 1:
            raise ValueError(f"Boundary override case {case_id} must occur exactly once; got {len(positions)}")
        case = result[positions[0]]
        original_area = case.get("candidate_area")
        if not isinstance(original_area, dict) or original_area.get("source") != "geonames" or original_area.get("area_level") != "geonames_bbox":
            raise ValueError(f"Case {case_id} is not an original GeoNames bbox candidate")
        if str(original_area.get("geonameId")) != str(entry["expected_geoname_id"]):
            raise ValueError(f"Case {case_id} GeoNames identifier does not match the verified input")
        old_geometry = _polygon(original_area.get("geojson"), allow_bbox=True)
        if not old_geometry.equals(old_geometry.envelope):
            raise ValueError(f"Case {case_id} original GeoNames geometry is not a bbox")
        boundary = entry["boundary"]
        geometry = _polygon(boundary["geometry"])
        if not geometry.intersects(old_geometry):
            raise ValueError(f"Case {case_id} verified boundary does not intersect its original bbox")
        address = boundary["address"]
        replacement = copy.deepcopy(original_area)
        replacement.pop("geonameId", None)
        replacement.update({"name": boundary.get("name") or entry["expected_name"],
                            "display_name": boundary["display_name"], "source": boundary["source"],
                            "osm_type": boundary["osm_type"], "osm_id": boundary["osm_id"],
                            "admin_country": address.get("country"), "admin1": address.get("state"),
                            "admin2": address.get("county") or original_area.get("admin2"),
                            "geometry_type": geometry.geom_type, "geojson_available": True,
                            "geojson": copy.deepcopy(boundary["geometry"]), "area_level": "verified_city_boundary",
                            "boundary_object_type": boundary.get("addresstype"),
                            "boundingbox": {"west": geometry.bounds[0], "south": geometry.bounds[1],
                                            "east": geometry.bounds[2], "north": geometry.bounds[3]}})
        case["candidate_area"] = replacement
        case["candidate_boundary_resolution"] = {
            "policy": POLICY_VERSION, "manifest_path": str(resolved),
            "original_candidate_area": copy.deepcopy(original_area),
            "verified_entity": copy.deepcopy(boundary),
            "used_ground_truth_for_selection": False,
        }
    return result
