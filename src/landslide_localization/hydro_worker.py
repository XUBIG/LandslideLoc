"""Road/water extraction in a subprocess with a component time limit."""
from __future__ import annotations

import json
import os
import sys
from itertools import islice
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely import wkb
from shapely.geometry import shape
from shapely.ops import unary_union


from landslide_localization.runtime_paths import REPO_ROOT as ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landslide_localization.hydro_road_utils import (  # noqa: E402
    extract_hydro_network_from_pbf,
    extract_road_network,
    extract_road_network_from_pbf,
    generate_weighted_buffer,
    get_rivers,
    prepare_buffer_unions,
)


def configure_cache(soft_cache: Path) -> None:
    soft_cache.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(soft_cache)
    ox.settings.log_console = False
    ox.settings.requests_timeout = 30
    ox.settings.overpass_settings = "[out:json][timeout:30]{maxsize}"

def configure_cache_only_failure_mode() -> dict[str, int]:
    """Permit validated cache hits but fail immediately on an HTTP cache miss."""
    import osmnx._overpass as ox_overpass

    ox_overpass._get_overpass_pause = lambda *_args, **_kwargs: 0

    state = {"network_cache_misses": 0}

    def unavailable(*_args, **_kwargs):
        state["network_cache_misses"] += 1
        raise RuntimeError("Overpass unavailable and --hydro-cache-only was selected")

    ox_overpass.requests.post = unavailable
    return state


def empty_lines(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs=crs)


def cached_pbf_roads(
    pbf_path: Path,
    raw_cache_path: Path,
    place_gdf: gpd.GeoDataFrame,
    work_crs: str,
) -> tuple[gpd.GeoDataFrame, bool]:
    """Backward-compatible wrapper for cached road extraction."""
    return cached_pbf_lines(pbf_path, raw_cache_path, place_gdf, work_crs, "road")


def cached_pbf_lines(
    pbf_path: Path,
    raw_cache_path: Path,
    place_gdf: gpd.GeoDataFrame,
    work_crs: str,
    component: str,
) -> tuple[gpd.GeoDataFrame, bool]:
    """Load an AOI line clip, scanning its PBF only on a cache miss."""
    missing_path = raw_cache_path.with_suffix(".missing.json")
    if raw_cache_path.is_file():
        geometry = wkb.loads(raw_cache_path.read_bytes())
        return gpd.GeoDataFrame([{"geometry": geometry}], crs=work_crs), True
    if missing_path.is_file():
        return empty_lines(work_crs), True

    if component == "road":
        lines = extract_road_network_from_pbf(str(pbf_path), place_gdf, target_crs=work_crs)
    elif component == "water":
        lines = extract_hydro_network_from_pbf(str(pbf_path), place_gdf, target_crs=work_crs)
    else:
        raise ValueError(f"Unsupported PBF component: {component}")
    raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if lines is None or lines.empty:
        temporary = missing_path.with_name(f".{missing_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps({"reason": f"no_{component}_features_in_aoi"}), encoding="utf-8"
        )
        temporary.replace(missing_path)
        return empty_lines(work_crs), False

    geometry = unary_union(lines.geometry)
    temporary = raw_cache_path.with_name(f".{raw_cache_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(geometry.wkb)
    temporary.replace(raw_cache_path)
    return gpd.GeoDataFrame([{"geometry": geometry}], crs=work_crs), False


def buffer_pbf_roads_chunked(
    roads: gpd.GeoDataFrame,
    place_gdf: gpd.GeoDataFrame,
    work_crs: str,
    distance_m: float,
    chunk_size: int = 1000,
):
    """Backward-compatible wrapper for chunked line buffering."""
    return buffer_lines_chunked(roads, place_gdf, work_crs, distance_m, chunk_size)


def buffer_lines_chunked(
    lines: gpd.GeoDataFrame,
    place_gdf: gpd.GeoDataFrame,
    work_crs: str,
    distance_m: float,
    chunk_size: int = 1000,
):
    """Buffer large cached linework without one huge GEOS allocation."""
    if lines is None or lines.empty:
        return None
    aoi = place_gdf.to_crs(work_crs).geometry.union_all()
    source = lines.to_crs(work_crs).geometry.union_all()
    components = iter(source.geoms) if hasattr(source, "geoms") else iter((source,))
    levels: list[object | None] = []

    while True:
        chunk = list(islice(components, chunk_size))
        if not chunk:
            break
        part = unary_union([geometry.buffer(distance_m) for geometry in chunk]).intersection(aoi)
        if part.is_empty:
            continue
        level = 0
        while level < len(levels) and levels[level] is not None:
            part = unary_union([levels[level], part])
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(part)
        else:
            levels[level] = part

    remaining = [geometry for geometry in levels if geometry is not None]
    return unary_union(remaining).intersection(aoi) if remaining else None


def first_available_pbf(
    entries: list[dict],
    place_gdf: gpd.GeoDataFrame,
    work_crs: str,
    component: str,
) -> tuple[gpd.GeoDataFrame, dict, list[dict]]:
    """Try specialized state, complete state, then national PBF entries."""
    attempts: list[dict] = []
    for entry in entries:
        source_path = Path(entry["path"])
        cache_path = Path(entry["raw_cache"])
        attempt = {"label": entry.get("label"), "path": str(source_path)}
        try:
            lines, cache_hit = cached_pbf_lines(
                source_path, cache_path, place_gdf, work_crs, component
            )
            attempt["raw_cache_hit"] = cache_hit
            attempt["status"] = "success" if not lines.empty else "empty"
            attempts.append(attempt)
            if not lines.empty:
                return lines, {**entry, "raw_cache_hit": cache_hit}, attempts
        except Exception as exc:
            attempt["status"] = "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
    return empty_lines(work_crs), {}, attempts


def main() -> None:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    work_crs = str(payload.get("work_crs") or "EPSG:5070")
    configure_cache(
        Path(payload["soft_osmnx_cache"]),
    )
    cache_only_state = configure_cache_only_failure_mode() if payload.get("hydro_cache_only") else None

    aoi = shape(payload["geometry"])
    place_gdf = gpd.GeoDataFrame([{"geometry": aoi}], crs="EPSG:4326")
    road_gdf = empty_lines(work_crs)
    water_gdf = empty_lines(work_crs)
    water_source = None
    road_source = None
    road_raw_cache_hit = None
    water_raw_cache_hit = None
    road_raw_cache = None
    water_raw_cache = None
    road_attempts: list[dict] = []
    water_attempts: list[dict] = []
    errors: dict[str, str] = {}

    if payload.get("need_road"):
        try:
            road_entries = list(payload.get("road_pbf_candidates") or [])
            if not road_entries and payload.get("road_pbf") and payload.get("road_raw_cache"):
                road_entries = [{
                    "label": "local_pbf",
                    "path": payload["road_pbf"],
                    "raw_cache": payload["road_raw_cache"],
                }]
            if road_entries:
                road_gdf, selected, road_attempts = first_available_pbf(
                    road_entries, place_gdf, work_crs, "road"
                )
                road_source = selected.get("label")
                road_raw_cache = selected.get("raw_cache")
                road_raw_cache_hit = selected.get("raw_cache_hit")
                if road_gdf.empty:
                    errors["road"] = "local_pbf_candidates_exhausted"
            else:
                road_gdf = extract_road_network(place_gdf, target_crs=work_crs)
                road_source = "overpass"
        except Exception as exc:  # worker isolation is the reference failure boundary
            errors["road"] = f"{type(exc).__name__}: {exc}"

    if payload.get("need_water"):
        try:
            water_entries = list(payload.get("water_pbf_candidates") or [])
            if water_entries:
                water_gdf, selected, water_attempts = first_available_pbf(
                    water_entries, place_gdf, work_crs, "water"
                )
                water_source = selected.get("label")
                water_raw_cache = selected.get("raw_cache")
                water_raw_cache_hit = selected.get("raw_cache_hit")
                if water_gdf.empty:
                    errors["water"] = "local_pbf_candidates_exhausted"
            else:
                gdb = payload.get("gdb_path")
                water_gdf, water_source = get_rivers(
                    gdb_path=gdb if gdb and Path(gdb).exists() else None,
                    polygon_gdf=place_gdf,
                    target_crs=work_crs,
                    prefer="osm",
                    bbox_expand=1.3,
                    osm_tags={"waterway": ["river", "stream", "ditch"]},
                    return_source=True,
                )
        except Exception as exc:
            errors["water"] = f"{type(exc).__name__}: {exc}"

    if cache_only_state and cache_only_state["network_cache_misses"]:
        component = "road" if payload.get("need_road") else "water"
        errors[component] = (
            "overpass_cache_miss: external Overpass unavailable; "
            "OSMnx cache contained no matching response"
        )

    road_union = buffer_lines_chunked(
        road_gdf, place_gdf, work_crs, float(payload["road_buffer_m"])
    )
    water_union = buffer_lines_chunked(
        water_gdf, place_gdf, work_crs, float(payload["water_buffer_m"])
    )
    output_path.write_text(
        json.dumps(
            {
                "road_union_wkb": road_union.wkb_hex if road_union is not None and not road_union.is_empty else None,
                "water_union_wkb": water_union.wkb_hex if water_union is not None and not water_union.is_empty else None,
                "water_source": water_source,
                "road_source": road_source,
                "road_raw_cache": road_raw_cache,
                "road_raw_cache_hit": road_raw_cache_hit,
                "water_raw_cache": water_raw_cache,
                "water_raw_cache_hit": water_raw_cache_hit,
                "road_pbf_attempts": road_attempts,
                "water_pbf_attempts": water_attempts,
                "errors": errors,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
