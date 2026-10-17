"""Stage-I containment from the computed Full candidate polygons."""
import json

TIERS = ("fine_area", "county_area", "state_area", "unknown")
TIER_LABELS = {"fine_area": "fine", "county_area": "county", "state_area": "state", "unknown": "other"}

def _id(value):
    # Tweet IDs exceed the exact-integer range of binary floating point.
    if isinstance(value, bool) or isinstance(value, float) or value is None:
        raise ValueError(f"Case ID must be an exact string/integer, not {value!r}")
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"Invalid case ID: {value!r}")
    return str(value)


def _tier(value):
    aliases = {"fine": "fine_area", "county": "county_area", "state": "state_area"}
    value = aliases.get(value, value) if isinstance(value, str) else value
    return value if value in TIERS else "unknown"


def full_stage1_records(raw_records, full_by_id, full_cache, promap):
    """Resolve Full Stage I from final candidate polygons and original GT only.

    Missing caches are explicitly unresolved, even if Stage-II metrics exist.
    """
    from shapely.geometry import Point
    from landslide_localization.preprocessing import get_gt
    from landslide_localization.soft_utils import parse_candidate_geojson

    result = []
    for raw in raw_records:
        case_id = _id(raw.get("tweets_id"))
        item = full_by_id[case_id]
        tier = _tier(item.get("candidate_area_tier") or (item.get("candidate_area") or {}).get("tier"))
        row = {
            "model": "full", "tweets_id": case_id, "tier": TIER_LABELS[tier],
            "status": "failed", "evaluable": False, "in_region": False,
            "failure_reason": "", "cache_source": "frozen_full_candidate_cache",
        }
        path = full_cache.path("candidate_regions", case_id, ".geojson")
        if not path.is_file():
            row["failure_reason"] = "frozen_candidate_region_missing"
            result.append(row)
            continue
        try:
            lon, lat = get_gt(dict(raw))
            if lon is None or lat is None:
                raise ValueError("invalid_reference_coordinates")
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            _, _, _, polygon = parse_candidate_geojson(data, promap)
            point = promap.reproject_geom(Point(lon, lat), "EPSG:4326", promap.WORK_CRS)
            inside = bool(polygon.covers(point))
            row.update(status="inside" if inside else "outside", evaluable=True, in_region=inside)
        except Exception as exc:
            row["failure_reason"] = f"frozen_candidate_parse_error:{type(exc).__name__}:{exc}"
        result.append(row)
    return result
