"""Resolve and classify geographic features mentioned in post text."""


import json
import re
from typing import Any, Mapping, Optional, Tuple

from shapely.geometry import shape

from landslide_localization import spatial as promap


def _valid_or_repaired_geometry(geom: Any) -> Optional[Any]:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if getattr(geom, "is_valid", False):
        return geom
    try:
        from shapely import make_valid

        repaired = make_valid(geom)
    except (ImportError, AttributeError):
        try:
            repaired = geom.buffer(0)
        except Exception:
            return None
    if repaired is None or repaired.is_empty or not repaired.is_valid:
        return None
    return repaired


def _geometry_from_geojson(geom_json: Any) -> Optional[Any]:
    if not isinstance(geom_json, dict):
        return None
    try:
        return _valid_or_repaired_geometry(shape(geom_json))
    except Exception:
        return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(_safe_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def classify_original_feature_type(evidence: Mapping[str, Any]) -> Tuple[str, str]:
    """Preserve an explicit type, otherwise refine the workflow's keyword rules.

    The validated-reference adapter's ``anchor_kind`` uses road/route/highway
    and river/creek/stream/water keywords.  The ordered refinements here retain
    scientifically useful original types while preserving those broad groups.
    Bridge/dam/trail landmarks intentionally remain ``other``.
    """
    explicit = evidence.get("original_feature_type") or evidence.get("feature_type")
    if explicit:
        original = re.sub(r"\s+", "_", str(explicit).strip().casefold())
    else:
        # Classify the mentioned token first.  Administrative words in a long
        # Nominatim display name (e.g. "Park County") must not turn a river or
        # road token into a park.  The resolved name is a fallback only.
        token_text = " ".join(
            _safe_text(evidence.get(key))
            for key in ("token", "normalized_token", "route_type", "highway", "waterway", "natural")
        ).casefold()
        resolved_text = " ".join(
            _safe_text(evidence.get(key))
            for key in ("resolved_name", "category", "feature_class")
        ).casefold()
        # Road terms precede landform terms so "Canyon Road" remains a road;
        # bridge/dam precede water terms so "Silver Creek Bridge" is a bridge.
        ordered = (
            ("interstate", r"\b(interstate|i[- ]?\d+)\b"),
            ("highway", r"\b(highway|hwy|freeway|expressway|us[- ]?\d+|sr[- ]?\d+)\b"),
            ("route", r"\b(route|state route|county route)\b"),
            ("road", r"\b(road|rd\.?|county road)\b"),
            ("street", r"\b(street|st\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?)\b"),
            ("bridge", r"\bbridge\b"),
            ("dam", r"\bdam\b"),
            ("river", r"\briver\b"),
            ("creek", r"\b(creek|brook)\b"),
            ("stream", r"\b(stream|ditch|waterway)\b"),
            ("lake", r"\b(lake|reservoir|pond)\b"),
            ("waterfall", r"\b(waterfall|falls)\b"),
            ("trail", r"\btrail\b"),
            ("tunnel", r"\btunnel\b"),
            ("canyon", r"\b(canyon|gorge)\b"),
            ("mountain", r"\b(mountain|mount|mt\.?|peak|ridge)\b"),
            ("park", r"\b(park|forest|preserve)\b"),
            ("landmark", r"\b(landmark|monument|cemetery|school|church|mine)\b"),
        )
        original = next((name for name, pattern in ordered if re.search(pattern, token_text)), "local_feature")
        if original == "local_feature":
            original = next((name for name, pattern in ordered if re.search(pattern, resolved_text)), "local_feature")

    if original in {"bridge", "dam", "trail", "tunnel", "canyon", "mountain", "park", "landmark", "local_feature"}:
        category = "other"
    elif original in {"river", "creek", "stream", "lake", "waterfall", "water", "waterway", "ditch", "brook", "reservoir", "pond"}:
        category = "river"
    elif original in {"road", "route", "highway", "street", "interstate", "freeway", "expressway"}:
        category = "road"
    else:
        category = "other"
    return original, category


def _resolve_text_geometry(evidence: Mapping[str, Any]) -> Optional[Any]:
    geom = _geometry_from_geojson(evidence.get("geojson"))
    if geom is not None:
        return geom
    resolved_name = evidence.get("resolved_name")
    if not resolved_name:
        return None
    try:
        geom_json, _ = promap.geocode_nominatim(str(resolved_name))
    except Exception:
        return None
    return _geometry_from_geojson(geom_json)


def _feature_name(evidence: Mapping[str, Any], default: str) -> str:
    return _safe_text(
        evidence.get("token") or evidence.get("normalized_token")
        or evidence.get("resolved_name") or evidence.get("name") or default
    )
