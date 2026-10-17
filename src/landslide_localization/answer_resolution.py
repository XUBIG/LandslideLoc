"""Select an administrative polygon from saved geocoder responses."""

from __future__ import annotations
from typing import Any, Iterable
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


POLYGON_TYPES = {"Polygon", "MultiPolygon"}


US_STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
}


COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u s a": "united states",
    "united states of america": "united states",
}


def normalized_answer(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def answer_key(value: Any) -> str:
    return normalized_answer(value).casefold()


def match_text(value: Any) -> str:
    """Normalize a place component for exact, punctuation-insensitive matching."""
    text = "".join(
        character.casefold() if character.isalnum() else " "
        for character in normalized_answer(value)
    )
    text = " ".join(text.split())
    return COUNTRY_ALIASES.get(text, text)


def answer_components(query: str) -> list[str]:
    return [match_text(part) for part in query.split(",") if match_text(part)]


def expected_addresstype(query: str) -> str | None:
    """Infer only unambiguous administrative levels from the answer text."""
    parts = answer_components(query)
    if not parts:
        return None
    first = parts[0]
    if first in US_STATE_NAMES and len(parts) >= 2 and parts[-1] == "united states":
        return "state"
    if first.endswith(" county"):
        return "county"
    if first.endswith(" parish"):
        return "county"
    if first in {"united states", "canada", "mexico", "united kingdom"}:
        return "country"
    return None


def candidate_match_score(
    result: dict[str, Any], query: str, rank: int
) -> tuple[int, ...] | None:
    """Score a polygon using answer text only; return None for a name mismatch."""
    components = answer_components(query)
    if not components:
        return None
    address = result.get("address") if isinstance(result.get("address"), dict) else {}
    candidate_values = {
        match_text(value)
        for value in [result.get("name"), *address.values()]
        if normalized_answer(value)
    }
    primary_exact = components[0] in candidate_values
    if not primary_exact:
        return None

    expected = expected_addresstype(query)
    addresstype = match_text(result.get("addresstype"))
    expected_match = int(expected is not None and addresstype == expected)
    context_matches = sum(component in candidate_values for component in components[1:])
    category = result.get("category") or result.get("class")
    is_admin = int(category == "boundary" and result.get("type") == "administrative")
    # Larger tuple is better. Rank is the final tie breaker only.
    return expected_match, context_matches, is_admin, -rank


def polygon_geometry(result: dict[str, Any]) -> BaseGeometry | None:
    """Return a usable polygon geometry, or None for points/lines/bad GeoJSON."""
    geojson = result.get("geojson")
    if not isinstance(geojson, dict) or geojson.get("type") not in POLYGON_TYPES:
        return None
    try:
        geometry = shape(geojson)
        if geometry.is_empty:
            return None
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty or geometry.geom_type not in POLYGON_TYPES:
            return None
        return geometry
    except (TypeError, ValueError):
        return None


def choose_ranked_polygon(
    results: Iterable[dict[str, Any]],
    query: str | None = None,
) -> tuple[int, dict[str, Any], BaseGeometry] | None:
    """Select an answer-matching polygon without looking at the evaluated point.

    When a query is supplied, the primary answer component must exactly match a
    candidate name/address value. Context components and unambiguous requested
    administrative levels then refine Nominatim's own rank order.
    """
    candidates: list[tuple[tuple[int, ...], int, dict[str, Any], BaseGeometry]] = []
    for rank, result in enumerate(results, start=1):
        if not isinstance(result, dict) or result.get("_error"):
            continue
        geometry = polygon_geometry(result)
        if geometry is None:
            continue
        if query is None:
            return rank, result, geometry
        score = candidate_match_score(result, query, rank)
        if score is not None:
            candidates.append((score, rank, result, geometry))
    if not candidates:
        return None
    _, rank, result, geometry = max(candidates, key=lambda item: item[0])
    return rank, result, geometry


def result_metadata(
    result: dict[str, Any], rank: int, geometry: BaseGeometry
) -> dict[str, Any]:
    category = result.get("category") or result.get("class")
    return {
        "rank": rank,
        "display_name": result.get("display_name"),
        "name": result.get("name"),
        "category": category,
        "type": result.get("type"),
        "addresstype": result.get("addresstype"),
        "osm_type": result.get("osm_type"),
        "osm_id": result.get("osm_id"),
        "place_id": result.get("place_id"),
        # Direct preprocessing derives administrative hints from its own match.
        "address": dict(result.get("address") or {}),
        "geometry_type": geometry.geom_type,
        "is_administrative": bool(
            category == "boundary" and result.get("type") == "administrative"
        ),
    }
