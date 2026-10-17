"""Geometry and name helpers used by offline feature reconstruction."""

from __future__ import annotations
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union


ROAD_SUFFIXES = {
    "blvd": "boulevard",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "hwy": "highway",
    "dr": "drive",
    "ln": "lane",
    "pkwy": "parkway",
}


STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def line_only(geometry):
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    if geometry.geom_type in {"LineString", "MultiLineString"}:
        return geometry
    if hasattr(geometry, "geoms"):
        lines = [
            part
            for part in geometry.geoms
            if part.geom_type in {"LineString", "MultiLineString"}
        ]
        return unary_union(lines) if lines else GeometryCollection()
    return GeometryCollection()
