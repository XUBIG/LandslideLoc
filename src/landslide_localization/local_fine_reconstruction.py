"""Offline reconstruction of named roads and water features from local OSM data."""

from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import unicodedata
import geopandas as gpd
import pandas as pd
import pyogrio
from pyproj import Transformer
from shapely import make_valid, from_wkb
from shapely.geometry import GeometryCollection, box, shape
from shapely.ops import linemerge, transform, unary_union
from shapely.strtree import STRtree
from landslide_localization import geometry_utils as geometry_helpers
from landslide_localization import text_features as fine_ref


NAME_FIELDS = (
    "name",
    "name:en",
    "alt_name",
    "official_name",
    "short_name",
    "loc_name",
    "gnis:feature_name",
    "gnis:name",
    "old_name",
)


SUFFIXES = {
    **geometry_helpers.ROAD_SUFFIXES,
    "ct": "court",
    "rte": "route",
    "trl": "trail",
    "cir": "circle",
    "ter": "terrace",
}


DIRECTIONS = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}


HYDRO_SUFFIXES = {"cr": "creek", "ck": "creek", "riv": "river", "br": "branch"}


GENERIC = (
    set(SUFFIXES)
    | set(SUFFIXES.values())
    | set(DIRECTIONS.values())
    | {
        "river",
        "creek",
        "stream",
        "fork",
        "branch",
        "run",
        "brook",
        "lake",
        "pond",
        "reservoir",
        "state",
        "county",
        "us",
        "sr",
        "the",
        "of",
        "and",
        "water",
        "canyon",
    }
)


TO_METRIC = Transformer.from_crs(4326, 5070, always_xy=True)


TO_WGS = Transformer.from_crs(5070, 4326, always_xy=True)


LOG = logging.getLogger("local_fine")


CACHE_VERSION = 4


def project(geometry, inverse=False):
    return transform((TO_WGS if inverse else TO_METRIC).transform, geometry)


@lru_cache(maxsize=100000)
def plain_name(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"\b([a-z])\.\s*([a-z])\.", lambda m: m[1] + m[2], value)
    value = re.sub(r"\bu\s*\.\s*s\s*\.?", "us", value)
    value = re.sub(r"\bs\s*\.\s*r\s*\.?", "sr", value)
    value = re.sub(r"['’]", "", value)
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


@lru_cache(maxsize=150000)
def normalize_road_name(value, state=""):
    text = plain_name(value)
    text = re.sub(r"^(hwy|rte) (?=\d)", lambda m: SUFFIXES[m[1]] + " ", text)
    for full, abbreviation in geometry_helpers.STATE_ABBR.items():
        prefix = full.casefold() + " "
        if text.startswith(prefix) and re.fullmatch(
            r"(?:(?:state )?(?:highway|hwy|route|rte) )?\d+[a-z]?", text[len(prefix) :]
        ):
            text = (
                abbreviation.lower()
                + " "
                + re.sub(
                    r"^(?:state )?(?:(?:highway|hwy|route|rte) )?",
                    "",
                    text[len(prefix) :],
                )
            )
            break
    text = re.sub(r"^(?:nb|sb|eb|wb) (?=(?:i|us|[a-z]{2})[ -]?\d)", "", text)
    # Context disambiguates SR; a bare number never becomes an interstate/US road.
    route = re.fullmatch(
        r"(us|[a-z]{2}|i|interstate|state)\s*(?:highway |hwy |route |rte )?(\d+[a-z]?)",
        text,
    )
    if route:
        prefix, number = route.groups()
        if prefix in {"sr", "state"}:
            prefix = geometry_helpers.STATE_ABBR.get(state, "SR").lower()
        if prefix == "interstate":
            prefix = "i"
        if prefix in {"us", "sr", "i"} | {
            s.lower() for s in geometry_helpers.STATE_ABBR.values()
        }:
            return f"{prefix} {number}"
    words = text.split()
    # Only suffix tokens: St. Mary Road must not become Street Mary Road.
    if words and words[-1] in DIRECTIONS:
        words[-1] = DIRECTIONS[words[-1]]
    suffix_index = (
        -2 if words and words[-1] in DIRECTIONS.values() and len(words) > 1 else -1
    )
    if words:
        words[suffix_index] = SUFFIXES.get(words[suffix_index], words[suffix_index])
        if words[0] in DIRECTIONS:
            words[0] = DIRECTIONS[words[0]]
    return " ".join(words)


@lru_cache(maxsize=50000)
def normalize_hydro_name(value, state=""):
    words = plain_name(value).split()
    if words:
        words[-1] = HYDRO_SUFFIXES.get(words[-1], words[-1])
    # River, creek, fork, branch, run and brook remain different identities.
    return " ".join(words)


def normalize(value, category, state=""):
    return (normalize_road_name if category == "ROAD" else normalize_hydro_name)(
        value, state
    )


@lru_cache(maxsize=30000)
def normalize_reference(value, state=""):
    text = normalize_road_name(value, state)
    # County-route prefix is inconsistently present on successive ways. Keep
    # this equivalence confined to ref tags, never remove it from ordinary names.
    return re.sub(r"^(?:cr|county (?:road|route|highway)) (?=[a-z]?\d)", "", text)


def directional_continuation(row, feature):
    query = normalize_road_name(row["original_fine_name"], row["state"])
    if query.split()[0] in DIRECTIONS.values():
        return False  # An explicitly requested north road is not a south road.
    for field, value in extract_name_candidates(feature.tags):
        if field in {"ref", "nat_ref", "loc_ref"}:
            continue
        words = normalize_road_name(value, row["state"] or feature.state).split()
        if words and words[0] in DIRECTIONS.values() and " ".join(words[1:]) == query:
            return True
        if words and words[-1] in DIRECTIONS.values() and " ".join(words[:-1]) == query:
            return True
    return False


def split_values(value):
    return [v.strip() for v in str(value or "").split(";") if v.strip()]


def parse_other_tags(value):
    """GDAL OSM hstore syntax, including escaped quotes and backslashes."""
    pairs = re.findall(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"', str(value or ""))
    return {
        re.sub(r'\\(["\\])', r"\1", k): re.sub(r'\\(["\\])', r"\1", v) for k, v in pairs
    }


def extract_name_candidates(tags):
    for field in NAME_FIELDS:
        for value in split_values(tags.get(field)):
            yield field, value
    if tags.get("tiger:name_base") and tags.get("tiger:name_type"):
        yield "tiger:assembled_name", " ".join(
            str(tags.get(k, ""))
            for k in (
                "tiger:name_direction_prefix",
                "tiger:name_base",
                "tiger:name_type",
                "tiger:name_direction_suffix",
            )
        ).strip()
    for field in ("ref", "nat_ref", "loc_ref"):
        for value in split_values(tags.get(field)):
            yield field, value


def safe_shape(value):
    try:
        g = shape(value)
        if not g.is_valid:
            g = make_valid(g)
        return g if not g.is_empty else None
    except (ValueError, TypeError, AttributeError):
        return None


def classify(evidence):
    # A fallback match can contain a full geocoder address; its street/county
    # tokens can misclassify casinos, towns, and mountains as roads or hydro.
    typed_evidence = dict(evidence)
    typed_evidence["resolved_name"] = str(evidence.get("resolved_name") or "").split(
        ","
    )[0]
    original, category = fine_ref.classify_original_feature_type(typed_evidence)
    token = plain_name(fine_ref._feature_name(evidence, ""))
    # Retain explicit types and project categories; fill omissions in its keyword set.
    if not evidence.get("feature_type") and not evidence.get("original_feature_type"):
        terminal_other = re.search(
            r"\b(bridge|dam|tunnel|trail|trailhead|boat launch|ranch|canyon|gorge|city|town)$",
            token,
        )
        terminal_road = re.search(
            r"\b(road|rd|street|st|avenue|ave|boulevard|blvd|highway|hwy|drive|dr|lane|ln|court|ct|parkway|pkwy|circle|cir|terrace|ter)(?: (?:n|s|e|w|ne|nw|se|sw|north|south|east|west))?$",
            token,
        )
        if re.fullmatch(r"(?:us|i|[a-z]{2}) \d+[a-z]?", normalize_road_name(token)):
            original, category = original if category == "road" else "route", "road"
        elif terminal_other:
            kind = terminal_other[1]
            original, category = {
                "trailhead": "trail",
                "boat launch": "landmark",
                "ranch": "landmark",
                "gorge": "canyon",
                "city": "local_feature",
                "town": "local_feature",
            }.get(kind, kind), "other"
        elif terminal_road:
            original, category = original if category == "road" else "road", "road"
        elif category == "other" and re.search(
            r"\b(canal|drain|ditch|stream|brook|creek|cr|river|reservoir)\b", token
        ):
            original, category = "waterway", "river"
        elif category == "other" and (
            re.fullmatch(r"(?:us|i|[a-z]{2}) \d+[a-z]?", normalize_road_name(token))
            or re.search(
                r"\b(drive|dr|court|ct|parkway|pkwy|circle|cir|terrace|ter)$", token
            )
        ):
            original, category = "road", "road"
    return original, {"road": "ROAD", "river": "HYDRO", "other": "OTHER"}[category]


@dataclass
class LocalFeature:
    osm_id: str
    tags: dict
    geometry: object
    source: str
    state: str
    layer: str


class LocalOSMStore:
    """Bounded memory cache; persistent state/query caches avoid repeat PBF scans.

    The project already uses GDAL/pyogrio's OSM layers. Its convenience loaders
    clip early and discard tags, so this adapter uses the same reader, retains
    other_tags, and postpones clipping. SQL keeps all refs plus inventory name
    tokens (including aliases); matching itself is exact/conservative, not SQL LIKE.
    """

    def __init__(self, args, inventory, cases, state_polygons):
        self.args, self.cases = args, cases
        self.memory = OrderedDict()
        self.files = {
            p.name.removesuffix(".osm.pbf").replace("_", " "): p
            for p in sorted(args.osm_dir.glob("*.osm.pbf"))
        }
        self.bounds = {
            state: box(*pyogrio.read_info(path, layer="lines")["total_bounds"])
            for state, path in self.files.items()
        }
        self.polygons = state_polygons
        self._state_cache = {}
        self.state_requests = defaultdict(list)
        self.cache_dir = Path(
            getattr(args, "local_cache_dir", None) or args.output / "cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for row in inventory:
            if row["category"] != "OTHER":
                row["search_states"] = self.states_for(row)
                for state in row["search_states"]:
                    self.state_requests[(state, row["category"])].append(row)

    def states_for(self, row):
        if row["case_id"] in self._state_cache:
            return self._state_cache[row["case_id"]]
        region = self.cases[row["case_id"]]["region"]
        if region is None:
            return []
        primary = row["state"]
        # A country-sized candidate with no state cannot safely identify a local name.
        if primary not in self.files and (
            row["candidate_area_tier"] == "broad_area" or project(region).area > 2e12
        ):
            return []
        search = project(project(region).buffer(self.args.search_buffer_m), True)
        states = []
        for state, bounds in self.bounds.items():
            geometry = self.polygons.get(state, bounds)
            if geometry.intersects(search):
                states.append(state)
        if primary in self.files and primary not in states:
            states.append(primary)
        result = sorted(states)
        self._state_cache[row["case_id"]] = result
        return result

    def _tokens(self, state, category):
        requests = self.state_requests.get((state, category), [])
        tokens = set()
        for row in requests:
            for name in [
                row["original_fine_name"],
                row.get("resolved_name", "").split(",")[0],
            ]:
                core = [
                    t
                    for t in plain_name(name).split()
                    if t not in GENERIC and (len(t) >= 3 or t.isdigit())
                ]
                tokens.update(core)
        return sorted(tokens)

    def _where(self, state, category, layer):
        # A short GDAL predicate is much faster than hundreds of LIKE clauses.
        # Arrow batches are then filtered on name/ref tag VALUES before decoding WKB.
        identity = "(name IS NOT NULL OR other_tags LIKE '%name\"=>%' OR other_tags LIKE '%ref\"=>%')"
        if layer == "lines":
            return (
                "highway IS NOT NULL AND "
                if category == "ROAD"
                else "waterway IS NOT NULL AND "
            ) + identity
        return identity

    def load_local_osm_for_state(self, state, category, raw=False):
        key = state, category, raw
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        source = self.files.get(state)
        if source is None:
            return []
        if not raw:
            directory = getattr(
                self.args,
                "road_state_dir" if category == "ROAD" else "hydro_state_dir",
                None,
            )
            source = (
                Path(directory)
                if directory
                else self.args.osm_dir
                / ("road_only" if category == "ROAD" else "hydro_only")
            ) / source.name
        if not source.exists():
            return []
        layers = ["lines"]
        if category == "HYDRO":
            layers += ["multilinestrings", "multipolygons"]
            if raw and any(
                r["fine_type"] == "waterfall" for r in self.state_requests[key[:2]]
            ):
                layers += ["points"]
        elif raw:
            layers += ["multilinestrings"]
        queries = {layer: self._where(state, category, layer) for layer in layers}
        tokens = self._tokens(state, category)
        token_pattern = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(t) for t in tokens) + r")(?!\w)",
                re.I,
            )
            if tokens
            else None
        )
        name_tag_pattern = re.compile(
            r'"(?:'
            + "|".join(re.escape(f) for f in NAME_FIELDS)
            + r'|tiger:name_base)"=>"((?:[^"\\]|\\.)*)"'
        )
        stat = source.stat()
        signature = hashlib.sha256(
            json.dumps(
                [
                    CACHE_VERSION,
                    str(source),
                    stat.st_size,
                    stat.st_mtime_ns,
                    queries,
                    tokens,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()[:20]
        cache = (
            self.cache_dir
            / f"{state.replace(' ', '_')}_{category}_{'raw' if raw else 'processed'}_{signature}.parquet"
        )
        if cache.exists():
            frame = gpd.read_parquet(cache)
            LOG.info(
                "Cache %s %s %s: %d rows",
                state,
                category,
                "raw" if raw else "processed",
                len(frame),
            )
        else:
            frames = []
            for layer in layers:
                LOG.info(
                    "Read %s / %s (inventory-filtered, complete way geometry)",
                    source,
                    layer,
                )
                retained = []
                with pyogrio.open_arrow(
                    source,
                    layer=layer,
                    where=queries[layer],
                    use_pyarrow=True,
                    batch_size=16384,
                ) as (meta, reader):
                    geom_col = meta["geometry_name"] or "wkb_geometry"
                    for batch in reader:
                        for data in batch.to_pylist():
                            other = str(data.get("other_tags") or "")
                            names = [
                                str(data.get("name") or "")
                            ] + name_tag_pattern.findall(other)
                            has_ref = category == "ROAD" and any(
                                f'"{field}"=>' in other
                                for field in ("ref", "nat_ref", "loc_ref")
                            )
                            if (
                                not has_ref
                                and token_pattern is not None
                                and not any(token_pattern.search(n) for n in names)
                            ):
                                continue
                            wkb = data.pop(geom_col)
                            if wkb is None:
                                continue
                            tags = parse_other_tags(other)
                            tags.update(
                                {
                                    k: str(v)
                                    for k, v in data.items()
                                    if k
                                    not in {
                                        "other_tags",
                                        "osm_id",
                                        "osm_way_id",
                                        "z_order",
                                    }
                                    and v is not None
                                }
                            )
                            road = (
                                bool(tags.get("highway")) or tags.get("route") == "road"
                            )
                            hydro = (
                                bool(tags.get("waterway"))
                                or tags.get("natural") == "water"
                                or bool(tags.get("water"))
                                or tags.get("landuse") == "reservoir"
                                or tags.get("type") == "waterway"
                            )
                            if not (road if category == "ROAD" else hydro):
                                continue
                            geometry = from_wkb(wkb)
                            if geometry.is_empty:
                                continue
                            osm_id = str(
                                data.get("osm_id") or data.get("osm_way_id") or ""
                            )
                            if (
                                category == "ROAD"
                                and tags.get("network")
                                and re.fullmatch(r"\d+[A-Za-z]?", tags.get("ref", ""))
                            ):
                                network = tags["network"].split(":")
                                if len(network) == 2 and network[0] == "US":
                                    tags["relation:original_ref"] = tags["ref"]
                                    tags["ref"] = network[1] + " " + tags["ref"]
                            retained.append(
                                {
                                    "osm_id": osm_id,
                                    "tags_json": json.dumps(tags, ensure_ascii=False),
                                    "layer": layer,
                                    "geometry": (
                                        geometry
                                        if geometry.is_valid
                                        else make_valid(geometry)
                                    ),
                                }
                            )
                LOG.info("  retained %d %s identities", len(retained), layer)
                if retained:
                    frames.append(gpd.GeoDataFrame(retained, crs=4326))
            frame = (
                gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
                if frames
                else gpd.GeoDataFrame(
                    {"osm_id": [], "tags_json": [], "layer": [], "geometry": []},
                    geometry="geometry",
                    crs=4326,
                )
            )
            frame.to_parquet(cache, index=False)
        result = [
            LocalFeature(
                str(r.osm_id),
                json.loads(r.tags_json),
                r.geometry,
                str(source),
                state,
                r.layer,
            )
            for r in frame.itertuples()
        ]
        self.memory[key] = result
        while len(self.memory) > self.args.memory_states:
            self.memory.popitem(last=False)
        return result

    def get(self, row, raw=False):
        found = {}
        for state in row.get("search_states", []):
            for feature in self.load_local_osm_for_state(state, row["category"], raw):
                # OSM identifiers deduplicate overlapping state extracts.
                found.setdefault((feature.layer, feature.osm_id), feature)
        return list(found.values())


def identity_match(row, feature, allow_fuzzy=False):
    query = row["original_fine_name"]
    category, state = row["category"], row["state"] or feature.state
    canonical = normalize(query, category, state)
    matches = []
    for field, value in extract_name_candidates(feature.tags):
        target = normalize(value, category, state)
        is_ref = field in {"ref", "nat_ref", "loc_ref"}
        if is_ref and category == "ROAD":
            target = normalize_reference(value, state)
            canonical = normalize_reference(query, state)
        else:
            canonical = normalize(query, category, state)
        if field == "name" and query.strip().casefold() == value.strip().casefold():
            matches.append((0, "exact_name", field, value, 1.0))
        elif canonical == target:
            rank = 3 if is_ref else 1 if field == "name" else 2
            matches.append(
                (
                    rank,
                    {1: "normalized_exact", 2: "alternative_name", 3: "ref_match"}[
                        rank
                    ],
                    field,
                    value,
                    1.0,
                )
            )
        elif (
            category == "ROAD"
            and is_ref
            and re.fullmatch(r"(?:highway|hwy|route|rte) \d+[a-z]?", canonical)
        ):
            if re.fullmatch(
                r"(?:us|i|[a-z]{2}) " + re.escape(canonical.split()[-1]), target
            ):
                matches.append((4, "route_alias", field, value, 1.0))
        elif allow_fuzzy and not is_ref and len(canonical) >= 9:
            a, b = canonical.split(), target.split()
            # Same suffix and token count, numeric identity and at least one
            # substantial exact token. No arbitrary nearest-name acceptance.
            if (
                len(a) == len(b)
                and a[-1:] == b[-1:]
                and re.findall(r"\d+", canonical) == re.findall(r"\d+", target)
            ):
                shared = set(a) & set(b) - GENERIC
                score = SequenceMatcher(None, canonical, target).ratio()
                if shared and score >= 0.93:
                    matches.append((5, "fuzzy_fallback", field, value, score))
    return min(matches) if matches else None


def compatible_type(row, feature):
    tags = feature.tags
    if row["category"] == "ROAD":
        return bool(tags.get("highway")) and not tags.get("highway", "").endswith(
            "_link"
        )
    if row["fine_type"] in {"lake", "reservoir", "pond"}:
        return feature.geometry.geom_type in {"Polygon", "MultiPolygon"}
    if row["fine_type"] == "waterfall":
        return tags.get("waterway") == "waterfall"
    return (
        feature.geometry.geom_type in {"LineString", "MultiLineString"}
        and bool(tags.get("waterway"))
        and tags.get("waterway") not in {"dam", "weir"}
    )


def connected_groups(features, tolerance_m):
    """Endpoint graph (never crossing-lines adjacency), with a tiny metric tolerance.

    GDAL does not expose node IDs. Exact endpoint coordinates are equivalent for
    intact ways; tolerance handles tiny digitization differences. No gaps are
    filled with fabricated line segments. Polygon connectivity uses intersection.
    """
    if not features:
        return []
    parent = list(range(len(features)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    cells = defaultdict(list)
    polygons = []
    for i, f in enumerate(features):
        g = project(f.geometry)
        if g.geom_type in {"Polygon", "MultiPolygon"}:
            polygons.append((i, g))
            continue
        parts = list(g.geoms) if hasattr(g, "geoms") else [g]
        for part in parts:
            coords = list(part.coords)
            for x, y in (coords[0], coords[-1]):
                cell = math.floor(x / tolerance_m), math.floor(y / tolerance_m)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for xx, yy, j in cells.get((cell[0] + dx, cell[1] + dy), []):
                            if math.hypot(x - xx, y - yy) <= tolerance_m:
                                join(i, j)
                cells[cell].append((x, y, i))
    if polygons:
        tree = STRtree([p[1] for p in polygons])
        for i, (index, geometry) in enumerate(polygons):
            for j in tree.query(geometry, predicate="intersects"):
                join(index, polygons[j][0])
    groups = defaultdict(list)
    for i, f in enumerate(features):
        groups[root(i)].append(f)
    return list(groups.values())


def merged_geometry(features):
    geometry = (
        unary_union([f.geometry for f in features])
        if features
        else GeometryCollection()
    )
    if geometry.geom_type == "MultiLineString":
        return linemerge(geometry)
    return geometry


def component_count(geometry):
    return (
        0
        if geometry.is_empty
        else len(geometry.geoms) if hasattr(geometry, "geoms") else 1
    )


def reconstruct_local_fine_feature(row, features, region, args):
    search = project(project(region).buffer(args.search_buffer_m), True)
    typed = [f for f in features if compatible_type(row, f)]
    hits = [(f, identity_match(row, f)) for f in typed]
    hits = [(f, m) for f, m in hits if m]
    local = [(f, m) for f, m in hits if f.geometry.intersects(search)]
    outside_hits = [(f, m) for f, m in hits if not f.geometry.intersects(search)]
    if not local:
        hits = [
            (f, identity_match(row, f, True))
            for f in typed
            if f.geometry.intersects(search)
        ]
        local = hits = [(f, m) for f, m in hits if m]
    empty = {
        "status": "NO_MATCH",
        "match_method": "",
        "candidate_matches": len(local),
        "same_name_outside_search": len(outside_hits),
        "outside_search_examples": [
            {"osm_id": f.osm_id, "field": m[2], "value": m[3], "state": f.state}
            for f, m in outside_hits[:10]
        ],
        "notes": [],
        "selected": [],
        "geometry": GeometryCollection(),
        "clipped": GeometryCollection(),
        "candidates": [f for f, _ in local],
    }
    if not local:
        return empty
    # A higher priority field chooses identity, but all equivalent ways are kept.
    best = min(m for _, m in local)
    primary = [(f, m) for f, m in local if m[0] == best[0]]
    primary_values = {
        normalize(m[3], row["category"], row["state"] or f.state) for f, m in primary
    }
    if len(primary_values) > 1:
        return {
            **empty,
            "status": "AMBIGUOUS",
            "match_method": best[1],
            "notes": [
                "Multiple canonical identities/networks have equal match priority"
            ],
        }
    all_hits = [
        f
        for f, m in hits
        if m[0] < 5
        or normalize(m[3], row["category"], row["state"] or f.state) in primary_values
    ]
    # A concurrent way (KY 30;KY 15) does not make KY 15 the requested
    # identity. Freeze numbered-route expansion to the query/resolved ref.
    route_ref = None
    if row["category"] == "ROAD":
        query_ref = normalize_reference(row["original_fine_name"], row["state"])
        if re.fullmatch(r"(?:us|i|[a-z]{2}) \d+[a-z]?", query_ref):
            route_ref = query_ref
        elif best[1] == "route_alias":
            route_ref = normalize_reference(best[3], row["state"])
    # Add ref-supported unnamed gaps before component selection. Otherwise a
    # short unnamed bridge splits two correctly named halves into an ambiguity.
    direct_ids = {(f.layer, f.osm_id) for f in all_hits}
    refs = {
        normalize_reference(v, row["state"] or f.state)
        for f in all_hits
        for v in split_values(f.tags.get("ref"))
    }
    if route_ref:
        refs = {route_ref}
    if row["category"] == "ROAD" and refs:
        explicit = bool(
            re.fullmatch(
                r"(?:us|i|[a-z]{2}) \d+[a-z]?",
                normalize_road_name(row["original_fine_name"], row["state"]),
            )
        )
        for f in typed:
            if (f.layer, f.osm_id) in direct_ids:
                continue
            frefs = {
                normalize_reference(v, row["state"] or f.state)
                for v in split_values(f.tags.get("ref"))
            }
            if refs & frefs and (
                explicit or not f.tags.get("name") or directional_continuation(row, f)
            ):
                all_hits.append(f)
    groups = connected_groups(all_hits, args.endpoint_tolerance_m)
    # Added ref-only pieces cannot seed a remote result on their own.
    primary_local_ids = {(f.layer, f.osm_id) for f, _ in local}
    relevant = [
        g for g in groups if any((f.layer, f.osm_id) in primary_local_ids for f in g)
    ]
    seed = safe_shape(row.get("seed_geometry"))
    separate_counties = 0
    if len(relevant) > 1:
        # Trusted text-derived geometry may disambiguate, never the landslide GT.
        distances = (
            sorted(
                (project(merged_geometry(g)).distance(project(seed)), i)
                for i, g in enumerate(relevant)
            )
            if seed is not None
            else []
        )
        if (
            distances
            and distances[0][0] <= args.seed_tolerance_m
            and (
                len(distances) == 1
                or distances[1][0] > max(args.seed_tolerance_m, distances[0][0] + 50)
            )
        ):
            selected = relevant[distances[0][1]]
            disconnected = len(relevant) - 1
            selected_counties = {
                f.tags["tiger:county"] for f in selected if f.tags.get("tiger:county")
            }
            selected_refs = {
                normalize_reference(v, row["state"] or f.state)
                for f in selected
                for v in split_values(f.tags.get("ref"))
            }
            for i, group in enumerate(relevant):
                if i == distances[0][1]:
                    continue
                other_counties = {
                    f.tags["tiger:county"] for f in group if f.tags.get("tiger:county")
                }
                other_refs = {
                    normalize_reference(v, row["state"] or f.state)
                    for f in group
                    for v in split_values(f.tags.get("ref"))
                }
                if (
                    selected_counties
                    and other_counties
                    and selected_counties.isdisjoint(other_counties)
                    and not selected_refs.intersection(other_refs)
                ):
                    separate_counties += 1
            disconnected -= separate_counties
        else:
            return {
                **empty,
                "status": "AMBIGUOUS",
                "match_method": best[1],
                "notes": [
                    f"{len(relevant)} disconnected same-name components; no unique seed agreement"
                ],
            }
    else:
        selected = relevant[0]
        disconnected = 0
    selected_ids = {(f.layer, f.osm_id) for f in selected}
    refs = set()
    for f in selected:
        refs.update(
            normalize_reference(v, row["state"] or f.state)
            for v in split_values(f.tags.get("ref"))
        )
    if route_ref:
        refs = {route_ref}
    expansion = []
    if row["category"] == "ROAD" and refs:
        explicit = bool(
            re.fullmatch(
                r"(?:us|i|[a-z]{2}) \d+[a-z]?",
                normalize_road_name(row["original_fine_name"], row["state"]),
            )
        )
        for f in typed:
            if (f.layer, f.osm_id) in selected_ids:
                continue
            frefs = {
                normalize_reference(v, row["state"] or f.state)
                for v in split_values(f.tags.get("ref"))
            }
            # A named road is not the entirety of every numbered route it uses.
            # Only unnamed or matching-name pieces continue a named identity.
            if refs & frefs and (
                explicit
                or not f.tags.get("name")
                or identity_match(row, f)
                or directional_continuation(row, f)
            ):
                expansion.append(f)
        expanded_groups = connected_groups(
            selected + expansion, args.endpoint_tolerance_m
        )
        selected = [
            f
            for g in expanded_groups
            if any((x.layer, x.osm_id) in selected_ids for x in g)
            for f in g
        ]
    geometry = merged_geometry(selected)
    clipped = geometry.intersection(region)
    local_only = merged_geometry([f for f in selected if f.geometry.intersects(search)])
    distance = project(seed).distance(project(geometry)) if seed is not None else None
    notes = []
    partial = disconnected > 0 or best[1] == "fuzzy_fallback"
    if disconnected:
        notes.append(
            f"{disconnected} disconnected candidate component(s) excluded by seed; completeness uncertain"
        )
    if distance is not None and distance > args.seed_tolerance_m:
        notes.append(
            f"Original seed disagrees by {distance:.1f} m; local identity requires review"
        )
        partial = True
    if clipped.is_empty:
        partial = True
        notes.append(
            "Match is in search buffer only; no geometry intersects candidate polygon"
        )
    length = project(geometry_helpers.line_only(geometry)).length
    clipped_length = project(geometry_helpers.line_only(clipped)).length
    if length > clipped_length + 10:
        notes.append(
            "Candidate clipping removes valid connected geometry; full geometry retained separately"
        )
    return {
        **empty,
        "status": "SUCCESS_PARTIAL" if partial else "SUCCESS_COMPLETE",
        "match_method": best[1],
        "matched_osm_field": best[2],
        "matched_osm_value": best[3],
        "match_score": best[4],
        "selected": selected,
        "geometry": geometry,
        "clipped": clipped,
        "segments_before_merge": len(selected),
        "components_after_merge": component_count(geometry),
        "connected_identity_components": len(relevant),
        "excluded_components": disconnected,
        "separate_county_components": separate_counties,
        "full_length_m": length,
        "clipped_length_m": clipped_length,
        "full_area_m2": project(geometry).area,
        "clipped_area_m2": project(clipped).area,
        "search_subset_length_m": project(
            geometry_helpers.line_only(local_only)
        ).length,
        "connected_segment_expansion": len(selected) > 1,
        "ref_expansion_segments": sum(
            (f.layer, f.osm_id) not in direct_ids for f in selected
        ),
        "directional_alias_segments": (
            sum(directional_continuation(row, f) for f in selected)
            if row["category"] == "ROAD"
            else 0
        ),
        "seed_distance_m": distance,
        "candidate_clipped": length > clipped_length + 10,
        "name_consistent": best[0] < 5,
        "spatial_consistent": not clipped.is_empty,
        "notes": notes,
    }


def evaluate_reconstruction(row, case, store, args):
    empty = {
        "status": "NO_MATCH",
        "match_method": "",
        "geometry": GeometryCollection(),
        "clipped": GeometryCollection(),
        "selected": [],
        "candidates": [],
        "notes": [],
    }
    region = case["region"]
    if region is None:
        return {
            **empty,
            "status": "DATA_MISSING",
            "notes": ["Candidate polygon missing", case["region_error"]],
        }
    if row["category"] == "OTHER":
        seed = safe_shape(row.get("seed_geometry"))
        # Reuse the existing resolution and eligibility result, no redesign.
        record = row["legacy_record"]
        if seed is not None and record.get("used_as_fine_evidence"):
            return {
                **empty,
                "status": "SUCCESS_COMPLETE",
                "match_method": "existing_other_geometry",
                "geometry": seed,
                "clipped": seed.intersection(region),
                "notes": [
                    "Existing project resolution preserved; identity not independently revalidated"
                ],
            }
        return {
            **empty,
            "notes": [
                record.get("failure_reason")
                or "No saved eligible geometry for other feature"
            ],
        }
    if not row.get("search_states"):
        return {
            **empty,
            "status": "AMBIGUOUS",
            "notes": [
                "State/candidate context insufficient; no national name search performed"
            ],
        }
    features = store.get(row)
    result = reconstruct_local_fine_feature(row, features, region, args)
    result["processed_status"] = result["status"]
    result["processed_length_m"] = result.get("full_length_m", 0)
    result["local_osm_files"] = sorted({f.source for f in features})
    result["raw_fallback_required"] = False
    result["raw_checked"] = False
    check_raw = args.raw_policy == "all" or (
        args.raw_policy == "diagnostic"
        and (
            result["status"] != "SUCCESS_COMPLETE"
            or result.get("candidate_clipped")
        )
    )
    if check_raw:
        raw = store.get(row, raw=True)
        raw_result = reconstruct_local_fine_feature(row, raw, region, args)
        result["raw_checked"] = True
        result["raw_status"] = raw_result["status"]
        result["raw_length_m"] = raw_result.get("full_length_m", 0)
        result["raw_osm_files"] = sorted({f.source for f in raw})
        proc_ids = {(f.layer, f.osm_id): f for f in features}
        raw_selected = {(f.layer, f.osm_id): f for f in raw_result.get("selected", [])}
        missing = [key for key in raw_selected if key not in proc_ids]
        changed = [
            key
            for key, f in raw_selected.items()
            if key in proc_ids
            and (
                f.tags != proc_ids[key].tags
                or not f.geometry.equals(proc_ids[key].geometry)
            )
        ]
        result["raw_missing_processed_ids"] = [list(k) for k in missing]
        result["raw_changed_processed_ids"] = [list(k) for k in changed]
        relation_matches = [
            f for f in raw if f.layer == "multilinestrings" and identity_match(row, f)
        ]
        result["raw_matching_relations"] = [
            {"id": f.osm_id, "tags": f.tags} for f in relation_matches
        ]
        if missing or changed:
            result["preprocessing_diagnosis"] = "RAW_HAS_MISSING_OR_CHANGED_FEATURES"
            if raw_result["status"].startswith("SUCCESS") and (
                not result["status"].startswith("SUCCESS") or missing
            ):
                metadata = {
                    k: v
                    for k, v in result.items()
                    if k.startswith(
                        ("raw_", "processed_", "local_osm", "preprocessing_")
                    )
                }
                result = {**raw_result, **metadata, "raw_fallback_required": True}
                result["notes"].append(
                    "Raw source used: relevant feature IDs/tags differ from processed extraction"
                )
        elif raw_result["status"] == "NO_MATCH" and result["status"] == "NO_MATCH":
            result["preprocessing_diagnosis"] = "NO_MATCH_IN_EITHER_SOURCE"
        else:
            result["preprocessing_diagnosis"] = "MATCHED_WAYS_AND_TAGS_PRESERVED"
        if row["category"] == "ROAD":
            result["notes"].append(
                "road_only filter preserves highway ways/tags but excludes parent route relations"
            )
    return result
