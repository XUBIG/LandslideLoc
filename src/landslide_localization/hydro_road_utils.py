# -*- coding:utf-8-*-
# @Time   :2025/8/12 上午3:32
# @Author :Bing Xu
# @FileName: hydro_road_utils.py
# @Software: PyCharm

"""
hydro_road_utils.py
====================

这个模块提供从OSM和NHD中提取道路和河流要素，并生成缓冲区及对应权重的辅助函数。
"""

import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import box, Point
from typing import Tuple, Optional, Dict, Any

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

import warnings
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)


DRIVABLE_HIGHWAY_TYPES = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "residential", "unclassified", "living_street",
    "service", "road",
}
HYDRO_WATERWAY_TYPES = {"river", "stream", "ditch"}


def _ensure_polygon_aoi(aoi_gdf: gpd.GeoDataFrame, target_crs: str, fallback_buffer_m: float = 10000) -> gpd.GeoDataFrame:
    """
    确保 AOI 为 Polygon/MultiPolygon；若是点/线，则在米制 CRS 下缓冲一圈再返回到目标 CRS。
    """
    if aoi_gdf.empty:
        return aoi_gdf
    aoi = aoi_gdf.iloc[[0]].copy()  # 仅用第一个几何
    if aoi.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon"):
        return aoi.to_crs(target_crs)

    # 投影到米制 CRS 做缓冲
    aoi_m = aoi.to_crs("EPSG:5070")
    ring = aoi_m.geometry.buffer(fallback_buffer_m).unary_union
    if isinstance(ring, (Polygon, MultiPolygon)):
        res = gpd.GeoDataFrame(geometry=[ring], crs=aoi_m.crs)
    else:
        res = gpd.GeoDataFrame(geometry=[ring.convex_hull], crs=aoi_m.crs)
    return res.to_crs(target_crs)

def _coerce_lines_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    仅保留线要素；若存在面要素，则取其边界作为线加入；并修补无效几何，丢弃空几何。
    """
    if gdf.empty:
        return gdf

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(0)  # 修补 invalid
    is_line = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    is_area = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])

    lines = gdf[is_line].copy()

    if is_area.any():
        borders = gdf[is_area].geometry.boundary
        borders_gdf = gpd.GeoDataFrame(gdf[is_area].drop(columns="geometry"), geometry=borders, crs=gdf.crs)
        lines = pd.concat([lines, borders_gdf], ignore_index=True)

    lines = lines[~lines.geometry.is_empty & lines.geometry.notnull()].copy()
    # explode 以免后续 contains、clip 出现 Multi*
    lines = lines.explode(index_parts=False, ignore_index=True)
    return lines


def extract_road_network(polygon_gdf: gpd.GeoDataFrame,
                         network_type: str = "drive",
                         target_crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    """
    从 OSM 按多边形区域提取道路网络，并投影到米制坐标。
    """
    geom = polygon_gdf.to_crs("EPSG:4326").geometry.union_all()
    try:
        G = ox.graph.graph_from_polygon(geom, network_type=network_type,
                                        simplify=True, retain_all=False)
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        return edges.to_crs(target_crs)
    except Exception:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)


def extract_road_network_from_pbf(pbf_path: str,
                                  polygon_gdf: gpd.GeoDataFrame,
                                  target_crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    """Read drivable roads for an AOI from a local OSM PBF file.

    GDAL's OSM ``lines`` layer is filtered by the AOI bounding box while it is
    read, then clipped to the exact polygon in the projected working CRS.
    Callers should cache the clipped result because country-scale PBF files do
    not provide a reusable per-AOI spatial index.
    """
    from pathlib import Path

    import pyogrio

    source = Path(pbf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    poly_4326 = _ensure_polygon_aoi(polygon_gdf, "EPSG:4326")
    if poly_4326.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    minx, miny, maxx, maxy = (float(value) for value in poly_4326.total_bounds)
    roads = pyogrio.read_dataframe(
        source,
        layer="lines",
        bbox=(minx, miny, maxx, maxy),
        columns=["osm_id", "name", "highway"],
        where="highway IS NOT NULL",
        use_arrow=False,
    )
    if roads is None or roads.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    roads = roads[roads["highway"].isin(DRIVABLE_HIGHWAY_TYPES)].copy()
    roads = roads[
        roads.geometry.notnull()
        & ~roads.geometry.is_empty
        & roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()
    if roads.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    roads_work = roads.to_crs(target_crs)
    aoi_work = _ensure_polygon_aoi(poly_4326, target_crs).geometry.union_all()
    roads_work = gpd.clip(roads_work, aoi_work)
    roads_work = roads_work[roads_work.geometry.notnull() & ~roads_work.geometry.is_empty].copy()
    if roads_work.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    return roads_work.explode(index_parts=False, ignore_index=True).set_crs(target_crs)


def extract_hydro_network_from_pbf(pbf_path: str,
                                   polygon_gdf: gpd.GeoDataFrame,
                                   target_crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    """Read river/stream/ditch lines for an AOI from a local OSM PBF."""
    from pathlib import Path

    import pyogrio

    source = Path(pbf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    poly_4326 = _ensure_polygon_aoi(polygon_gdf, "EPSG:4326")
    if poly_4326.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    minx, miny, maxx, maxy = (float(value) for value in poly_4326.total_bounds)
    hydro = pyogrio.read_dataframe(
        source,
        layer="lines",
        bbox=(minx, miny, maxx, maxy),
        columns=["osm_id", "name", "waterway"],
        where="waterway IS NOT NULL",
        use_arrow=False,
    )
    if hydro is None or hydro.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    hydro = hydro[hydro["waterway"].isin(HYDRO_WATERWAY_TYPES)].copy()
    hydro = hydro[
        hydro.geometry.notnull()
        & ~hydro.geometry.is_empty
        & hydro.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()
    if hydro.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    hydro_work = hydro.to_crs(target_crs)
    aoi_work = _ensure_polygon_aoi(poly_4326, target_crs).geometry.union_all()
    hydro_work = gpd.clip(hydro_work, aoi_work)
    hydro_work = hydro_work[
        hydro_work.geometry.notnull() & ~hydro_work.geometry.is_empty
    ].copy()
    if hydro_work.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    return hydro_work.explode(index_parts=False, ignore_index=True).set_crs(target_crs)

def extract_rivers_from_osm(polygon_gdf: gpd.GeoDataFrame,
                            tags: Optional[Dict[str, Any]] = None,
                            target_crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    """
    从 OSM 提取水系，统一转为线，并与 AOI 裁剪。避免 overlay 的混合几何报错。
    """
    if tags is None:
        tags = {"waterway": ["river", "stream", "ditch"]}

    poly_4326 = polygon_gdf.to_crs("EPSG:4326")
    geom = poly_4326.geometry.union_all()

    try:
        gdf = ox.features_from_polygon(geom, tags=tags)
    except Exception:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    # 只保留线（面转边界）
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    gdf = _coerce_lines_only(gdf)

    if gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    # AOI 一定是 Polygon
    aoi = _ensure_polygon_aoi(poly_4326, target_crs=target_crs)

    # 投影到目标 CRS 后裁剪
    gdf_proj = gdf.to_crs(target_crs)
    aoi_proj = aoi.to_crs(target_crs)
    aoi_union = unary_union(aoi_proj.geometry)

    rivers = gpd.clip(gdf_proj, aoi_union)
    rivers = rivers[~rivers.geometry.is_empty & rivers.geometry.notnull()].copy()
    rivers = rivers.explode(index_parts=False, ignore_index=True)
    return rivers.set_crs(target_crs)

def extract_rivers_from_nhd(gdb_path: str,
                            polygon_gdf: gpd.GeoDataFrame,
                            target_crs: str = "EPSG:5070",
                            layers: Tuple[str, ...] = ("NetworkNHDFlowline", "NonNetworkNHDFlowline"),
                            bbox_expand: float = 1.0) -> gpd.GeoDataFrame:
    """
    从 NHDPlus 提取线状水系，并裁剪到 AOI。
    """
    poly_5070 = _ensure_polygon_aoi(polygon_gdf, target_crs)
    minx, miny, maxx, maxy = poly_5070.total_bounds
    if bbox_expand != 1.0:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        w, h = (maxx - minx) * bbox_expand / 2, (maxy - miny) * bbox_expand / 2
        minx, maxx, miny, maxy = cx - w, cx + w, cy - h, cy + h
    bbox_geom = box(minx, miny, maxx, maxy)

    parts = []
    for lyr in layers:
        try:
            sub = gpd.read_file(gdb_path, layer=lyr, engine="fiona", bbox=bbox_geom)
            if not sub.empty:
                parts.append(sub)
        except Exception:
            continue

    if not parts:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    flow = pd.concat(parts, ignore_index=True)
    flow = flow.to_crs(target_crs)

    aoi_union = unary_union(poly_5070.geometry)
    flow_clip = gpd.clip(flow, aoi_union)
    flow_clip = flow_clip[~flow_clip.geometry.is_empty & flow_clip.geometry.notnull()].copy()
    flow_clip = flow_clip.explode(index_parts=False, ignore_index=True)
    return flow_clip.set_crs(target_crs)

def get_rivers(gdb_path: Optional[str], polygon_gdf: gpd.GeoDataFrame,
               target_crs: str = "EPSG:5070", prefer: str = "nhd",
               bbox_expand: float = 1.2, osm_tags: Optional[Dict[str, Any]] = None,
               return_source: bool = False) -> Any:
    """
    统一获取水系数据：优先 NHD 或 OSM。
    """
    prefer = prefer.lower()
    source = None
    rivers = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    if prefer == "nhd" and gdb_path:
        rivers = extract_rivers_from_nhd(gdb_path, polygon_gdf, target_crs=target_crs,
                                         bbox_expand=bbox_expand)
        source = "NHD"
        if rivers.empty:
            rivers = extract_rivers_from_osm(polygon_gdf, tags=osm_tags, target_crs=target_crs)
            source = "OSM"
    else:
        rivers = extract_rivers_from_osm(polygon_gdf, tags=osm_tags, target_crs=target_crs)
        source = "OSM"
        if rivers.empty and gdb_path:
            rivers = extract_rivers_from_nhd(gdb_path, polygon_gdf, target_crs=target_crs,
                                             bbox_expand=bbox_expand)
            source = "NHD"
    return (rivers, source) if return_source else rivers

def generate_weighted_buffer(polygon_gdf: gpd.GeoDataFrame,
                             road_gdf: gpd.GeoDataFrame,
                             river_gdf: gpd.GeoDataFrame,
                             buffer_params: Dict[str, float]) -> gpd.GeoDataFrame:
    """
    对道路与河流生成缓冲区并赋权。使用 clip 而非 overlay，避免几何类型问题。
    """
    # 统一 CRS（优先沿用道路 CRS；否则河流 CRS；都没有则用 EPSG:5070）
    target_crs = road_gdf.crs or river_gdf.crs or "EPSG:5070"

    aoi = _ensure_polygon_aoi(polygon_gdf, target_crs)
    aoi_union = unary_union(aoi.geometry)

    weighted_rows = []

    # 道路缓冲
    if road_gdf is not None and not road_gdf.empty:
        roads = road_gdf.to_crs(target_crs)
        rb = roads.buffer(buffer_params.get("road", 100))
        rb = gpd.GeoDataFrame(geometry=rb, crs=target_crs)
        rb = gpd.clip(rb, aoi_union)
        if not rb.empty:
            weighted_rows.append({
                "geometry": rb.unary_union,
                "weight": float(buffer_params.get("road_weight", 1.0)),
                "kind": "road"
            })

    # 河流缓冲
    if river_gdf is not None and not river_gdf.empty:
        rivers = river_gdf.to_crs(target_crs)
        rvb = rivers.buffer(buffer_params.get("river", 200))
        rvb = gpd.GeoDataFrame(geometry=rvb, crs=target_crs)
        rvb = gpd.clip(rvb, aoi_union)
        if not rvb.empty:
            weighted_rows.append({
                "geometry": rvb.unary_union,
                "weight": float(buffer_params.get("river_weight", 1.0)),
                "kind": "river"
            })

    if not weighted_rows:
        return gpd.GeoDataFrame(columns=["geometry", "weight", "kind"], geometry="geometry", crs=target_crs)

    return gpd.GeoDataFrame(weighted_rows, geometry="geometry", crs=target_crs)

def compute_weight(point: Point,
                   road_union: Optional[Any],
                   river_union: Optional[Any],
                   buffer_params: Dict[str, float]) -> float:
    """
    对单个点计算加权结果。如果落在道路缓冲区内，则乘 road_weight；落在河流缓冲区内则乘 river_weight。
    """
    weight = 1.0
    road_w = buffer_params.get("road_weight", 1.0)
    river_w = buffer_params.get("river_weight", 1.0)
    if road_union is not None and road_union.contains(point):
        weight *= road_w
    if river_union is not None and river_union.contains(point):
        weight *= river_w
    return weight

def prepare_buffer_unions(buffer_gdf: gpd.GeoDataFrame) -> Tuple[Optional[Any], Optional[Any]]:
    """
    将加权缓冲区拆成 road_union 与 river_union。
    优先使用 'kind' 字段；若没有，则按权重最大者当作 river（保留兼容）。
    """
    road_union = None
    river_union = None

    if "kind" in buffer_gdf.columns:
        road_rows = buffer_gdf[buffer_gdf["kind"] == "road"]
        river_rows = buffer_gdf[buffer_gdf["kind"] == "river"]
        if not road_rows.empty:
            road_union = unary_union(road_rows.geometry)
        if not river_rows.empty:
            river_union = unary_union(river_rows.geometry)
        return road_union, river_union

    # 兼容旧逻辑（不推荐）
    for _, row in buffer_gdf.iterrows():
        if row["weight"] == buffer_gdf["weight"].max():
            river_union = row["geometry"]
        else:
            road_union = row["geometry"]
    return road_union, river_union
