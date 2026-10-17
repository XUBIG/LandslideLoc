"""Spatial geometry, geocoding, image evidence, rainfall, and map evaluation helpers."""


import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
import requests
from geopy.distance import geodesic
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from shapely.ops import transform


WORK_CRS = "EPSG:5070"  # NAD83 / Conus Albers，单位米
DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


NOMINATIM_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
NOMINATIM_CACHE_FILE = os.path.join(NOMINATIM_CACHE_DIR, "nominatim_cache.json")
_nominatim_cache = None


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _load_nominatim_cache():
    global _nominatim_cache
    if _nominatim_cache is not None:
        return
    os.makedirs(NOMINATIM_CACHE_DIR, exist_ok=True)
    if os.path.exists(NOMINATIM_CACHE_FILE):
        try:
            with open(NOMINATIM_CACHE_FILE, "r", encoding="utf-8") as f:
                _nominatim_cache = json.load(f)
            dprint(f"[CACHE] Loaded Nominatim cache: {NOMINATIM_CACHE_FILE}, entries={len(_nominatim_cache)}")
        except Exception as e:
            dprint(f"[CACHE] Failed to load cache: {e}, start empty")
            _nominatim_cache = {}
    else:
        _nominatim_cache = {}
        dprint("[CACHE] Cache not found, start empty")


def _save_nominatim_cache():
    if _nominatim_cache is None:
        return
    try:
        with open(NOMINATIM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_nominatim_cache, f, ensure_ascii=False)
        dprint(f"[CACHE] Saved cache: entries={len(_nominatim_cache)}")
    except Exception as e:
        dprint(f"[CACHE] Save failed: {e}")


def geocode_nominatim(place: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _load_nominatim_cache()
    key = f"search::{_normalize_query(place)}"

    if key in _nominatim_cache:
        cached = _nominatim_cache[key]
        if cached is None:
            dprint(f"[NOMINATIM][CACHE HIT][search] '{place}' -> failed before")
            return None, None
        dprint(f"[NOMINATIM][CACHE HIT][search] '{place}'")
        return cached.get("geometry"), cached.get("properties")

    time.sleep(1.0)
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "geojson", "polygon_geojson": 1, "limit": 1}
    headers = {"User-Agent": "LandslideInferenceBot/1.0 (xubing2333@qq.com)"}

    dprint(f"[NOMINATIM][HTTP][search] '{place}'")
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            dprint(f"[NOMINATIM][HTTP ERROR][search] status={response.status_code}")
            _nominatim_cache[key] = None
            _save_nominatim_cache()
            return None, None
        data = response.json()
    except Exception as e:
        dprint(f"[NOMINATIM][EXCEPTION][search] '{place}': {e}")
        _nominatim_cache[key] = None
        _save_nominatim_cache()
        return None, None

    if data.get("features"):
        geom = data["features"][0]["geometry"]
        props = data["features"][0]["properties"]
        _nominatim_cache[key] = {"geometry": geom, "properties": props}
        _save_nominatim_cache()
        return geom, props

    dprint(f"[NOMINATIM][EMPTY][search] '{place}'")
    _nominatim_cache[key] = None
    _save_nominatim_cache()
    return None, None


def reproject_geom(shapely_geom, src_crs: str, dst_crs: str):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, shapely_geom)


def load_image_semantics(image_json_path: str):
    """
    读取 image_semantics_vector_prob.json
    返回: dict[tweets_id] -> list[ per-image dict ]
    """
    if not image_json_path or not os.path.exists(image_json_path):
        dprint(f"[IMG] image semantics json not found: {image_json_path}")
        return {}
    with open(image_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    by_tweet = {}
    for fname, attrs in raw.items():
        base = os.path.basename(fname)
        m = re.match(r"^(\d+)_\d+\.(jpg|png|jpeg)$", base, re.IGNORECASE)
        if not m:
            m = re.match(r"^(\d+)_\d+$", base)
        if not m:
            continue
        tid = m.group(1)

        def get_prob(key):
            if key in attrs and isinstance(attrs[key], dict):
                return float(attrs[key].get("prob", 0.0))
            return 0.0

        p_ls = get_prob("has_landslide")
        p_road = get_prob("has_road")
        p_water = get_prob("has_water")
        p_bare = get_prob("has_bare_soil")

        by_tweet.setdefault(tid, []).append({
            "p_landslide": p_ls,
            "p_road": p_road,
            "p_water": p_water,
            "p_bare": p_bare,
            "fname": base,
        })

    dprint(f"[IMG] Loaded image semantics for {len(by_tweet)} tweets from {image_json_path}")
    return by_tweet


def aggregate_probs_per_tweet(per_images: list,
                              landslide_gate_thr: float = 0.5,
                              mode: str = "noisy_or"):
    """
    返回: (keep_tweet, p_ls_agg, p_road_agg, p_water_agg, debug_dict)
    keep_tweet=False 表示“该推文图片里基本没滑坡”，可过滤（这里用于先验分档）。
    """
    # Explicit modes are essential: startswith("noisy_or") used to make the
    # supposedly pure noisy_or mode include 30% weighted average as well.
    if mode not in {"noisy_or_weighted", "noisy_or", "weighted_avg"}:
        raise ValueError(f"Unknown image aggregation mode: {mode!r}")
    if not per_images:
        return False, 0.0, 0.0, 0.0, {"note": "no_images", "aggregation_mode": mode}

    ls_probs = np.array([img["p_landslide"] for img in per_images], dtype=float)
    has_ls = np.any(ls_probs >= landslide_gate_thr)
    if not has_ls:
        return False, float(ls_probs.max(initial=0.0)), 0.0, 0.0, {"note": "no_landslide_gate", "aggregation_mode": mode}

    mask_sel = ls_probs >= landslide_gate_thr
    if not np.any(mask_sel):
        mask_sel = ls_probs >= (landslide_gate_thr * 0.8)

    sel = [per_images[i] for i in range(len(per_images)) if mask_sel[i]]
    w = np.array([max(sel[i]["p_landslide"], 1e-3) for i in range(len(sel))], dtype=float)

    p_road_arr  = np.array([d["p_road"]  for d in sel], dtype=float)
    p_water_arr = np.array([d["p_water"] for d in sel], dtype=float)

    if mode in {"noisy_or", "noisy_or_weighted"}:
        p_road_agg  = 1.0 - float(np.prod(1.0 - p_road_arr))  if len(p_road_arr)  else 0.0
        p_water_agg = 1.0 - float(np.prod(1.0 - p_water_arr)) if len(p_water_arr) else 0.0
        mix = 0.3
        if mode == "noisy_or_weighted" and w.sum() > 0 and len(sel) > 0:
            p_road_agg  = (1-mix)*p_road_agg  + mix*float((w*p_road_arr).sum()/w.sum())
            p_water_agg = (1-mix)*p_water_agg + mix*float((w*p_water_arr).sum()/w.sum())
    else:
        if w.sum() > 0:
            p_road_agg  = float((w*p_road_arr).sum()/w.sum())  if len(p_road_arr)  else 0.0
            p_water_agg = float((w*p_water_arr).sum()/w.sum()) if len(p_water_arr) else 0.0
        else:
            p_road_agg  = float(p_road_arr.mean())  if len(p_road_arr)  else 0.0
            p_water_agg = float(p_water_arr.mean()) if len(p_water_arr) else 0.0

    return True, float(ls_probs.max(initial=0.0)), p_road_agg, p_water_agg, {
        "n_images": len(per_images),
        "n_used": int(np.sum(mask_sel)),
        "gate_thr": landslide_gate_thr,
        "aggregation_mode": mode,
        "weighted_average_mix": 0.3 if mode == "noisy_or_weighted" else (1.0 if mode == "weighted_avg" else 0.0),
    }


def load_candidate_dataset(candidate_file: str, metadata_file: str):
    """Load candidates while taking evaluation coordinates from the GT metadata."""
    with open(candidate_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    with open(metadata_file, "r", encoding="utf-8-sig") as f:
        metadata = json.load(f)
    by_id = {str(item.get("tweets_id")): item for item in metadata}
    if len(by_id) != len(metadata):
        raise ValueError("GT metadata contains duplicate case IDs")
    merged = []
    for candidate in candidates:
        case_id = str(candidate.get("tweets_id"))
        if case_id not in by_id:
            raise ValueError(f"Missing GT metadata for {case_id}")
        metadata_record = by_id[case_id]
        coords = metadata_record.get("coordinates")
        if (not isinstance(coords, (list, tuple)) or len(coords) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) for value in coords)
                or not -180 <= coords[0] <= 180 or not -90 <= coords[1] <= 90):
            raise ValueError(f"Invalid GT coordinates for {case_id}: {coords!r}")
        source = dict(metadata_record)
        source.update(candidate)
        # Candidate exports contain historical labels; a later annotation
        # correction must take effect without modifying inference geometries.
        source["coordinates"] = list(coords)
        merged.append(source)
    return merged


def candidate_geometry(item: Dict[str, Any]):
    """Prefer the selected candidate polygon; re-geocode legacy compact outputs."""
    area = item.get("candidate_area") or {}
    geojson = area.get("geojson")
    if isinstance(geojson, dict):
        return geojson, {"candidate_source": "pipeline_geojson"}
    if area.get("geojson_available") and area.get("display_name"):
        geometry, properties = geocode_nominatim(area["display_name"])
        if geometry is not None:
            props = dict(properties or {})
            props["candidate_source"] = "pipeline_display_name"
            return geometry, props
    return None, {"candidate_source": "pipeline_unresolved"}


def local_text_evidence(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retain geocoded fine features suitable for within-polygon up-weighting."""
    usable = []
    for evidence in item.get("weight_evidence", []):
        if evidence.get("weight_hint") not in {"high", "medium"}:
            continue
        if evidence.get("admin_match_to_main") not in {"same_admin1", "same_admin2"}:
            continue
        if not evidence.get("resolved_name"):
            continue
        usable.append(evidence)
    return usable


def build_analysis_polygon(geom_work, geom_type: str,
                           point_buffer_m: float = 5000.0,
                           line_buffer_m: float = 5000.0):
    if geom_type == "Point":
        return geom_work.buffer(point_buffer_m)
    elif geom_type in ["LineString", "MultiLineString"]:
        return geom_work.buffer(line_buffer_m)
    elif geom_type in ["Polygon", "MultiPolygon"]:
        return geom_work.buffer(0)
    else:
        return geom_work.convex_hull


WEEKDAY_MAP = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def infer_event_date_from_text(tweet_text: str, tweet_date_str: str):
    """
    规则解析 tweet 文本中的相对时间表达，返回：
      (event_date_str, source, debug)
    成功 source="text"，失败返回 (None, None, debug)
    默认倾向“过去发生”（不晚于 tweet_date）
    """
    if not tweet_text or not tweet_date_str:
        return None, None, {"reason": "no_text_or_date"}

    text = tweet_text.lower()
    base = datetime.strptime(tweet_date_str, "%Y-%m-%d").date()

    if re.search(r"\byesterday\b", text):
        return (base - timedelta(days=1)).strftime("%Y-%m-%d"), "text", {"match": "yesterday"}
    if re.search(r"\btoday\b", text):
        return base.strftime("%Y-%m-%d"), "text", {"match": "today"}
    if re.search(r"\blast night\b", text) or re.search(r"\btonight\b", text):
        return (base - timedelta(days=1)).strftime("%Y-%m-%d"), "text", {"match": "last_night"}

    m = re.search(r"\b(\d+)\s+days?\s+ago\b", text)
    if m:
        n = int(m.group(1))
        return (base - timedelta(days=n)).strftime("%Y-%m-%d"), "text", {"match": f"{n}_days_ago"}

    if re.search(r"\blast week\b", text):
        return (base - timedelta(days=7)).strftime("%Y-%m-%d"), "text", {"match": "last_week"}
    if re.search(r"\blast month\b", text):
        return (base - timedelta(days=30)).strftime("%Y-%m-%d"), "text", {"match": "last_month"}

    wd_pat = r"\b(last|this|next)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b"
    m = re.search(wd_pat, text)
    if m:
        qualifier = (m.group(1) or "").strip()
        wd_str = m.group(2)
        target = WEEKDAY_MAP.get(wd_str)
        if target is not None:
            cur = base.weekday()

            if qualifier == "next":
                delta = (target - cur) % 7
                delta = 7 if delta == 0 else delta
                d = base + timedelta(days=delta)
                return d.strftime("%Y-%m-%d"), "text", {"match": f"next_{wd_str}"}

            if qualifier == "last":
                delta = (cur - target) % 7
                delta = 7 if delta == 0 else delta
                d = base - timedelta(days=delta) - timedelta(days=7)
                return d.strftime("%Y-%m-%d"), "text", {"match": f"last_{wd_str}"}

            # this/none：取最近一次不晚于 base 的 weekday
            delta = (cur - target) % 7
            d = base - timedelta(days=delta)
            return d.strftime("%Y-%m-%d"), "text", {"match": f"{qualifier or 'near'}_{wd_str}"}

    return None, None, {"reason": "no_relative_time_matched"}


def load_rainfall_to_sus_grid(
    rainfall_path: str,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
):
    if (not rainfall_path) or (not os.path.exists(rainfall_path)):
        return None

    dst = np.full(dst_shape, np.nan, dtype=float)

    with rasterio.open(rainfall_path) as src:
        src_arr = src.read(1).astype(float)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_arr[src_arr == src_nodata] = np.nan
        src_arr[src_arr < 0] = np.nan

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    return dst


def _pixel_area_km2_approx(transform, lat_ref: float, lon_ref: float):
    """
    近似计算 EPSG:4326 下单个像元面积（km^2），用于小窗口统计。
    """
    dx_deg = float(transform.a)
    dy_deg = float(abs(transform.e))
    dx_m = geodesic((lat_ref, lon_ref), (lat_ref, lon_ref + dx_deg)).meters
    dy_m = geodesic((lat_ref, lon_ref), (lat_ref + dy_deg, lon_ref)).meters
    return float((dx_m * dy_m) / 1e6)


def evaluate_probmap_case_from_array(
    arr,
    transform,
    gt_lon: float,
    gt_lat: float,
    top_percents=(1, 5, 10, 15, 20, 30),
):
    valid_mask = np.isfinite(arr)
    result = {
        "gt_in_valid_raster": False,
        "pr": 0.0,
        "percentile_rank": 0.0,
        "gt_pixel_prob": None,
    }
    for p in top_percents:
        result[f"hit_top{p}"] = False
        result[f"area_top{p}_km2"] = 0.0
        result[f"area_top{p}_percent"] = 0.0

    if not np.any(valid_mask):
        return result

    try:
        inv = ~transform
        col_f, row_f = inv * (gt_lon, gt_lat)
        row, col = int(np.floor(row_f)), int(np.floor(col_f))
    except Exception:
        return result

    if row < 0 or col < 0 or row >= arr.shape[0] or col >= arr.shape[1]:
        return result

    p_true = arr[row, col]
    if not np.isfinite(p_true):
        return result

    valid_vals = arr[valid_mask]
    pr = float(np.mean(valid_vals <= p_true))
    pix_area_km2 = _pixel_area_km2_approx(transform, gt_lat, gt_lon)

    result.update({
        "gt_in_valid_raster": True,
        "pr": pr,
        "percentile_rank": pr,
        "gt_pixel_prob": float(p_true),
    })
    thresholds = np.nanpercentile(valid_vals, [100 - p for p in top_percents])
    for p, threshold in zip(top_percents, thresholds):
        thr = float(threshold)
        result[f"hit_top{p}"] = bool(p_true >= thr)
        top_cnt = int(np.sum(valid_mask & (arr >= thr)))
        result[f"area_top{p}_km2"] = float(top_cnt * pix_area_km2)
        result[f"area_top{p}_percent"] = float(100.0 * top_cnt / valid_vals.size)
    return result
