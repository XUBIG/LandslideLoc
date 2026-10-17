"""Frozen rainfall and global normalization functions, evaluated in landslide_localization."""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
DEFAULT_ID_SIGMOID_SCALE = 0.15


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def load_day_to_current_grid(
    promap,
    rainfall_dir: Path,
    day,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
) -> Optional[np.ndarray]:
    path = rainfall_dir / str(day.year) / f"PRISM_ppt_stable_800mD2_{day.strftime('%Y%m%d')}_bil.bil"
    arr = promap.load_rainfall_to_sus_grid(str(path), dst_shape, dst_transform, dst_crs)
    if arr is None:
        return None
    arr = arr.astype(float)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr < 0] = np.nan
    return arr

def load_rainfall_series_to_current_grid(
    promap,
    t_star: str,
    rainfall_dir: Path,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
    n_days: int,
) -> List[np.ndarray]:
    if not t_star:
        return []
    d0 = datetime.strptime(t_star, "%Y-%m-%d").date()
    grids: List[np.ndarray] = []
    for lag in range(n_days):
        daily = load_day_to_current_grid(promap, rainfall_dir, d0 - timedelta(days=lag), dst_shape, dst_transform, dst_crs)
        if daily is not None:
            grids.append(daily)
    return grids

def stack_sum_preserve_mask(grids: List[np.ndarray]) -> Optional[np.ndarray]:
    if not grids:
        return None
    stack = np.stack(grids, axis=0)
    valid = np.any(np.isfinite(stack), axis=0)
    out = np.nansum(stack, axis=0).astype(float)
    out[~valid] = np.nan
    return out

def api_grid(grids: List[np.ndarray], decay: float) -> Optional[np.ndarray]:
    if not grids:
        return None
    out = np.zeros_like(grids[0], dtype=float)
    valid = np.zeros_like(grids[0], dtype=bool)
    for lag, grid in enumerate(grids):
        mask_arr = np.isfinite(grid)
        out[mask_arr] += (grid[mask_arr] * float(decay ** lag)).astype(float)
        valid |= mask_arr
    out[~valid] = np.nan
    return out

def build_method_score(
    promap,
    method: str,
    t_star: str,
    rainfall_dir: Path,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    debug: Dict[str, Any] = {"method": method, "normalization": "none"}
    if not t_star:
        return None, None, {**debug, "reason": "no_t_star"}

    if method == "daily_linear":
        grids = load_rainfall_series_to_current_grid(promap, t_star, rainfall_dir, dst_shape, dst_transform, dst_crs, 1)
        if not grids:
            return None, None, {**debug, "reason": "no_daily_rain"}
        raw = grids[0]
        score = np.clip(raw / float(args.rain_scale_mm), 0.0, float(args.rain_cap)).astype(float)
        score[~np.isfinite(raw)] = np.nan
        debug.update({"rain_scale_mm": float(args.rain_scale_mm), "rain_cap": float(args.rain_cap)})
        return score, raw, debug

    if method in {"r3_saturating", "r7_saturating"}:
        days = 3 if method == "r3_saturating" else 7
        grids = load_rainfall_series_to_current_grid(promap, t_star, rainfall_dir, dst_shape, dst_transform, dst_crs, days)
        raw = stack_sum_preserve_mask(grids)
        if raw is None:
            return None, None, {**debug, "reason": f"no_r{days}_rain"}
        score = (1.0 - np.exp(-raw / float(args.tau_mm))).astype(float)
        score = np.clip(score, 0.0, 1.0)
        score[~np.isfinite(raw)] = np.nan
        debug.update({"days": days, "tau_mm": float(args.tau_mm)})
        return score, raw, debug

    if method == "api7_saturating":
        grids = load_rainfall_series_to_current_grid(promap, t_star, rainfall_dir, dst_shape, dst_transform, dst_crs, 7)
        raw = api_grid(grids, args.api_decay)
        if raw is None:
            return None, None, {**debug, "reason": "no_api7_rain"}
        score = (1.0 - np.exp(-raw / float(args.tau_mm))).astype(float)
        score = np.clip(score, 0.0, 1.0)
        score[~np.isfinite(raw)] = np.nan
        debug.update({"days": 7, "api_decay": float(args.api_decay), "tau_mm": float(args.tau_mm)})
        return score, raw, debug

    if method == "id_trigger":
        sigmoid_scale = float(getattr(args, "id_sigmoid_scale", DEFAULT_ID_SIGMOID_SCALE))
        if not np.isfinite(sigmoid_scale) or sigmoid_scale <= 0.0:
            raise ValueError(f"id_sigmoid_scale must be finite and greater than zero; got {sigmoid_scale!r}")
        durations = (1, 3, 5, 7)
        thresholds = {
            1: float(args.id_threshold_1),
            3: float(args.id_threshold_3),
            5: float(args.id_threshold_5),
            7: float(args.id_threshold_7),
        }
        grids = load_rainfall_series_to_current_grid(promap, t_star, rainfall_dir, dst_shape, dst_transform, dst_crs, max(durations))
        if not grids:
            return None, None, {**debug, "reason": "no_id_rain"}
        exceed_maps = []
        for days in durations:
            cum = stack_sum_preserve_mask(grids[:days])
            if cum is None:
                continue
            intensity = cum / float(days)
            exceed_maps.append((intensity / (thresholds[days] + 1e-9)).astype(float))
        if not exceed_maps:
            return None, None, {**debug, "reason": "empty_id_exceed"}
        raw = np.maximum.reduce(exceed_maps).astype(float)
        valid = np.isfinite(raw)
        score = sigmoid((raw - 1.0) / sigmoid_scale).astype(float)
        score = np.clip(score, 0.0, 1.0)
        score[~valid] = np.nan
        debug.update(
            {
                "id_durations": list(durations),
                "id_thresholds_mm_per_day": thresholds,
                "id_sigmoid_scale": sigmoid_scale,
            }
        )
        return score, raw, debug

    return None, None, {**debug, "reason": f"unknown_method:{method}"}

def global_sus_min_max(src, stripe_height: int = 256) -> Tuple[float, float, int]:
    """Read the source raster exactly and return its global valid-data extrema."""
    global_min = float("inf")
    global_max = float("-inf")
    valid_count = 0
    row_offsets = range(0, src.height, stripe_height)
    for row_off in tqdm(row_offsets, total=(src.height + stripe_height - 1) // stripe_height, desc="Scanning global SUS range"):
        height = min(stripe_height, src.height - row_off)
        data = src.read(1, window=Window(0, row_off, src.width, height), masked=False)
        valid = np.isfinite(data)
        if src.nodata is not None:
            valid &= data != src.nodata
        if not np.any(valid):
            continue
        values = data[valid]
        global_min = min(global_min, float(np.min(values)))
        global_max = max(global_max, float(np.max(values)))
        valid_count += int(values.size)
    if valid_count == 0:
        raise RuntimeError("The SUS raster contains no valid pixels.")
    if not global_max > global_min:
        raise RuntimeError(f"Global SUS range is degenerate: min={global_min}, max={global_max}")
    return global_min, global_max, valid_count
