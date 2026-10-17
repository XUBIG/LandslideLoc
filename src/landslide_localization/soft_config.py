from __future__ import annotations

from landslide_localization.runtime_paths import REPO_ROOT as ROOT
SOFT_ROOT = ROOT / "outputs"

# Define the evaluation grid once and import it everywhere else.
TOP_K_VALUES = [1, 5, 10, 15, 20, 30, 40, 50]

# Weighted Soft-OR parameters and formula strings are centralized here so the
# CLI, evaluator, cache metadata, and result metadata cannot drift apart.
DEFAULT_W_S = 0.8
DEFAULT_W_R = 1.0
WEIGHTED_SOFT_OR_FORMULA = (
    "H = 1 - (1 - w_s*S_norm)*(1 - w_r*R_id_trigger)"
)
WEIGHTED_SOFT_OR_EXPANDED_FORMULA = (
    "H = w_s*S_norm + w_r*R_id_trigger - "
    "w_s*w_r*S_norm*R_id_trigger"
)
FULL_WEIGHTED_SOFT_OR_FORMULA = "H_full = H_weighted_softOR * F_fine"

EXPERIMENT_FORMULAS = {
    "E1_sus": "S_norm",
    "E2_rain": "R_id_trigger",
    "E3_sus_rain_softor": WEIGHTED_SOFT_OR_FORMULA,
    "E4_sus_fine": "S_norm * F_fine",
    "E5_full_softor_fine": (
        "[w_s*S_norm + w_r*R_id_trigger - "
        "w_s*w_r*S_norm*R_id_trigger] * F_fine"
    ),
    "E6_rain_fine": "R_id_trigger * F_fine",
}
EXPERIMENT_ORDER = list(EXPERIMENT_FORMULAS)

ROAD_BUFFER_M = 100.0
WATER_BUFFER_M = 200.0
OTHER_BUFFER_M = 2000.0
TEXT_MULTIPLIER = 2

# These are the exact defaults in spatial.py.
IMAGE_PROB_MAP_KIND = "linear"
IMAGE_K_ROAD = 0.7
IMAGE_K_WATER = 0.7
IMAGE_MULTIPLIER_LO = 0.75
IMAGE_MULTIPLIER_HI = 2.0
IMAGE_BASE_ROAD_WEIGHT = 1
IMAGE_BASE_WATER_WEIGHT = 1
IMAGE_JOINT_CAP = 2.5
LANDSLIDE_GATE_THRESHOLD = 0.5

SUS_NORMALIZATION = "global_minmax"
RAINFALL_METHOD = "R_id_trigger"
FUSION_METHOD = "weighted_soft_or"
PROCESSING_VERSION = "landslide_localization_noisyor_alltouched_v1"
NUMERICAL_PRECISION = "default"
RASTER_MASK_POLICY = "all_touched_source_mask_v1"
# Candidate geometry semantics are versioned separately from raster numerics.
CANDIDATE_REGION_POLICY_VERSION = "focusarea_no_local_buffer_v1"

# These values describe the already-built, fusion-independent upstream cache.
# Keeping them frozen preserves compatibility with the costly v5 candidate,
# text, hydro, semantic, S_norm, R_id_trigger, and F_fine artifacts.
UPSTREAM_CACHE_FUSION_METHOD = "soft_or"
UPSTREAM_CACHE_PROCESSING_VERSION = "landslide_localization_noisyor_alltouched_upstream_v1"

DEFAULT_CANDIDATE_FILE = ROOT / "data/full/process/candidates.json"
DEFAULT_GT_FILE = ROOT / "data/full/annotations.json"
DEFAULT_SUS_RASTER = ROOT / "data/external/susceptibility/n10_conus.tif"
DEFAULT_RAINFALL_DIR = ROOT / "data/external/prism_rainfall"
DEFAULT_IMAGE_SEMANTICS = ROOT / "data/full/process/image_semantics.json"
DEFAULT_GDB_PATH = ROOT / "data/external/water/NHDPlus_H_National_Release_2.gdb"
DEFAULT_RESULTS_DIR = SOFT_ROOT / "results"
DEFAULT_SPLITS_DIR = ROOT / "data/full/splits"
DEFAULT_DEVELOPMENT_SPLIT = DEFAULT_SPLITS_DIR / "development.json"
DEFAULT_TEST_SPLIT = DEFAULT_SPLITS_DIR / "test.json"
DEFAULT_SPLIT_SUMMARY = DEFAULT_SPLITS_DIR / "split_summary.json"

TIER_DISPLAY_NAMES = {
    "fine_area": "fine",
    "county_area": "county",
    "state_area": "state",
}

# Current-flow geometry constants.
POINT_BUFFER_M = 5000.0
LINE_BUFFER_M = 5000.0
MAX_CANDIDATE_AREA_KM2 = 500000.0
# Retain the historical fixed thresholds so old runs remain reproducible.
ID_TRIGGER_THRESHOLDS = {1: 20.0, 3: 14.0, 5: 10.0, 7: 8.0}

# The ablation runner defaults to the duration-dependent I-D threshold formula.
DEFAULT_ID_TRIGGER_THRESHOLD_MODE = "formula"
ID_TRIGGER_FORMULA_COEFFICIENT = 13.04
ID_TRIGGER_FORMULA_EXPONENT = -0.44
DEFAULT_RAIN_SIGMOID_SCALE = 0.6


def fixed_manifest() -> dict:
    """Return the fusion-independent v5 cache contract.

    Runtime Soft-OR weights intentionally do not belong here.  The runner adds
    them to a separate fusion signature and to cache/result metadata.
    """
    return {
        "numerical_precision": NUMERICAL_PRECISION,
        "raster_mask_policy": RASTER_MASK_POLICY,
        "road_buffer_m": ROAD_BUFFER_M,
        "water_buffer_m": WATER_BUFFER_M,
        "other_buffer_m": OTHER_BUFFER_M,
        "text_multiplier": TEXT_MULTIPLIER,
        "sus_normalization": SUS_NORMALIZATION,
        "rainfall_method": RAINFALL_METHOD,
        "fusion_method": UPSTREAM_CACHE_FUSION_METHOD,
        "processing_version": UPSTREAM_CACHE_PROCESSING_VERSION,
        "top_k_values": TOP_K_VALUES,
        "image_prob_map_kind": IMAGE_PROB_MAP_KIND,
        "image_k_road": IMAGE_K_ROAD,
        "image_k_water": IMAGE_K_WATER,
        "image_multiplier_lo": IMAGE_MULTIPLIER_LO,
        "image_multiplier_hi": IMAGE_MULTIPLIER_HI,
        "image_base_road_weight": IMAGE_BASE_ROAD_WEIGHT,
        "image_base_water_weight": IMAGE_BASE_WATER_WEIGHT,
        "image_joint_cap": IMAGE_JOINT_CAP,
        "landslide_gate_threshold": LANDSLIDE_GATE_THRESHOLD,
        "id_trigger_thresholds_mm_per_day": ID_TRIGGER_THRESHOLDS,
    }
