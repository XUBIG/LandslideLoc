#!/usr/bin/env python3
"""Build the paper tables from Full, ablation, and Direct result files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_localization import table_metrics as tables
STAGE2_FIELDS = ("candidate_area_tier", "candidate_area_km2", "gt_inside_region",
                 "gt_in_valid_raster", "has_prediction", "score_at_gt", "PR", "point_error_km")
def require(condition, message):
    if not condition:
        raise ValueError(message)



def normalized_stage1(rows, model, annotations):
    output = []
    for row in rows:
        cid = tables.case_id(row)
        require(cid in annotations, f"Missing annotation: {cid}")
        require(row.get("model", model) == model, f"Unexpected Stage I model in {model} run")
        output.append({"case_id": cid, "model": model,
                       "evaluable": row.get("evaluable"), "in_region": row.get("in_region"),
                       "tier": row.get("tier"),
                       "coordinates_confidence": annotations[cid].get("coordinates_confidence")})
    return output


def normalized_stage2(rows, allowed_models, annotations):
    output = []
    for row in rows:
        cid = tables.case_id(row)
        require(cid in annotations, f"Missing annotation: {cid}")
        model = row.get("experiment_id", row.get("model"))
        require(model in allowed_models, f"Unexpected Stage II model: {model}")
        confidence = annotations[cid].get("coordinates_confidence")
        if row.get("coordinates_confidence") is not None:
            require(row["coordinates_confidence"] == confidence, f"Annotation confidence changed: {cid}")
        converted = {"case_id": cid, "model": model, **{key: row.get(key) for key in STAGE2_FIELDS},
                     "coordinates_confidence": confidence}
        for k in tables.TOP_K:
            for field in (f"HitTop{k}", f"AreaTop{k}_km2", f"AreaTop{k}_percent"):
                converted[field] = row.get(field)
        output.append(converted)
    return output


def collect(full, direct, ids, annotations):
    """Read per-case results and calculate the paper tables."""
    full, direct = Path(full), Path(direct)
    if not (direct / "stage1_test.json").is_file():
        direct = direct / "direct_only"
    stage1 = normalized_stage1(tables.read_json(full / "stage1_test.json"), "full", annotations)
    stage1 += normalized_stage1(tables.read_json(direct / "stage1_test.json"), tables.DIRECT, annotations)
    stage2 = normalized_stage2(tables.read_json(full / "ablation_case_metrics.json"), set(tables.ABLATIONS), annotations)
    stage2 += normalized_stage2(tables.read_json(direct / "stage2_test.json"), {tables.DIRECT}, annotations)
    models = {row["model"] for row in stage2}
    require(tables.FULL in models, "Full run must include E5 to compare with Direct")
    require(set(tables.ABLATIONS) <= models, "All six ablations are required; run scripts/run_full.py")
    stage1.sort(key=lambda row: (row["model"], row["case_id"]))
    stage2.sort(key=lambda row: (row["model"], row["case_id"]))
    return tables.build_tables(stage1, stage2, ids, annotations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-results", type=Path, default=ROOT / "outputs/full")
    parser.add_argument("--direct-results", type=Path, default=ROOT / "outputs/direct")
    parser.add_argument("--test-split", type=Path, default=ROOT / "data/public/splits/test.json")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/public/annotations.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/recomputed_tables")
    args = parser.parse_args(argv)
    ids = tables.split_ids(tables.read_json(args.test_split))
    annotation_rows = tables.read_json(args.annotations)
    require(isinstance(annotation_rows, list) and bool(annotation_rows), "Annotations must be a nonempty list")
    annotations = tables.index_rows(annotation_rows)
    results = collect(args.full_results, args.direct_results, ids, annotations)
    tables.write_outputs(results, args.output)
    print(json.dumps({"n_test": len(ids), "n_common": results["stage2_common"]["n_common"],
                      "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
