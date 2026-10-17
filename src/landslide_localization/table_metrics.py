"""Calculate paper tables from completed Full, ablation, and Direct results."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median

TOP_K = (1, 5, 10, 15, 20, 30, 40, 50)
DIRECT = "direct_llm"
FULL = "E5_full_softor_fine"
ABLATIONS = ("E1_sus", "E2_rain", "E3_sus_rain_softor", "E4_sus_fine", FULL, "E6_rain_fine")
LABELS = dict(zip(ABLATIONS, ("SUSC", "RAIN", "SUSC+RAIN", "SUSC+GEO", "FULL", "RAIN+GEO")))
LABELS[DIRECT] = "Direct LLM"
CONFIDENCE = {1.0: "High", 0.75: "Medium", 0.5: "Low", 0.25: "Very low"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def case_id(row):
    value = row.get("case_id", row.get("tweets_id")) if isinstance(row, dict) else row
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError(f"Case IDs must be exact nonempty strings or integers: {value!r}")
    return str(value)


def index_rows(rows):
    result = {}
    for row in rows:
        cid = case_id(row)
        if cid in result:
            raise ValueError(f"Duplicate case ID: {cid}")
        result[cid] = row
    return result


def split_ids(payload):
    if isinstance(payload, dict):
        payload = next((payload[key] for key in ("case_ids", "ids", "records") if key in payload), None)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Test split must contain a nonempty list of case IDs")
    return list(index_rows(payload))


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def valid(row):
    return row["gt_inside_region"] is True and row["gt_in_valid_raster"] is True


def percentage(count, denominator):
    return 100.0 * count / denominator if denominator else None


def numeric_stats(values):
    values = [v for v in values if finite_number(v)]
    return {"n": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None}


def summarize_stage1(rows):
    n = len(rows)
    matched = sum(r["evaluable"] for r in rows)
    inside = sum(r["in_region"] for r in rows)
    return {"n_test": n, "n_matched": matched, "n_inside": inside,
            "n_unmatched": n - matched, "n_outside": matched - inside,
            "matched_percent": percentage(matched, n),
            "inclusion_percent": percentage(inside, n),
            "conditional_inclusion_percent": percentage(inside, matched)}


def summarize_stage2(rows):
    good = [row for row in rows if valid(row)]
    output = {"n_test": len(rows), "n_valid": len(good),
              "PR": numeric_stats(r["PR"] for r in good), "hit": {}, "area_km2": {}}
    for k in TOP_K:
        hits = sum(row[f"HitTop{k}"] for row in good)
        output["hit"][str(k)] = {"count": hits,
            "conditional_percent": percentage(hits, len(good)),
            "end_to_end_percent": percentage(hits, len(rows))}
        output["area_km2"][str(k)] = numeric_stats(row.get(f"AreaTop{k}_km2") for row in good)
    return output


def validate_model_rows(rows, ids, stage):
    models = {}
    for row in rows:
        model = row.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Every metric row must have a model name")
        models.setdefault(model, []).append(row)
    if not models:
        raise ValueError(f"Stage {stage} metrics are empty")
    indexed = {}
    for model, model_rows in models.items():
        by_id = index_rows(model_rows)
        if set(by_id) != set(ids):
            raise ValueError(f"Stage {stage} model {model} does not cover exactly the supplied test split")
        for row in model_rows:
            if stage == 1:
                if type(row.get("evaluable")) is not bool or type(row.get("in_region")) is not bool:
                    raise ValueError("Stage I evaluable and in_region must be booleans")
                if row["in_region"] and not row["evaluable"]:
                    raise ValueError("An unmatched Stage I record cannot be inside its region")
            else:
                for field in ("gt_inside_region", "gt_in_valid_raster"):
                    if type(row.get(field)) is not bool:
                        raise ValueError(f"{field} must be boolean")
                if row["gt_in_valid_raster"] and not row["gt_inside_region"]:
                    raise ValueError("Raster-valid GT must be inside the candidate region")
                if valid(row) and not (finite_number(row.get("PR")) and 0 <= row["PR"] <= 1):
                    raise ValueError("Every valid case must have a finite PR in [0, 1]")
                previous = False
                for k in TOP_K:
                    hit = row.get(f"HitTop{k}")
                    if type(hit) is not bool or (hit and not valid(row)):
                        raise ValueError("Hit must be boolean; invalid cases must be misses")
                    if previous and not hit:
                        raise ValueError("Hits must be monotone as K increases")
                    previous = hit
        indexed[model] = by_id
    return indexed


def build_tables(stage1_rows, stage2_rows, ids, annotations=None):
    """Return all tables without altering input rows or selecting on Hit outcomes."""
    s1 = validate_model_rows(stage1_rows, ids, 1)
    s2 = validate_model_rows(stage2_rows, ids, 2)
    if not {DIRECT, "full"} <= s1.keys() or not {DIRECT, FULL} <= s2.keys():
        raise ValueError("Both Direct LLM and FULL must be present in both stages")
    for stage2_model, stage1_model in ((DIRECT, DIRECT), (FULL, "full")):
        for cid in ids:
            if valid(s2[stage2_model][cid]) and not s1[stage1_model][cid]["in_region"]:
                raise ValueError(f"Stage-II valid GT is not inside its Stage-I region: {stage2_model}/{cid}")
    annotations = annotations or {}
    if annotations and not set(ids) <= annotations.keys():
        raise ValueError("Annotations do not contain every supplied test ID")
    for model_rows in s2.values():
        for cid, row in model_rows.items():
            confidence = annotations.get(cid, {}).get("coordinates_confidence")
            if confidence is not None and row.get("coordinates_confidence") is not None and confidence != row["coordinates_confidence"]:
                raise ValueError(f"Annotation confidence differs from result metadata: {cid}")
    common = [cid for cid in ids if valid(s2[DIRECT][cid]) and valid(s2[FULL][cid])]
    common_stats = {}
    own = {}
    for model in (DIRECT, FULL):
        own[model] = summarize_stage2([s2[model][cid] for cid in ids])
        paired = summarize_stage2([s2[model][cid] for cid in common])
        # End-to-end always uses the complete test split, never the intersection.
        common_stats[model] = {"n_common": len(common), "PR": paired["PR"],
            "hit": {str(k): {"count": paired["hit"][str(k)]["count"],
                             "percent": paired["hit"][str(k)]["conditional_percent"]} for k in TOP_K},
            "end_to_end": {str(k): {"n_test": len(ids), "count": own[model]["hit"][str(k)]["count"],
                                    "percent": own[model]["hit"][str(k)]["end_to_end_percent"]} for k in TOP_K}}
    ablations = {model: summarize_stage2([s2[model][cid] for cid in ids]) for model in ABLATIONS if model in s2}
    confidence_tables, scales = {}, {}
    for model, model_rows in s2.items():
        confidence_tables[model], scales[model] = {}, {}
        confidence_groups, scale_groups = {}, {}
        for cid in ids:
            row = model_rows[cid]
            confidence = annotations.get(cid, {}).get("coordinates_confidence", row.get("coordinates_confidence"))
            group = CONFIDENCE.get(confidence, "Unknown")
            confidence_groups.setdefault(group, []).append(row)
            scale_groups.setdefault(row.get("candidate_area_tier") or "unknown", []).append(row)
        for group, rows in confidence_groups.items():
            confidence_tables[model][group] = summarize_stage2(rows)
        for group, rows in scale_groups.items():
            scales[model][group] = summarize_stage2(rows)
    ablation_common = [cid for cid in ids if all(valid(s2[model][cid]) for model in ablations)] if ablations else []
    return {"schema_version": 1, "n_test": len(ids),
        "definitions": {
            "PR": "Percentile rank, not precision/recall; averaged only over valid cases.",
            "valid": "GT inside the candidate region AND GT in valid scoring raster.",
            "common_cases": "Intersection of Direct LLM and FULL valid case IDs, fixed for every K and PR; not selected on Top-K Hit.",
            "top_k": "Top K percent of each model's own valid candidate raster, with inclusive percentile ties.",
            "end_to_end": "All supplied test IDs; unmatched, outside-region and raster-invalid cases count as misses.",
            "area": "Mean/median selected raster area in square kilometres over the same valid cases.",
            "scale": "Each model's own candidate_area_tier; cross-model scale groups need not contain identical IDs.",
            "execution": "Aggregation of completed per-case scores; no model inference, geocoding or raster scoring."},
        "stage1": {model: summarize_stage1([rows[cid] for cid in ids]) for model, rows in s1.items()},
        "stage2_common": {"case_ids": common, "n_common": len(common), "models": common_stats},
        "stage2_conditional": own, "ablations": ablations,
        "ablation_common_case_ids": ablation_common,
        "ablation_valid_populations_identical": all({cid for cid in ids if valid(s2[model][cid])} == set(ablation_common) for model in ablations),
        "confidence": confidence_tables, "scale": scales}



def flatten_stats(model, stats, **groups):
    row = {**groups, "model": model, "label": LABELS.get(model, model),
           "n_test": stats["n_test"], "n_valid": stats["n_valid"],
           "PR_mean": stats["PR"]["mean"], "PR_median": stats["PR"]["median"]}
    for k in TOP_K:
        hit, area = stats["hit"][str(k)], stats["area_km2"][str(k)]
        row.update({f"HitTop{k}_count": hit["count"], f"ConditionalHitTop{k}_percent": hit["conditional_percent"],
                    f"EndToEndHitTop{k}_percent": hit["end_to_end_percent"],
                    f"AreaTop{k}_mean_km2": area["mean"], f"AreaTop{k}_median_km2": area["median"]})
    return row


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "stage1.csv", [{"model": model, **stats} for model, stats in result["stage1"].items()])
    rows = []
    for model, stats in result["stage2_common"]["models"].items():
        row = {"model": model, "label": LABELS[model], "n_common": stats["n_common"],
               "PR_mean": stats["PR"]["mean"], "PR_median": stats["PR"]["median"], "n_test": result["n_test"]}
        for k in TOP_K:
            row.update({f"CommonHitTop{k}_count": stats["hit"][str(k)]["count"],
                        f"CommonHitTop{k}_percent": stats["hit"][str(k)]["percent"],
                        f"EndToEndHitTop{k}_count": stats["end_to_end"][str(k)]["count"],
                        f"EndToEndHitTop{k}_percent": stats["end_to_end"][str(k)]["percent"]})
        rows.append(row)
    write_csv(output / "stage2_common_and_end_to_end.csv", rows)
    write_csv(output / "stage2_conditional.csv", [flatten_stats(m, s) for m, s in result["stage2_conditional"].items()])
    write_csv(output / "ablations.csv", [flatten_stats(m, s) for m, s in result["ablations"].items()])
    for key in ("confidence", "scale"):
        write_csv(output / f"{key}.csv", [flatten_stats(m, s, group=g) for m, groups in result[key].items() for g, s in groups.items()])
    write_csv(output / "common_case_ids.csv", [{"case_id": cid} for cid in result["stage2_common"]["case_ids"]])
