from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from landslide_localization.soft_config import (
    DEFAULT_DEVELOPMENT_SPLIT,
    DEFAULT_SPLIT_SUMMARY,
    DEFAULT_TEST_SPLIT,
)


def safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "custom"


def _ids_from_document(document: Any, path: Path) -> list[str]:
    if isinstance(document, list):
        raw_ids = document
    elif isinstance(document, dict):
        raw_ids = None
        for key in ("case_ids", "sampled_tweets_ids", "ids"):
            if isinstance(document.get(key), list):
                raw_ids = document[key]
                break
        if raw_ids is None and isinstance(document.get("records"), list):
            raw_ids = [row.get("tweets_id") for row in document["records"]]
        if raw_ids is None:
            raise ValueError(
                f"Split file {path} must contain case_ids, sampled_tweets_ids, ids, or records"
            )
    else:
        raise ValueError(f"Split file must contain a JSON array or object: {path}")

    ids = [str(value) for value in raw_ids]
    if any(not value or value == "None" for value in ids):
        raise ValueError(f"Split file contains a missing case ID: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Split file contains duplicate case IDs: {path}")
    return ids


def load_split_document(path: Path) -> tuple[dict[str, Any], list[str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    ids = _ids_from_document(document, path)
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    for name, frozen_path in (
        ("development", DEFAULT_DEVELOPMENT_SPLIT),
        ("test", DEFAULT_TEST_SPLIT),
    ):
        if path.resolve() == frozen_path.resolve():
            validate_event_split_document(document, path, name)
            break
    return {key: value for key, value in metadata.items() if not key.endswith("_sha256")}, ids


def current_split_summary() -> dict[str, Any]:
    """Read the supplied partition summary."""
    document = json.loads(DEFAULT_SPLIT_SUMMARY.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Split summary must be an object: {DEFAULT_SPLIT_SUMMARY}")
    return document


def validate_event_split_document(
    document: Any, path: Path, split: str
) -> dict[str, Any]:
    """Validate selected IDs and complete event groups in the supplied partition."""
    if split not in ("development", "test"):
        raise ValueError(f"Unsupported frozen split {split!r}")

    def fail(reason: str) -> None:
        raise ValueError(f"Invalid event-grouped {split} split {path}: {reason}")

    summary = current_split_summary()
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
        fail("missing split metadata")
    metadata = document["metadata"]
    ids = _ids_from_document(document, path)
    id_set = set(ids)
    if metadata.get("split") != split or len(ids) != summary.get(f"{split}_count"):
        fail("split name or population size does not match the frozen summary")

    version = summary.get("event_grouping_version")
    if not isinstance(version, str) or not version:
        fail("the active summary has no event-grouping version")
    if metadata.get("event_grouping_version") != version:
        fail("event_grouping_version does not match the current frozen summary")

    manifest_path = DEFAULT_SPLIT_SUMMARY.parent / "event_groups.json"
    if not manifest_path.is_file():
        fail(f"event-group manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("event_grouping_version") != version:
        fail("event-group manifest version does not match the current summary")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        fail("event-group manifest has no groups")
    group_by_case: dict[str, str] = {}
    seen_groups: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            fail("event-group manifest contains a malformed group")
        group_id = group.get("event_group_id")
        members = group.get("case_ids")
        if not isinstance(group_id, str) or not group_id or group_id in seen_groups:
            fail("event-group IDs are missing or duplicated")
        if not isinstance(members, list) or not members:
            fail(f"event group {group_id} has no members")
        seen_groups.add(group_id)
        members = [str(value) for value in members]
        if any(not value or value == "None" or value in group_by_case for value in members):
            fail("manifest case IDs are missing or assigned more than once")
        if len(members) != len(set(members)):
            fail("manifest case IDs are duplicated within an event group")
        group_by_case.update({case_id: group_id for case_id in members})
        overlap = id_set.intersection(members)
        if overlap and overlap != set(members):
            fail(f"event group {group_id} crosses the split boundary")
    if not id_set.issubset(group_by_case):
        fail("selected case IDs are missing from the event-group manifest")
    other_path = DEFAULT_TEST_SPLIT if split == "development" else DEFAULT_DEVELOPMENT_SPLIT
    other_ids = set(_ids_from_document(json.loads(other_path.read_text(encoding="utf-8")), other_path))
    if id_set.intersection(other_ids):
        fail("development and test case IDs overlap")
    if id_set.union(other_ids) != set(group_by_case):
        fail("development and test case IDs must cover the event-group manifest")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != len(ids):
        fail("records do not match the selected case IDs")
    record_ids = [str(row.get("tweets_id")) for row in records if isinstance(row, dict)]
    if len(record_ids) != len(ids) or len(set(record_ids)) != len(ids) or set(record_ids) != id_set:
        fail("record IDs differ from the selected case IDs")
    if any(row.get("event_group_id") != group_by_case[str(row["tweets_id"])] for row in records):
        fail("record event-group assignments differ from the manifest")
    return {
        f"{split}_case_ids": sorted(ids),
        "event_grouping_version": version,
        "event_group_ids": sorted({group_by_case[case_id] for case_id in ids}),
    }


@dataclass(frozen=True)
class SplitSelection:
    name: str
    output_name: str
    split_file: Optional[Path]
    case_ids: list[str]
    requested_case_count: int
    source_case_count: int
    metadata: dict[str, Any]


def select_cases(
    cases: Sequence[Mapping[str, Any]],
    split: str = "full",
    split_file: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> tuple[list[Mapping[str, Any]], SplitSelection]:
    """Select the supplied partition while preserving source record order."""
    indexed: dict[str, Mapping[str, Any]] = {}
    ordered_ids: list[str] = []
    for case in cases:
        case_id = str(case.get("tweets_id"))
        if not case_id or case_id == "None":
            raise ValueError("Source dataset contains a missing tweets_id")
        if case_id in indexed:
            raise ValueError(f"Source dataset contains duplicate tweets_id={case_id}")
        indexed[case_id] = case
        ordered_ids.append(case_id)

    if split_file is not None:
        path = split_file.resolve()
        metadata, requested_ids = load_split_document(path)
        name = run_name or safe_run_name(path.stem)
        output_name = safe_run_name(name)
        source_path = path
    elif split == "full":
        metadata = {"definition": "all records in the source candidate/ground-truth dataset"}
        requested_ids = ordered_ids
        name = "full"
        output_name = "full"
        source_path = None
    else:
        if split == "development":
            source_path = DEFAULT_DEVELOPMENT_SPLIT.resolve()
        elif split == "test":
            source_path = DEFAULT_TEST_SPLIT.resolve()
        else:
            raise ValueError(f"Unsupported split {split!r}")
        metadata, requested_ids = load_split_document(source_path)
        name = split
        output_name = split

    unknown = sorted(set(requested_ids).difference(indexed))
    if unknown:
        preview = ", ".join(unknown[:10])
        raise ValueError(
            f"Split contains {len(unknown)} IDs absent from the source dataset: {preview}"
        )
    requested_set = set(requested_ids)
    selected = [indexed[case_id] for case_id in ordered_ids if case_id in requested_set]
    if len(selected) != len(requested_ids):
        raise AssertionError("Selected case count differs from the validated split ID count")
    selection = SplitSelection(
        name=name,
        output_name=output_name,
        split_file=source_path,
        case_ids=sorted(requested_ids),
        requested_case_count=len(requested_ids),
        source_case_count=len(cases),
        metadata=metadata,
    )
    return selected, selection


