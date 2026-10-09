"""Deterministic AESP scoring on extracted answers and original-image regions.

This module performs no model inference and reads no files. Support is
conditioned on the model's answer, independently of answer correctness.
Prediction boxes use original-image pixels for localization and evidence support.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
import re
from typing import Any

import numpy as np

from .regions import validate_box, validate_prediction_regions


SUPPORT_RULE_ID = "answer_evidence_support"
TASKS = ("existence", "counting", "comparison")


def normalize_answer(value: Any, task: str) -> str | None:
    """Normalize an already extracted answer; unresolved/negative values fail.

    This is deliberately not a free-text answer extractor. In particular,
    contradictory prose, floats, booleans, and negative counts are INVALID.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task_type: {task!r}")
    if value is None or isinstance(value, bool):
        return None
    if task == "counting":
        if isinstance(value, Integral):
            return str(int(value)) if value >= 0 else None
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not re.fullmatch(r"[+-]?[0-9]+", value, re.ASCII):
            return None
        try:
            count = int(value)
        except ValueError:
            return None
        return str(count) if count >= 0 else None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in ("yes", "no") else None


def comparison_operator(sample: Mapping[str, Any]) -> str:
    """Use an explicit relation, or recognize the benchmark's question forms."""
    explicit = sample.get("comparison_operator")
    if explicit is not None:
        if explicit not in (">", "<", "="):
            raise ValueError("comparison_operator must be '>', '<', or '='")
        return explicit
    question = " ".join(str(sample.get("question", "")).lower().split())
    match = re.match(r"^are there (more|fewer|less)\b.+\bthan\b", question)
    if match:
        return ">" if match[1] == "more" else "<"
    if re.search(r"\b(same number|equal numbers?)\b", question):
        return "="
    raise ValueError("Comparison needs an explicit operator or a recognized question")


def _positive_dimension(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value <= 0
            or not float(value).is_integer()):
        raise ValueError(f"{name} must be a positive integer image dimension")
    return float(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _relation(a: int, b: int, operator: str) -> bool:
    if operator == ">":
        return a > b
    if operator == "<":
        return a < b
    if operator == "=":
        return a == b
    raise ValueError("Unknown comparison operator")


def _validate_sample(sample: Mapping[str, Any]) -> tuple[dict, str, str | None]:
    required = ("sample_id", "task_type", "source_dataset", "target_classes",
                "gt_answer", "all_relevant_gt_objects", "image_width", "image_height")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError("Annotation missing required fields: " + ", ".join(missing))
    _nonempty_string(sample["sample_id"], "sample_id")
    _nonempty_string(sample["source_dataset"], "source_dataset")
    task = sample["task_type"]
    if task not in TASKS:
        raise ValueError(f"Unknown task_type: {task!r}")
    width = _positive_dimension(sample["image_width"], "image_width")
    height = _positive_dimension(sample["image_height"], "image_height")
    classes = sample["target_classes"]
    expected = 2 if task == "comparison" else 1
    if not isinstance(classes, list) or len(classes) != expected:
        raise ValueError(f"{task} requires {expected} ordered target class(es)")
    for category in classes:
        _nonempty_string(category, "target class")
    if len(set(classes)) != len(classes):
        raise ValueError("target_classes must be unique")
    objects = sample["all_relevant_gt_objects"]
    if not isinstance(objects, list):
        raise ValueError("all_relevant_gt_objects must be a list")
    grouped = {category: [] for category in classes}
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping) or "class" not in obj or "bbox" not in obj:
            raise ValueError(f"GT object {index} needs class and bbox")
        if obj["class"] not in grouped:
            raise ValueError(f"GT object {index} has a class outside target_classes")
        grouped[obj["class"]].append(validate_box(obj["bbox"], width, height))
    truth = normalize_answer(sample["gt_answer"], task)
    if truth is None:
        raise ValueError("Invalid ground-truth answer")
    operator = comparison_operator(sample) if task == "comparison" else None
    totals = {category: len(value) for category, value in grouped.items()}
    if task == "counting":
        expected_truth = str(totals[classes[0]])
    elif task == "existence":
        expected_truth = "yes" if totals[classes[0]] else "no"
    else:
        expected_truth = "yes" if _relation(totals[classes[0]], totals[classes[1]], operator) else "no"
    if truth != expected_truth:
        raise ValueError("gt_answer disagrees with all_relevant_gt_objects")
    return grouped, truth, operator


def iou_matrix(pred: list, gt: list) -> np.ndarray:
    """Axis-aligned, continuous-coordinate IoU (no inclusive-pixel +1)."""
    a = np.asarray(pred, dtype=float).reshape(-1, 4)
    b = np.asarray(gt, dtype=float).reshape(-1, 4)
    lower = np.maximum(a[:, None, :2], b[None, :, :2])
    upper = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.maximum(upper - lower, 0)
    intersection = size[..., 0] * size[..., 1]
    union = (np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None]
             + np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :] - intersection)
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def maximum_match(matrix: np.ndarray, threshold: float = 0.5) -> int:
    """Maximum cardinality of the IoU-threshold bipartite graph.

    Deterministic augmenting paths give exactly the TP count obtained by
    maximizing the original binary adjacency matrix with linear assignment.
    No SciPy dependency, greedy approximation, or recursion depth limit.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Matching requires a finite two-dimensional matrix")
    if not matrix.size:
        return 0
    if not math.isfinite(threshold):
        raise ValueError("Matching threshold must be finite")
    adjacency = [np.flatnonzero(row >= threshold).tolist() for row in matrix]
    left_match = [-1] * matrix.shape[0]
    right_match = [-1] * matrix.shape[1]
    count = 0
    for start in range(matrix.shape[0]):
        parents = [-1] * matrix.shape[1]
        visited_left = {start}
        queue = deque([start])
        free_right = -1
        while queue and free_right < 0:
            left = queue.popleft()
            for right in adjacency[left]:
                if parents[right] >= 0:
                    continue
                parents[right] = left
                matched_left = right_match[right]
                if matched_left < 0:
                    free_right = right
                    break
                if matched_left not in visited_left:
                    visited_left.add(matched_left)
                    queue.append(matched_left)
        if free_right < 0:
            continue
        right = free_right
        while right >= 0:
            left = parents[right]
            previous_right = left_match[left]
            left_match[left] = right
            right_match[right] = left
            right = previous_right
        count += 1
    return count


def covered_counts(targets: dict[str, list], boxes: list) -> dict[str, int]:
    """Count each GT instance once if any prediction box contains its center."""
    rectangles = np.asarray(boxes, dtype=float).reshape(-1, 4)
    result = {}
    for category, gt in targets.items():
        if not gt or not boxes:
            result[category] = 0
            continue
        ground_truth = np.asarray(gt, dtype=float)
        centers = (ground_truth[:, :2] + ground_truth[:, 2:]) / 2
        hits = np.all((centers[None, :, :] >= rectangles[:, None, :2])
                      & (centers[None, :, :] <= rectangles[:, None, 2:]), axis=-1)
        result[category] = int(np.any(hits, axis=0).sum())
    return result


def _supported(task: str, classes: list[str], counts: dict[str, int],
               answer: str | None, operator: str | None) -> bool:
    if answer is None:
        return False
    if task == "counting":
        return counts[classes[0]] == int(answer)
    if task == "existence":
        return (counts[classes[0]] > 0) == (answer == "yes")
    return _relation(counts[classes[0]], counts[classes[1]], operator) == (answer == "yes")


def score_case(sample: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict:
    """Score one annotation/prediction pair; malformed geometry raises ValueError.

    ``pred_answer`` must already be extracted. ``None`` explicitly means an
    invalid/unresolved semantic answer. Missing prediction fields are errors,
    not implicit empty evidence or implicit invalid answers. ``model`` is
    optional result metadata and never changes the mathematical protocol.
    Question/image metadata comes from the annotation. Coordinates must be
    original-image pixel xyxy boxes; no coordinate conversion is performed.
    """
    if not isinstance(sample, Mapping) or not isinstance(prediction, Mapping):
        raise ValueError("sample and prediction must be mappings")
    grouped, truth, operator = _validate_sample(sample)
    missing = [key for key in ("sample_id", "pred_answer", "evidence_boxes")
               if key not in prediction]
    if missing:
        raise ValueError("Prediction missing required fields: " + ", ".join(missing))
    if prediction["sample_id"] != sample["sample_id"]:
        raise ValueError("Prediction sample_id does not match annotation")
    boxes = validate_prediction_regions(
        prediction, sample["image_width"], sample["image_height"])
    task, classes = sample["task_type"], sample["target_classes"]
    answer = normalize_answer(prediction["pred_answer"], task)
    eligible = not (task == "existence" and truth == "no")
    gt = [box for category in classes for box in grouped[category]]
    totals = {category: len(grouped[category]) for category in classes}
    counts = covered_counts(grouped, boxes) if eligible else {}
    supported = int(eligible and _supported(task, classes, counts, answer, operator))
    correct = int(answer is not None and answer == truth)
    matrix = iou_matrix(boxes, gt) if eligible else np.empty((0, 0))
    result = {
        "sample_id": sample["sample_id"], "task_type": task,
        "source_dataset": sample["source_dataset"], "target_classes": list(classes),
        "gt_answer": truth, "pred_answer": answer,
        "semantic_status": "INVALID" if answer is None else "extracted",
        "answer_correct": correct, "evidence_eligible": int(eligible),
        # Evidence support depends on answer content, independently of correctness.
        "support_without_answer": supported, "evidence_supported": supported,
        "support_rule": SUPPORT_RULE_ID,
        "joint_success": int(correct and supported),
        "covered_target_counts": counts, "target_totals": totals,
        "relevant_gt_count": len(gt), "predicted_box_count": len(boxes),
        "loc_tp": maximum_match(matrix, 0.5) if eligible else 0,
        "loc_pred": len(boxes) if eligible else 0,
        "loc_gt": len(gt) if eligible else 0,
        "loc_tp_0.25": maximum_match(matrix, 0.25) if eligible else 0,
        "loc_tp_0.75": maximum_match(matrix, 0.75) if eligible else 0,
    }
    if "model" in prediction:
        result["model"] = prediction["model"]
    if task == "comparison":
        a, b = (totals[c] for c in classes)
        result["comparison_operator"] = operator
        result["relation"] = ">" if a > b else "<" if a < b else "="
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return 100 * numerator / denominator if denominator else None


def _metrics(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["evidence_eligible"]]
    correct = sum(row["answer_correct"] for row in rows)
    eligible_correct = sum(row["answer_correct"] for row in eligible)
    support = sum(row["support_without_answer"] for row in eligible)
    joint = sum(row["joint_success"] for row in eligible)
    tp = sum(row["loc_tp"] for row in eligible)
    pred = sum(row["loc_pred"] for row in eligible)
    gt = sum(row["loc_gt"] for row in eligible)
    return {
        "N_all": len(rows), "N_eligible": len(eligible),
        "N_correct_eligible": eligible_correct, "AA": _ratio(correct, len(rows)),
        "Loc_P": _ratio(tp, pred), "Loc_R": _ratio(tp, gt),
        "Loc_F1": _ratio(2 * tp, pred + gt),
        "SupportRate": _ratio(support, len(eligible)),
        "ESR": _ratio(joint, eligible_correct), "Joint": _ratio(joint, len(eligible)),
        "Unsupported_at_Correct": _ratio(eligible_correct - joint, eligible_correct),
        "answer_correct": correct, "supported_correct": joint,
        "unsupported_correct": eligible_correct - joint,
        "invalid_answers": sum(row["pred_answer"] is None for row in rows),
        "loc_tp": tp, "loc_pred": pred, "loc_gt": gt,
    }


def _density(n: int) -> str:
    return ("0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3
            else "4-7" if n <= 7 else "8-15" if n <= 15 else "16+")


def summarize(cases: Iterable[dict]) -> dict[str, dict]:
    """Return grouped metrics for one prediction set.

    Metrics are percentages; a zero denominator produces ``None``. Duplicate
    sample IDs are rejected instead of double weighting cases or pooling models.
    """
    groups = defaultdict(list)
    groups["overall"] = []
    seen = set()
    for row in cases:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate scored sample_id: {sample_id}")
        seen.add(sample_id)
        task, source = row["task_type"], row["source_dataset"]
        keys = ["overall", "task:" + task, "source:" + source,
                "task_source:" + task + "|" + source]
        if row["evidence_eligible"]:
            keys.append("density:" + _density(row["relevant_gt_count"]))
        if task == "counting":
            keys.append("count:" + row["gt_answer"])
        if task == "comparison":
            keys.append("relation:" + row["relation"])
        for key in keys:
            groups[key].append(row)
    return {key: _metrics(rows) for key, rows in groups.items()}
