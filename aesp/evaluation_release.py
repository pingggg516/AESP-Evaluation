"""Frozen prediction evaluator, independent of any VLM inference interface.

The bundled lock is an integrity/reproducibility record, not a signature or a
claim that the answer extractor is infallible. No model loads during this check.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


EVALUATION_ID = "aesp-evaluator"
ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "assets" / "evaluation_lock.json"
SOURCE_FILES = (
    "__init__.py", "__main__.py", "cli.py", "evaluation_release.py",
    "extraction.py", "binary_guard.py", "extractor_identity.py", "generation_status.py", "io.py", "parsing.py",
    "pipeline.py", "regions.py", "scoring.py",
)
REFERENCE_CPU = {"python_minor": "3.10", "numpy": "2.2.6"}
SUPPORTED_EVALUATION_RUNTIMES = (REFERENCE_CPU,)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _source_digest(path):
    # Git/platform newline conversion must not change a Python source identity.
    return _digest(path.read_text(encoding="utf-8").encode("utf-8"))


def build_lock():
    """Maintainer helper; never called to silently update a user's saved lock."""
    from .extraction import LLMExtractor, prompt_metadata, REFERENCE_EXTRACTOR_RUNTIME
    from .extractor_identity import load_manifest
    metadata = LLMExtractor().metadata()
    settings = {k: v for k, v in metadata["settings"].items()
                if k not in ("device_map", "max_memory", "local_files_only")}
    assets = json.loads((ROOT / "assets" / "SHA256SUMS.json").read_text(encoding="utf-8"))
    resources = {"assets/" + name: _digest((ROOT / "assets" / name).read_bytes())
                 for name in assets}
    for name in ("assets/SHA256SUMS.json", "assets/extractor_checkpoint.json"):
        resources[name] = _digest((ROOT / name).read_bytes())
    load_manifest()
    return {
        "evaluation_id": EVALUATION_ID,
        "scope": "prediction_preparation_and_scoring",
        "source_hash_encoding": "utf8_universal_newlines",
        "source_sha256": {name: _source_digest(ROOT / name) for name in SOURCE_FILES},
        "resource_sha256": resources,
        "reference_cpu": REFERENCE_CPU,
        "supported_evaluation_runtimes": list(SUPPORTED_EVALUATION_RUNTIMES),
        "reference_extractor": REFERENCE_EXTRACTOR_RUNTIME,
        "answer_extraction": {
            "model": metadata["model"], "revision": metadata["revision"],
            "prompt_hashes": prompt_metadata(), "settings": settings,
            "decision_policy_sha256": metadata["decision_policy_sha256"],
            "checkpoint_manifest": "assets/extractor_checkpoint.json",
            "inputs": ["question", "task_type", "target_classes", "answer_text"],
            "image_or_gt_access": False,
            "binary_extraction": "binary answer extraction; closed-grammar semantic guard after valid Qwen A/B/Z output; raw generation preserved",
        },
        "external_submission": {
            "command": "python -m aesp evaluate",
            "workflow": ["validate_prediction_jsonl", "validate_original_pixel_boxes_and_extract_answers", "score"],
            "output_files": ["predictions.json", "extracted_answers.json", "case_results.json",
                             "metrics.json", "run_manifest.json"],
            "output_format": "UTF-8 JSON with two-space indentation and a final newline; sample files are arrays of objects; metrics and manifest are objects",
            "answer_cache": "optional extracted_answers.json array from a completed evaluation; IDs must match the submitted predictions; validated before reuse",
            "file_format": "UTF-8 JSONL (optional BOM); one object per unique nonblank string sample_id; every prediction ID must have a matching annotation; unused annotations are excluded from evaluation",
            "evaluation_samples": "submitted predictions only, processed in annotation order; metrics and run manifest record annotation, evaluated and unused annotation counts",
            "annotation_required_fields": ["sample_id", "question", "task_type", "source_dataset",
                "target_classes", "gt_answer", "all_relevant_gt_objects", "image_width", "image_height"],
            "gt_objects": "all_relevant_gt_objects: [{class: target_class, bbox: [x1,y1,x2,y2]}]; original image pixels",
            "raw_prediction_required_fields": ["sample_id", "raw_answer", "evidence_boxes"],
            "raw_prediction_regions": "only evidence_boxes; raw_answer coordinates never supply scoring boxes",
            "coordinate_contract": "original-image pixels, origin at upper left, x rightward, y downward; no unit option or coordinate conversion",
            "structured_boxes_format": "evidence_boxes: [[xmin,ymin,xmax,ymax], ...]; no boxes: []; four finite numbers, positive area, entirely within the original image; values preserved without clipping or reordering",
            "obsolete_coordinate_options": "unit/transform/region-format/raw-grounding/polygon fields rejected; caller prepares original-image pixel xyxy boxes",
            "answer_view": "unambiguous JSON language fields plus comments and surrounding prose; other responses preserved whole for the frozen extractor; no coordinate grammar required",
            "json_response_comments": "line/block comments in response JSON retained in source order with semantic fields; explanations and corrections preserved",
            "generation_status": "reject_explicit_failure_or_truncation_before_extraction_and_scoring; missing_metadata_unknown",
            "vlm_inference_required": False,
        },
        "scoring": {
            "tasks": ["existence", "counting", "comparison"],
            "coordinate_format": "original_image_continuous_xyxy",
            "localization_iou_threshold": 0.5,
            "localization_matching": "maximum_cardinality_one_to_one",
            "support": "each_GT_center_covered_by_any_predicted_box_counted_once",
            "evidence_subset": "exclude_GT_negative_existence",
            "AA": "answer_correct/all_samples",
            "Loc_P_R_F1": "micro_aggregate_on_evidence_subset",
            "SupportRate": "answer_supported/evidence_subset",
            "ESR": "correct_and_supported/correct_answers_in_evidence_subset",
            "Joint": "correct_and_supported/evidence_subset",
            "invalid_answers": "retained_in_denominators",
            "zero_denominator": None, "metric_scale": "percent",
        },
        "limits": [
            "No arbitrary-model inference compatibility claim.",
            "Scoring geometry comes only from submitted original-image pixel xyxy evidence_boxes.",
            "Frozen LLM extraction can make semantic errors.",
            "Source integrity does not authenticate externally supplied predictions or caches.",
        ],
    }


def verify_evaluation():
    """Fail before scoring if packaged evaluator sources/prompts have drifted."""
    from .io import read_json
    saved = read_json(LOCK_PATH)
    expected = build_lock()
    if saved != expected:
        changed = []
        for key in ("source_sha256", "resource_sha256"):
            before, after = saved.get(key, {}), expected[key]
            changed.extend(name for name in sorted(set(before) | set(after))
                           if before.get(name) != after.get(name))
        detail = ", ".join(changed) if changed else "protocol/settings/schema"
        raise ValueError("Frozen evaluator differs from its release lock: " + detail)
    return {"id": EVALUATION_ID, "lock_sha256": _digest(LOCK_PATH.read_bytes()),
            "integrity": "verified", "scope": saved["scope"]}


def runtime_versions():
    from .extraction import REFERENCE_EXTRACTOR_RUNTIME
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in REFERENCE_EXTRACTOR_RUNTIME:
        if package == "python_minor":
            continue
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["matches_reference_cpu"] = (
        ".".join(platform.python_version_tuple()[:2]) == REFERENCE_CPU["python_minor"]
        and versions["numpy"] == REFERENCE_CPU["numpy"])
    versions["supported_evaluation_runtime"] = any(
        ".".join(platform.python_version_tuple()[:2]) == profile["python_minor"]
        and versions["numpy"] == profile["numpy"]
        for profile in SUPPORTED_EVALUATION_RUNTIMES)
    return versions


def validate_evaluation_runtime(versions=None):
    """Require a recorded CPU environment before preparing or scoring results."""
    versions = runtime_versions() if versions is None else versions
    minor = ".".join(str(versions.get("python", "")).split(".")[:2])
    if not any(minor == profile["python_minor"] and
               versions.get("numpy") == profile["numpy"]
               for profile in SUPPORTED_EVALUATION_RUNTIMES):
        raise ValueError(
            "Unsupported evaluation environment: use Python 3.10 and run "
            "python -m pip install -r requirements.txt from the project root; got Python "
            + str(versions.get("python"))
            + ", NumPy " + str(versions.get("numpy")))
    return versions


def report(checkpoint=None):
    result = {"evaluation_protocol": verify_evaluation(), "runtime": runtime_versions(),
              "checkpoint": {"status": "not_checked"}}
    if checkpoint is not None:
        from .extractor_identity import verify_checkpoint
        result["checkpoint"] = verify_checkpoint(checkpoint)
    return result
