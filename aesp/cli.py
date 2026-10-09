"""Portable command line entry points; outputs are never overwritten."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from .io import aligned, indexed, new_output, read_json, read_jsonl, sha256, write_json


def evaluation_code_fingerprint():
    from .evaluation_release import SOURCE_FILES, _source_digest, LOCK_PATH
    root = Path(__file__).resolve().parent
    value = {"source_sha256": {name: _source_digest(root / name) for name in SOURCE_FILES},
             "evaluation_lock_sha256": sha256(LOCK_PATH),
             "extraction_assets_manifest_sha256": sha256(root / "assets" / "SHA256SUMS.json")}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), **value}


def run_evaluate(args):
    from .extraction import LLMExtractor
    from .pipeline import answer_row, cached_decision, normalize_one, summarize_inference_provenance
    from .scoring import _validate_sample, score_case, summarize
    from .evaluation_release import verify_evaluation, validate_evaluation_runtime
    evaluation_protocol = verify_evaluation()
    runtime = validate_evaluation_runtime()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.batch_size != 4:
        raise ValueError("The frozen answer extraction protocol requires --batch-size 4")
    annotations = indexed(read_jsonl(args.annotations), "annotations")
    for sample in annotations.values():
        _validate_sample(sample)
        if not isinstance(sample.get("question"), str) or not sample["question"].strip():
            raise ValueError("Annotation question must be a nonempty string")
    predictions = indexed(read_jsonl(args.predictions), "raw predictions")
    aligned(annotations, predictions, "raw predictions")
    inference_provenance = summarize_inference_provenance(predictions.values())
    labels = [p.get("model", args.model or "custom") for p in predictions.values()]
    if any(not isinstance(name, str) or not name.strip() for name in labels):
        raise ValueError("model must be a nonempty string")
    names = set(labels)
    if len(names) != 1 or (args.model is not None and names != {args.model}):
        raise ValueError("Evaluate one model per invocation; model labels disagree")
    model = next(iter(names))
    answers = None
    if args.answers is not None:
        rows = read_json(args.answers)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Answer cache must be the extracted_answers.json array from a completed evaluation")
        answers = indexed(rows, "answer cache")
        aligned(annotations, answers, "answer cache")
    input_hashes = {
        "annotations_sha256": sha256(args.annotations),
        "raw_predictions_sha256": sha256(args.predictions),
        "answer_cache_sha256": sha256(args.answers) if args.answers else None,
    }
    normalized, records, decisions, pending = [], [], [], []
    for sid, sample in annotations.items():
        try:
            pred, record, decision = normalize_one(sample, predictions[sid], answers[sid] if answers else None, args.model)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Sample {sid}: {exc}") from exc
        normalized.append(pred); records.append(record); decisions.append(decision)
        if decision is None:
            pending.append(len(records) - 1)
    metadata = None
    if pending:
        max_memory = json.loads(args.max_memory) if args.max_memory else None
        if max_memory is not None and not isinstance(max_memory, dict):
            raise ValueError("--max-memory must be a JSON object")
        extractor = LLMExtractor(model_path=args.llm_model, device_map=args.device_map,
                                 max_memory=max_memory, local_files_only=not args.allow_download,
                                 batch_size=args.batch_size)
        outputs = extractor.extract([records[i] for i in pending])
        if len(outputs) != len(pending):
            raise ValueError("Extractor returned an incomplete batch")
        for index, output in zip(pending, outputs):
            value = cached_decision(records[index], output)
            normalized[index]["pred_answer"] = value["answer"]
            decisions[index] = {**output, **answer_row(records[index], value, "frozen_llm")}
        metadata = extractor.metadata()
    cases = [score_case(annotations[pred["sample_id"]], pred) for pred in normalized]
    report = {"model": model, "groups": summarize(cases), "protocol": "AESP",
              "inference_provenance": inference_provenance,
              "evaluation_protocol": evaluation_protocol}
    extraction_identities = {}
    for row in decisions:
        identity = row.get("extractor_identity")
        if identity is not None:
            key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False).encode("utf-8")).hexdigest()
            entry = extraction_identities.setdefault(key, {
                "identity": identity, "new_extractions": 0, "reused_records": 0})
            entry["new_extractions" if row["route"] == "frozen_llm" else "reused_records"] += 1
    routes = dict(Counter(row["route"] for row in decisions))
    implementation = evaluation_code_fingerprint()
    out = new_output(args.output)
    write_json(out / "predictions.json", normalized)
    write_json(out / "extracted_answers.json", decisions)
    write_json(out / "case_results.json", cases)
    write_json(out / "metrics.json", report)
    output_hashes = {name + "_sha256": sha256(out / (name + ".json"))
                     for name in ("predictions", "extracted_answers", "case_results", "metrics")}
    write_json(out / "run_manifest.json", {
        "mode": "evaluate", "records": len(cases), "model": model, "routes": routes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **input_hashes, **output_hashes, "extractor": metadata,
        "answer_extraction_provenance": {
            "new_llm_extractions": len(pending),
            "reused_cache_records": routes.get("reused_cache", 0),
            "direct_rule_records": routes.get("strict", 0),
            "recorded_extractor_identities": extraction_identities,
            "cache_execution_authenticated": False,
            "cache_validation": "record consistency; not proof of prior execution",
        },
        "inference_provenance": inference_provenance,
        "evaluation_implementation": implementation,
        "evaluation_protocol": evaluation_protocol, "runtime": runtime,
        "python": sys.version, "command": sys.argv,
    })
    print(json.dumps(report["groups"]["overall"], ensure_ascii=False, indent=2))


def run_evaluation_spec(args):
    from .evaluation_release import LOCK_PATH, verify_evaluation
    identity = verify_evaluation()
    print(json.dumps({**read_json(LOCK_PATH), "identity": identity}, ensure_ascii=False, indent=2))


def run_verify_evaluation(args):
    from .evaluation_release import report
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    result = report(args.checkpoint)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="AESP: answer, localization, evidence support and joint success")
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("evaluation-spec", help="Print the independent frozen prediction evaluator")
    spec.set_defaults(func=run_evaluation_spec)
    verify = commands.add_parser("verify-evaluation", help="Verify evaluator integrity and optional local Qwen3 checkpoint; no inference")
    verify.add_argument("--checkpoint", type=Path, help="Hash an existing frozen Qwen3-14B checkpoint; never download weights")
    verify.add_argument("--output", type=Path, help="Optional new JSON report file; must not exist")
    verify.set_defaults(func=run_verify_evaluation)
    evaluate = commands.add_parser("evaluate", help="Extract language answers and score submitted original-image pixel xyxy boxes in one run")
    evaluate.add_argument("--annotations", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path, help="UTF-8 JSONL; each record requires sample_id, raw_answer and evidence_boxes; boxes must be original-image pixel [xmin,ymin,xmax,ymax]; use [] for no boxes")
    evaluate.add_argument("--output", required=True, type=Path, help="New output directory; must not exist")
    evaluate.add_argument("--model", help="Optional expected model name")
    evaluate.add_argument("--answers", type=Path, help="Optional extracted_answers.json from a completed evaluation; input, prompt and generation evidence must match")
    evaluate.add_argument("--llm-model", default="Qwen/Qwen3-14B", help="Local Qwen3-14B directory or official ID (default: Qwen/Qwen3-14B); cached/local files only unless --allow-download")
    evaluate.add_argument("--device-map", default="auto")
    evaluate.add_argument("--max-memory", help='Optional JSON device map, e.g. {"0":"21GiB","1":"21GiB"}')
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--allow-download", action="store_true", help="Permit downloading the frozen Qwen3 checkpoint when extraction is needed")
    evaluate.set_defaults(func=run_evaluate)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
