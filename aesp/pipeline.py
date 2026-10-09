"""Model-independent orchestration. No images or ground truth enter the extractor."""

import copy
import re

from .extraction import (MAX_INPUT_TOKENS, MAX_NEW_TOKENS, build_user_prompt,
                         objsha, parse_short, payload, strict_extract,
                         validate_extraction_identity)
from .parsing import answer_view
from .regions import validate_prediction_regions
from .generation_status import validate_generation_status

INPUT_FIELDS = ("question", "task_type", "target_classes", "source_dataset", "image_width", "image_height")
PROVENANCE_FIELDS = ("inference_protocol", "image_sha256", "generation_calls",
                     "request_audit", "image_metadata", "generation_diagnostics",
                     "input_requirements")
GENERATION_FIELDS = ("generation", "answer_generation", "grounding_generation",
                     "callmetadata", "generation_status", "technical_metadata",
                     "joint_generation", "elapsed_seconds", "prompt", "prompts",
                     "prompt_sha256", "shared_prompt", "shared_prompt_sha256", "answer_reused", "grounding_reused",
                     "call_metadata", "metadata", "error", "technical_error", "status", "finish_reason",
                     "stop_reason", "processing_status", "inference_status", "possible_truncation",
                     "truncated", "ended_with_eos", "eos_observed", "output_budget_hit",
                     "input_prefix_mismatch_count", "output_token_count",
                     "decoded_stop_suffix_removed", "scoreable", "tokens", "budget",
                     "calls", "native_diagnostics",
                     "generation_metadata", "candidate_generation_status",
                     "generation_validation", "submission_status")


def _inference_protocol(prediction):
    """Read declared provenance; never infer a protocol from the model name."""
    nested = prediction.get("inference_provenance")
    if nested is not None and not isinstance(nested, dict):
        raise ValueError("inference_provenance must be an object")
    candidates = [prediction.get("inference_protocol")]
    if nested is not None:
        candidates.append(nested.get("inference_protocol"))
    protocols, runtime_hashes, dataset_hashes = set(), set(), set()
    for value in candidates:
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError("inference_protocol must be an object")
        name, digest = value.get("id"), value.get("sha256")
        if (not isinstance(name, str) or not name.strip() or
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
            raise ValueError("inference_protocol requires a nonempty id and SHA256")
        protocols.add((name, digest.lower()))
        runtime = value.get("runtime_configuration_sha256")
        if runtime is not None:
            if not isinstance(runtime, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", runtime):
                raise ValueError("runtime_configuration_sha256 must be a SHA256")
            runtime_hashes.add(runtime.lower())
        dataset = value.get("dataset_sha256")
        if dataset is not None:
            if not isinstance(dataset, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", dataset):
                raise ValueError("dataset_sha256 must be a SHA256")
            dataset_hashes.add(dataset.lower())
    if len(protocols) > 1:
        raise ValueError("Prediction contains conflicting inference protocols")
    if len(runtime_hashes) > 1:
        raise ValueError("Prediction contains conflicting runtime configurations")
    if len(dataset_hashes) > 1:
        raise ValueError("Prediction contains conflicting dataset fingerprints")
    if not protocols:
        return None
    name, digest = next(iter(protocols))
    return {"id": name, "sha256": digest,
            **({"runtime_configuration_sha256": next(iter(runtime_hashes))} if runtime_hashes else {}),
            **({"dataset_sha256": next(iter(dataset_hashes))} if dataset_hashes else {})}


def summarize_inference_provenance(predictions):
    """Summarize known/unknown records and reject mixed declared protocols."""
    known = unknown = runtime_known = runtime_unknown = 0
    protocols, runtime_hashes, dataset_hashes = set(), set(), set()
    for prediction in predictions:
        protocol = _inference_protocol(prediction)
        if protocol is None:
            unknown += 1
        else:
            known += 1
            protocols.add((protocol["id"], protocol["sha256"]))
        if protocol is not None and "runtime_configuration_sha256" in protocol:
            runtime_known += 1
            runtime_hashes.add(protocol["runtime_configuration_sha256"])
        else:
            runtime_unknown += 1
        if protocol is not None and "dataset_sha256" in protocol:
            dataset_hashes.add(protocol["dataset_sha256"])
    if len(protocols) > 1:
        raise ValueError("Predictions contain conflicting inference protocol IDs or hashes")
    if len(runtime_hashes) > 1:
        raise ValueError("Predictions contain conflicting runtime configuration hashes")
    if len(dataset_hashes) > 1:
        raise ValueError("Predictions contain conflicting dataset fingerprints")
    protocol = None
    if protocols:
        name, digest = next(iter(protocols))
        protocol = {"id": name, "sha256": digest}
    return {"status": "partial" if known and unknown else "known" if known else "unknown",
            "protocol": protocol, "known_records": known, "unknown_records": unknown,
            "dataset_sha256": next(iter(dataset_hashes)) if dataset_hashes else None,
            "runtime_configuration": {
                "status": "partial" if runtime_known and runtime_unknown else "known" if runtime_known else "unknown",
                "sha256": next(iter(runtime_hashes)) if runtime_hashes else None,
                "known_records": runtime_known, "unknown_records": runtime_unknown,
                "weight_identity_verified": False}}


def preserve_inference_provenance(prediction):
    """Keep source/generation diagnostics separate from normalized score inputs."""
    _inference_protocol(prediction)
    value = copy.deepcopy(prediction.get("inference_provenance") or {})
    for key in PROVENANCE_FIELDS:
        if key in prediction:
            if key in value and value[key] != prediction[key]:
                raise ValueError("Conflicting inference provenance: " + key)
            value[key] = copy.deepcopy(prediction[key])
    generation = {key: copy.deepcopy(prediction[key]) for key in GENERATION_FIELDS if key in prediction}
    if generation:
        previous = value.get("generation_metadata", {})
        if not isinstance(previous, dict):
            raise ValueError("generation_metadata must be an object")
        for key, item in generation.items():
            if key in previous and previous[key] != item:
                raise ValueError("Conflicting generation metadata: " + key)
        value["generation_metadata"] = {**previous, **generation}
    return value


def make_record(sample, prediction, model=None):
    if prediction.get("sample_id") != sample["sample_id"]:
        raise ValueError("Prediction/sample ID mismatch")
    for key in INPUT_FIELDS:
        if key in prediction and prediction[key] != sample[key]:
            raise ValueError(f"{sample['sample_id']}: prediction metadata differs: {key}")
    name = prediction.get("model", model or "custom")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model must be a nonempty string")
    if model is not None and name != model:
        raise ValueError("--model differs from prediction model")
    p = {**prediction, **{k: sample[k] for k in INPUT_FIELDS}, "model": name}
    text, rule = answer_view(p)
    record = {k: sample[k] for k in ("sample_id", "question", "task_type", "target_classes")}
    record.update(record_id=name + ":" + sample["sample_id"], origin_model=name, answer_text=text)
    record["input_sha256"] = objsha(payload(record))
    return p, record, rule


def cached_decision(record, cached):
    """Check cache consistency, never certify who executed an external cache.

    Non-strict answers require the full frozen extractor receipt. A label plus
    self-declared prompt hash is insufficient; missing receipts must be rebuilt
    with the extractor rather than silently upgraded to a verified result.
    """
    if not isinstance(cached, dict):
        raise ValueError("Answer cache record must be an object")
    validate_generation_status(cached)
    if cached.get("sample_id") != record["sample_id"] or cached.get("record_id") != record["record_id"]:
        raise ValueError("Answer cache record/sample mismatch")
    if cached.get("model", record["origin_model"]) != record["origin_model"]:
        raise ValueError("Answer cache model mismatch")
    if cached.get("processing_status") != "completed":
        raise ValueError("Answer cache contains unfinished/failed extraction")
    if "eos_observed" in cached and cached["eos_observed"] is not True:
        raise ValueError("Answer cache contains truncated extraction")
    if cached.get("input_sha256") != record["input_sha256"]:
        raise ValueError(f"{record['record_id']}: stale answer cache (text/input hash)")
    value = cached.get("decision")
    if not isinstance(value, dict) or value.get("status") not in ("extracted", "no_answer", "unresolved"):
        raise ValueError("Answer cache has invalid decision schema")
    if (value["status"] == "extracted") != (value.get("answer") is not None):
        raise ValueError("Answer cache status/answer mismatch")
    if "pred_answer" in cached and cached["pred_answer"] != value.get("answer"):
        raise ValueError("Answer cache pred_answer/decision mismatch")
    if value["status"] == "extracted":
        a = value["answer"]
        if record["task_type"] == "counting":
            if not isinstance(a, str) or not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", a) or str(int(a)) != a:
                raise ValueError("Answer cache has noncanonical count")
        elif a not in ("yes", "no"):
            raise ValueError("Answer cache has noncanonical binary answer")
    direct = strict_extract(record)
    if direct is not None:
        if value != direct:
            raise ValueError("Answer cache differs from deterministic extraction")
    else:
        required = {"model", "extractor_identity", "raw_extraction", "eos_observed", "messages_sha256",
                    "token_ids_sha256", "input_tokens", "output_token_ids", "output_tokens",
                    "eos_token_ids", "input_prefix_mismatch_count"}
        missing = required - set(cached)
        if missing:
            raise ValueError("Non-strict answer cache lacks extraction evidence: " + ", ".join(sorted(missing)))
        identity = validate_extraction_identity(cached["extractor_identity"])
        prompt = build_user_prompt(record)
        if cached.get("prompt_sha256") != objsha(prompt):
            raise ValueError(f"{record['record_id']}: stale answer cache (prompt hash)")
        if cached["messages_sha256"] != objsha([{"role": "user", "content": prompt}]):
            raise ValueError("Answer cache messages differ from the frozen prompt")
        digest = cached["token_ids_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Answer cache input token hash must be a SHA256")
        count = cached["input_tokens"]
        if type(count) is not int or not 0 < count <= MAX_INPUT_TOKENS:
            raise ValueError("Answer cache has invalid input token count")
        if cached["eos_observed"] is not True or cached["input_prefix_mismatch_count"] != 0:
            raise ValueError("Answer cache generation was truncated or changed its input prefix")
        tokens, eos_ids = cached["output_token_ids"], cached["eos_token_ids"]
        if (not isinstance(eos_ids, list) or not eos_ids or
                any(type(token) is not int or token < 0 for token in eos_ids) or
                len(set(eos_ids)) != len(eos_ids)):
            raise ValueError("Answer cache EOS token IDs must be unique nonnegative integers")
        if identity.get("eos_token_ids") != eos_ids:
            raise ValueError("Answer cache EOS token IDs disagree with its extractor receipt")
        if (not isinstance(tokens, list) or not 0 < len(tokens) <= MAX_NEW_TOKENS or
                any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Answer cache has invalid generated token IDs")
        if type(cached["output_tokens"]) is not int or cached["output_tokens"] != len(tokens):
            raise ValueError("Answer cache output token count differs from generated IDs")
        if tokens[-1] not in eos_ids or any(token in eos_ids for token in tokens[:-1]):
            raise ValueError("Answer cache token sequence must end at its first EOS")
        if not isinstance(cached["raw_extraction"], str) or parse_short(cached["raw_extraction"], record) != value:
            raise ValueError("Answer cache decision differs from raw extraction")
    return copy.deepcopy(value)


def reused_answer_row(record, cached, value):
    """Preserve every receipt field on cache reuse and identify the reuse as such."""
    row = copy.deepcopy(cached)
    history = row.get("cache_reuse_history", [])
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("Answer cache reuse history must be a list of objects")
    history.append({"source_row_sha256": objsha(cached), "source_route": cached.get("route"),
                    "input_binding_checked": True,
                    "prompt_binding_checked": strict_extract(record) is None,
                    "new_extraction": False, "execution_authenticated": False})
    row.setdefault("original_route", cached.get("route", "external_cache"))
    row.update(record_id=record["record_id"], sample_id=record["sample_id"],
               model=record["origin_model"], pred_answer=value["answer"],
               decision=copy.deepcopy(value), route="reused_cache", cache_reuse_history=history)
    return row


def answer_row(record, value, route):
    direct = strict_extract(record)
    return {"record_id": record["record_id"], "sample_id": record["sample_id"],
            "model": record["origin_model"], "input_sha256": record["input_sha256"],
            "prompt_sha256": None if direct is not None else objsha(build_user_prompt(record)),
            "decision": value, "pred_answer": value["answer"], "route": route,
            "processing_status": "completed"}


def normalize_one(sample, prediction, cached=None, model=None):
    submission_status = validate_generation_status(prediction)
    missing = [key for key in ("sample_id", "raw_answer", "evidence_boxes") if key not in prediction]
    if missing:
        raise ValueError("Prediction missing required fields: " + ", ".join(missing))
    if not isinstance(prediction["raw_answer"], str):
        raise ValueError("raw_answer must be a string")
    boxes = validate_prediction_regions(prediction, sample["image_width"], sample["image_height"])
    p, record, rule = make_record(sample, prediction, model)
    direct = strict_extract(record)
    if cached is not None:
        value, route = cached_decision(record, cached), "reused_cache"
    else:
        value, route = direct, "strict"
    normalized = {"sample_id": sample["sample_id"], "model": p["model"],
                  "raw_answer": prediction["raw_answer"], "evidence_boxes": boxes,
                  "answer_view_rule": rule,
                  "inference_provenance": preserve_inference_provenance(prediction),
                  **{k: sample[k] for k in INPUT_FIELDS}}
    normalized['inference_provenance']['submission_status'] = submission_status
    if value is not None:
        normalized["pred_answer"] = value["answer"]
    decision = (None if value is None else reused_answer_row(record, cached, value)
                if cached is not None else answer_row(record, value, route))
    return normalized, record, decision
