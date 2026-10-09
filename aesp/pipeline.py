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


def make_record(sample, prediction, model=None):
    if prediction.get("sample_id") != sample["sample_id"]:
        raise ValueError("Prediction/sample ID mismatch")
    name = "custom" if model is None else model
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model must be a nonempty string")
    p = {"sample_id": prediction["sample_id"], "raw_answer": prediction["raw_answer"],
         "evidence_boxes": prediction["evidence_boxes"],
         **{k: sample[k] for k in INPUT_FIELDS}, "model": name}
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
    fields = ("sample_id", "raw_answer", "evidence_boxes")
    missing = [key for key in fields if key not in prediction]
    if missing:
        raise ValueError("Prediction missing required fields: " + ", ".join(missing))
    if not isinstance(prediction["raw_answer"], str):
        raise ValueError("raw_answer must be a string")
    prediction = {key: prediction[key] for key in fields}
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
                  **{k: sample[k] for k in INPUT_FIELDS}}
    if value is not None:
        normalized["pred_answer"] = value["answer"]
    decision = (None if value is None else reused_answer_row(record, cached, value)
                if cached is not None else answer_row(record, value, route))
    return normalized, record, decision
