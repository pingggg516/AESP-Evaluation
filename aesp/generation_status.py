"""Check generation status in AESP answer-extraction results and caches.

External prediction failure/truncation flags do not block submission. This check
protects the evaluator's own extraction results before they are used for scoring.
"""

from collections.abc import Mapping
import re


GUARD_ID = "generation-status"

_FAILED = {
    "failed", "failure", "error", "technical_error", "inference_failed",
    "generation_truncated", "invalid_output",
    "generation_failed", "inference_failure", "incomplete", "truncated", "length", "length_limit",
    "max_tokens", "max_new_tokens", "max_output_tokens", "max_completion_tokens", "token_limit",
    "generation_budget_exhausted", "stopped_without_eos", "out_of_memory", "oom", "cuda_oom",
    "cuda_out_of_memory", "cuda_error", "cuda_runtime_error", "exception", "oom_killed",
    "timeout", "timed_out", "cancelled", "canceled", "aborted", "interrupted",
    "pending", "queued", "running", "in_progress", "content_filter",
    "content_filtered", "safety", "blocked", "recitation", "malformed_function_call",
    "失败", "推理失败", "生成失败", "运行失败", "错误", "技术错误", "显存不足", "内存不足",
    "未完成", "不完整", "已截断", "截断", "超时", "已取消", "取消", "已中断", "中断",
    "输出达到上限", "达到长度上限", "长度上限", "内容过滤", "内容被过滤",
}
_COMPLETED = {"completed", "complete", "success", "succeeded", "ok", "eos",
              "stop", "stop_sequence", "end_turn", "finished"}
_STATUS_FIELDS = ("generation_status", "inference_status", "processing_status", "status",
                  "finish_reason", "stop_reason")
_BOOL_FIELDS = ("possible_truncation", "truncated", "ended_with_eos", "eos_observed",
                "output_budget_hit", "decoded_stop_suffix_removed", "scoreable")
_COUNT_FIELDS = ("output_token_count", "tokens", "budget", "input_prefix_mismatch_count")
_OBJECT_CONTAINERS = {
    "generation", "generation_diagnostics", "technical_metadata", "metadata",
    "native_diagnostics", "inference_provenance", "generation_metadata",
    "candidate_generation_status", "generation_validation", "submission_status", "request_audit",
    "call_metadata",
}
_CALL_CONTAINERS = {"answer_generation", "grounding_generation", "callmetadata", "calls",
                    "generation_calls"}
_CALL_SCOPES = _CALL_CONTAINERS | {
    "generation", "generation_diagnostics", "joint_generation", "call_metadata", "native_diagnostics",
}


def validate_generation_status(prediction):
    """Return a small audit, or raise ValueError identifying sample and field.

Called for answer-extraction results and cached extraction receipts.
Unknown vendor status strings remain unknown; recognized failure/truncation
statuses always reject, even when another field claims success. Nonempty error
values reject. Null/empty error values and missing metadata are allowed.
``generation_calls`` accepts call metadata or a nonnegative integer count; a
count alone is not completion evidence. Conflicting EOS flags in the same
metadata object are rejected, while a lone false flag may accompany normal stop.
This does not certify authenticity of externally supplied completion metadata.
"""
    if not isinstance(prediction, Mapping):
        raise ValueError("Prediction must be an object")
    sample = str(prediction.get("sample_id", "<unknown sample>"))
    inspected = 0
    completed = False
    unknown = False
    active = set()

    def fail(path, reason):
        raise ValueError(f"{sample}: generation cannot be scored: {path}: {reason}")

    def visit(value, path, depth=0, call_scope=False):
        nonlocal inspected, completed, unknown
        if depth > 40:
            fail(path, "generation metadata exceeds supported nesting depth")
        if id(value) in active:
            fail(path, "cyclic generation metadata")
        active.add(id(value))
        local_completed = False
        try:
            for field in ("error", "technical_error"):
                if field not in value:
                    continue
                inspected += 1
                item = value[field]
                if item is None or item is False or item == "":
                    continue
                fail(path + "." + field, "reported generation error")
            for field in _BOOL_FIELDS:
                if field not in value:
                    continue
                inspected += 1
                item = value[field]
                if item is not None and type(item) is not bool:
                    fail(path + "." + field, "expected boolean or null")
                if field in ("possible_truncation", "truncated") and item is True:
                    fail(path + "." + field, "reported output truncation")
                if field == "scoreable" and item is False:
                    fail(path + "." + field, "record explicitly marked unscoreable")
                if field in ("ended_with_eos", "eos_observed") and item is True:
                    completed = True
                    local_completed = True
            if (type(value.get("ended_with_eos")) is bool and
                    type(value.get("eos_observed")) is bool and
                    value["ended_with_eos"] != value["eos_observed"]):
                fail(path, "ended_with_eos contradicts eos_observed")
            for field in _COUNT_FIELDS:
                if field in value:
                    inspected += 1
                    item = value[field]
                    if item is not None and (type(item) is not int or item < 0):
                        fail(path + "." + field, "expected nonnegative integer or null")
                    if field == "input_prefix_mismatch_count" and item is not None and item > 0:
                        fail(path + "." + field, "generated sequence does not preserve the input prefix")
            for field in _STATUS_FIELDS:
                if field not in value:
                    continue
                inspected += 1
                item = value[field]
                if item is None:
                    unknown = True
                    continue
                if not isinstance(item, str) or not item.strip():
                    fail(path + "." + field, "expected nonempty status string or null")
                status = re.sub(r"[\s-]+", "_", item.strip().lower())
                if status in _FAILED:
                    fail(path + "." + field, "reported " + status)
                if status in _COMPLETED:
                    completed = True
                    local_completed = True
                else:
                    unknown = True
                if status == "eos" and (value.get("ended_with_eos") is False or
                                         value.get("eos_observed") is False):
                    fail(path + "." + field, "EOS status contradicts EOS metadata")
            stopped_normally = (
                value.get("ended_with_eos") is True or value.get("eos_observed") is True or
                value.get("decoded_stop_suffix_removed") is True or any(
                    isinstance(value.get(field), str) and value[field].strip().lower() in
                    {"eos", "stop", "stop_sequence", "end_turn"}
                    for field in ("finish_reason", "stop_reason")))
            if stopped_normally:
                completed = True
                local_completed = True
            if value.get("output_budget_hit") is True and not stopped_normally:
                fail(path + ".output_budget_hit", "output budget reached without confirmed normal stop")
            # Some inputs expose token count + budget + EOS instead of
            # finish_reason. Do not infer truncation from a configured limit alone.
            if (value.get("eos_observed") is False and
                    type(value.get("tokens")) is int and
                    type(value.get("budget")) is int and value["budget"] > 0 and
                    value["tokens"] >= value["budget"] and
                    not stopped_normally):
                fail(path, "generation token budget exhausted without EOS")
            for field, item in value.items():
                if field == "joint_generation" and type(item) is bool:
                    # Existing Kosmos records use this as a same-call flag.
                    continue
                if field not in _OBJECT_CONTAINERS | _CALL_CONTAINERS | {"joint_generation"}:
                    continue
                child = path + "." + field
                if field == "generation_calls" and type(item) is int:
                    # generation_calls may store only the number of calls.
                    # Inspect nested objects/lists, but do not treat a count as
                    # either a completed call or a reported technical failure.
                    inspected += 1
                    if item < 0:
                        fail(child, "call count must be a nonnegative integer")
                    continue
                if item is None:
                    if field in _CALL_SCOPES:
                        unknown = True
                    continue
                if isinstance(item, Mapping):
                    child_completed = visit(item, child, depth + 1, field in _CALL_SCOPES)
                    local_completed = child_completed or local_completed
                elif isinstance(item, list) and field in _CALL_CONTAINERS | {"joint_generation"}:
                    if not item:
                        unknown = True
                    for index, entry in enumerate(item):
                        if not isinstance(entry, Mapping):
                            fail(f"{child}[{index}]", "expected metadata object")
                        child_completed = visit(entry, f"{child}[{index}]", depth + 1, True)
                        local_completed = child_completed or local_completed
                else:
                    expected = "expected metadata object" + (
                        " or list of objects" if field in _CALL_CONTAINERS else "")
                    if field == "generation_calls":
                        expected += ", nonnegative integer count, or null"
                    fail(child, expected)
            if call_scope and not local_completed:
                unknown = True
            return local_completed
        finally:
            active.remove(id(value))

    visit(prediction, "prediction")
    return {"guard_id": GUARD_ID,
            "status": "completed" if completed and not unknown else "unknown",
            "scoreable": True, "metadata_fields_checked": inspected}
