"""Frozen text-only answer extraction, independent of images and ground truth.

Importing this module does not import PyTorch or load/download model weights.
Actual prompt strings and source metadata are in ``assets/``. Negative integer
strings intentionally survive extraction; the scoring canonicalizer must reject
negative object counts. The count prompt and input payload contract are unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterable, Mapping

from .extractor_identity import (CheckpointIdentityError, MANIFEST_SHA256, load_manifest,
                                 unverified_identity, validate_verified_identity,
                                 verify_checkpoint)

TASKS = ("existence", "counting", "comparison")
FIELDS = ("question", "task_type", "target_classes", "answer_text")
MODEL_ID = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
SEED = 20260906
MAX_INPUT_TOKENS = 6000
MAX_NEW_TOKENS = 256
ASSETS = Path(__file__).resolve().parent / "assets"
REFERENCE_EXTRACTOR_RUNTIME = {
    "python_minor": "3.10", "numpy": "2.2.6", "torch": "2.5.1+cu121",
    "transformers": "4.51.3", "accelerate": "1.3.0", "tokenizers": "0.21.1",
    "huggingface-hub": "0.36.2", "safetensors": "0.8.0",
}


def dumps(value: Any) -> str:
    """Serialize with default separators and sorted Unicode keys."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def objsha(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the four answer-extraction input fields."""
    value = {key: record[key] for key in FIELDS}
    if value["task_type"] not in TASKS or not isinstance(value["answer_text"], str):
        raise ValueError("invalid_input")
    if not isinstance(value["question"], str) or not value["question"].strip():
        raise ValueError("empty_question")
    if not isinstance(value["target_classes"], list) or not value["target_classes"]:
        raise ValueError("missing_target_classes")
    return value


def strict_extract(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract bare answers; return None for responses requiring the LLM."""
    text = record["answer_text"].strip()
    if not text:
        return {"status": "no_answer", "answer": None, "spans": []}
    pattern = r"([+-]?[0-9]+)[.!]?" if record["task_type"] == "counting" else r"(yes|no)[.!]?"
    match = re.fullmatch(pattern, text, re.I | re.ASCII)
    if not match:
        return None
    try:
        answer = str(int(match[1])) if record["task_type"] == "counting" else match[1].lower()
    except ValueError:
        return None
    return {"status": "extracted", "answer": answer, "spans": [text]}


def _source_prompts() -> tuple[str, str]:
    """Read the actual prompt text without metadata or JSON escaping."""
    binary = (ASSETS / "binary_prompt.txt").read_text(encoding="utf-8")
    counting = (ASSETS / "counting_examples.txt").read_text(encoding="utf-8")
    if not binary.strip() or not counting.strip():
        raise ValueError("empty_extraction_prompt")
    return binary, counting


BINARY_TEMPLATE, COUNTING_DEMOS = _source_prompts()
RULE = (ASSETS / "extraction_rule.txt").read_text(encoding="utf-8").strip()
COUNTING_HINT = (
    "Hint: Please answer the question requiring an integer answer and provide the final value, "
    "e.g., 1, 2, 3, at the end."
)
COUNTING_EXTRACTION_BOUNDARY = (
    "Extract only the count asserted in the supplied model response; do not solve the question yourself. "
    "If the response does not specify a unique exact count for the queried objects, "
    "output UNRESOLVED. Output only the integer or UNRESOLVED, without an explanation."
)


def build_user_prompt(record: Mapping[str, Any]) -> str:
    """Build the exact frozen prompt; no image, GT, boxes or model name is read."""
    question, text, task = record["question"], record["answer_text"], record["task_type"]
    if not isinstance(question, str) or not isinstance(text, str):
        raise ValueError("invalid_text_input")
    if task in ("existence", "comparison"):
        return BINARY_TEMPLATE.format(
            rule=RULE.format(invalid="Z"),
            question=json.dumps(question, ensure_ascii=False),
            response=json.dumps(text, ensure_ascii=False))
    elif task == "counting":
        query = COUNTING_HINT + "\nQuestion: " + question
        return (COUNTING_DEMOS + "\n\n" + RULE.format(invalid="UNRESOLVED") + "\n\n"
                + COUNTING_EXTRACTION_BOUNDARY + "\n\n" + query + "\n\nModel response: "
                + text + "\n\nExtracted answer: ")
    else:
        raise ValueError("unknown_task")


def parse_short(text: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the LLM's exact option/integer output, without recovery heuristics."""
    value = text.strip()
    if record["task_type"] in ("existence", "comparison"):
        if value not in ("A", "B", "Z"):
            raise ValueError("not_a_single_valid_option")
        from .binary_guard import binary_guard
        checked = binary_guard(record)
        if checked is not None:
            return checked
        if value == "Z":
            return {"status": "unresolved", "answer": None}
        return {"status": "extracted", "answer": {"A": "yes", "B": "no"}[value]}
    if record["task_type"] != "counting":
        raise ValueError("unknown_task")
    if value == "UNRESOLVED":
        return {"status": "unresolved", "answer": None}
    if not re.fullmatch(r"[+-]?[0-9]+", value, flags=re.ASCII):
        raise ValueError("not_an_exact_integer")
    return {"status": "extracted", "answer": str(int(value))}


def prompt_metadata() -> dict[str, Any]:
    """JSON-object hashes are distinct from raw asset byte hashes."""
    expected = json.loads((ASSETS / "SHA256SUMS.json").read_text(encoding="utf-8"))
    actual = {name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ExtractionError("frozen_asset_hash_mismatch")
    return {
        "binary_template_sha256": objsha(BINARY_TEMPLATE),
        "counting_demos_sha256": objsha(COUNTING_DEMOS),
        "extraction_rule_sha256": objsha(RULE),
        "asset_byte_sha256": actual,
        "hash_serialization": "json.dumps(ensure_ascii=False, sort_keys=True, allow_nan=False); UTF-8",
    }


class ExtractionError(RuntimeError):
    """A technical failure, never an answer-level INVALID result.

    ``results`` preserves completed/failed records from the attempted run for
    caller-side diagnostics. Do not score a run after catching this exception.
    """

    def __init__(self, message: str, results: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.results = [] if results is None else results


def extractor_runtime_versions() -> dict[str, Any]:
    """Read package metadata only; no torch import, GPU use, or model access."""
    versions: dict[str, Any] = {"python": platform.python_version()}
    for name in REFERENCE_EXTRACTOR_RUNTIME:
        if name == "python_minor":
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_extractor_runtime(versions=None) -> dict[str, str]:
    """Require the reference software profile, not identical external hardware."""
    versions = extractor_runtime_versions() if versions is None else versions
    expected_keys = (set(REFERENCE_EXTRACTOR_RUNTIME) - {"python_minor"}) | {"python"}
    if type(versions) is not dict or set(versions) != expected_keys:
        raise ExtractionError("extractor_runtime_metadata_invalid")
    python = versions["python"]
    if not isinstance(python, str) or not re.fullmatch(r"3\.10\.\d+", python):
        raise ExtractionError("extractor_runtime_mismatch:python; use Python 3.10 and install requirements.txt from the project root")
    for name, expected in REFERENCE_EXTRACTOR_RUNTIME.items():
        if name != "python_minor" and versions[name] != expected:
            raise ExtractionError("extractor_runtime_mismatch:" + name + "; install requirements.txt from the project root")
    return dict(versions)


def frozen_settings() -> dict[str, Any]:
    return {
        "dtype": "bfloat16", "quantization": None, "do_sample": False,
        "num_beams": 1, "batch_size": 4,
        "max_new_tokens": MAX_NEW_TOKENS, "max_input_tokens": MAX_INPUT_TOKENS,
        "seed": SEED, "enable_thinking": False, "padding_side": "left",
        "add_special_tokens": False, "truncation": False, "use_cache": True,
        "attn_implementation": "sdpa", "trust_remote_code": False,
        "allow_tf32": False,
    }


def decision_policy_sha256() -> str:
    """Bind answer decisions to their code, independent of platform newlines."""
    root = Path(__file__).resolve().parent
    return objsha({
        name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for name in ("extraction.py", "binary_guard.py")
    })


def validate_extraction_identity(identity) -> dict[str, Any]:
    """Check a cache's declared identity; this cannot authenticate execution."""
    load_manifest()
    expected = {
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "checkpoint_manifest_sha256": MANIFEST_SHA256,
        "checkpoint_verification": "all_required_file_bytes_sha256",
        "settings_sha256": objsha(frozen_settings()),
        "prompt_hashes_sha256": objsha(prompt_metadata()),
        "decision_policy_sha256": decision_policy_sha256(),
    }
    if type(identity) is not dict or set(identity) != set(expected) | {"runtime", "eos_token_ids"}:
        raise ExtractionError("extraction_identity_invalid_schema")
    for key, value in expected.items():
        if type(identity[key]) is not type(value) or identity[key] != value:
            raise ExtractionError("extraction_identity_mismatch:" + key)
    validate_extractor_runtime(identity["runtime"])
    eos = identity["eos_token_ids"]
    if (not isinstance(eos, list) or not eos or
            any(type(token) is not int or token < 0 for token in eos) or len(eos) != len(set(eos))):
        raise ExtractionError("extraction_identity_invalid_eos")
    return json.loads(json.dumps(identity, allow_nan=False))


class LLMExtractor:
    """Qwen3-14B answer extractor using the frozen decoding protocol.

    ``model_path`` accepts the official repository ID or an existing local model
    directory. Loading is lazy; by default only existing cached/local files are
    used. A local directory must contain the pinned revision: a path alone cannot
    establish weight identity, which is recorded explicitly in ``metadata()``.
    Device placement and memory budgets are portable and do not assume two GPUs.
    """

    def __init__(self, model_path: str | Path = MODEL_ID, *, device_map: Any = "auto",
                 max_memory: Mapping[Any, Any] | None = None,
                 local_files_only: bool = True, batch_size: int = 4):
        if type(batch_size) is not int or batch_size != 4:
            raise ValueError("frozen_extractor_requires_batch_size_4")
        self.model_path = str(model_path)
        self.device_map = device_map
        self.max_memory = None if max_memory is None else {
            int(key) if isinstance(key, str) and key.isdigit() else key: value
            for key, value in max_memory.items()
        }
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self._model = self._tokenizer = self._torch = None
        self._versions: dict[str, str] = {}
        self._prompt_metadata = prompt_metadata()
        self._checkpoint_identity = unverified_identity()
        self._generation_config_sha256 = None
        self._eos_token_ids = None

    def metadata(self) -> dict[str, Any]:
        if type(self.batch_size) is not int or self.batch_size != 4:
            raise ExtractionError("frozen_extractor_requires_batch_size_4")
        identity = (validate_verified_identity(self._checkpoint_identity)
                    if self._model is not None else unverified_identity())
        versions = dict(self._versions) if self._versions else extractor_runtime_versions()
        if self._model is not None:
            validate_extractor_runtime(versions)
        return {
            "model": MODEL_ID, "model_path": self.model_path, "revision": MODEL_REVISION,
            "weight_identity": "verified-sha256" if self._model is not None else "unverified",
            "checkpoint_identity": identity,
            "loaded": self._model is not None,
            "settings": {
                **frozen_settings(),
                "local_files_only": self.local_files_only, "device_map": self.device_map,
                "max_memory": None if self.max_memory is None else {str(k): v for k, v in self.max_memory.items()},
            },
            "prompt_hashes": dict(self._prompt_metadata), "versions": versions,
            "decision_policy_sha256": decision_policy_sha256(),
            "gt_or_image_accessed": False, "human_labels_accessed": False,
            "commercial_api_calls": 0,
        }

    def load(self) -> None:
        versions = validate_extractor_runtime()
        if type(self.batch_size) is not int or self.batch_size != 4:
            raise ExtractionError("frozen_extractor_requires_batch_size_4")
        if self._model is not None:
            try:
                validate_verified_identity(self._checkpoint_identity)
            except CheckpointIdentityError as exc:
                raise ExtractionError("checkpoint_identity_failed: " + str(exc)) from exc
            return
        prompt_metadata()
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Passing local_files_only below enforces offline weight access without
        # globally changing HF_HUB_OFFLINE for unrelated application components.
        try:
            if Path(self.model_path).is_dir():
                checkpoint = str(Path(self.model_path).resolve())
            elif self.model_path == MODEL_ID:
                from huggingface_hub import snapshot_download
                checkpoint = snapshot_download(
                    repo_id=MODEL_ID, revision=MODEL_REVISION,
                    allow_patterns=[item["path"] for item in load_manifest()["files"]],
                    local_files_only=self.local_files_only, token=False)
            else:
                raise CheckpointIdentityError("checkpoint_directory_missing:" + self.model_path)
            identity = verify_checkpoint(checkpoint)
        except Exception as exc:
            raise ExtractionError("checkpoint_identity_failed: " + str(exc)) from exc
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ExtractionError("Install requirements.txt in Python 3.10 before answer extraction") from exc
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Resolve/download first, verify every byte, then load only that local
        # directory. A second remote lookup cannot substitute unverified files.
        common = {"revision": MODEL_REVISION, "local_files_only": True,
                  "trust_remote_code": False}
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left", **common)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": self.device_map,
                      "attn_implementation": "sdpa", **common}
            if self.max_memory is not None:
                kwargs["max_memory"] = self.max_memory
            model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
            model.eval()
        except Exception as exc:
            raise ExtractionError("model_loading_failed: " + str(exc)) from exc
        self._torch, self._tokenizer, self._model = torch, tokenizer, model
        self._checkpoint_identity = identity
        self._versions = versions
        self._generation_config_sha256 = objsha(model.generation_config.to_dict())
        eos = model.generation_config.eos_token_id
        self._eos_token_ids = list(eos) if isinstance(eos, list) else [eos]

    def _identity(self) -> dict[str, Any]:
        validate_verified_identity(self._checkpoint_identity)
        return validate_extraction_identity({
            "model": MODEL_ID, "revision": MODEL_REVISION,
            "checkpoint_manifest_sha256": MANIFEST_SHA256,
            "checkpoint_verification": "all_required_file_bytes_sha256",
            "settings_sha256": objsha(frozen_settings()),
            "prompt_hashes_sha256": objsha(prompt_metadata()),
            "decision_policy_sha256": decision_policy_sha256(),
            "runtime": dict(self._versions), "eos_token_ids": list(self._eos_token_ids),
        })

    def extract(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Extract non-strict records. Any processing failure aborts the run."""
        rows = list(records)
        if not rows:
            return []
        prepared = []
        for record in rows:
            if strict_extract(record) is not None:
                raise ExtractionError("strict_record_sent_to_llm:" + str(record.get("record_id", record.get("sample_id"))))
            input_hash = objsha(payload(record))
            user_prompt = build_user_prompt(record)
            prompt_hash = objsha(user_prompt)
            for field, expected in (("input_sha256", input_hash), ("prompt_sha256", prompt_hash)):
                if field in record and record[field] != expected:
                    raise ExtractionError(field + "_mismatch")
            prepared.append((record, user_prompt, input_hash, prompt_hash))
        self.load()
        tok, model, torch = self._tokenizer, self._model, self._torch
        if (objsha(model.generation_config.to_dict()) != self._generation_config_sha256 or
                tok.padding_side != "left" or model.training or
                torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ExtractionError("frozen_extractor_runtime_settings_changed")
        extraction_identity = self._identity()
        device = model.get_input_embeddings().weight.device
        results: list[dict[str, Any]] = []
        for offset in range(0, len(prepared), self.batch_size):
            batch = prepared[offset:offset + self.batch_size]
            texts, metadata = [], []
            try:
                for record, user_prompt, _, _ in batch:
                    messages = [{"role": "user", "content": user_prompt}]
                    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    canonical = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                    actual = tok(text, add_special_tokens=False)["input_ids"]
                    if actual != canonical:
                        raise ExtractionError("chat_template_tokenization_mismatch", results)
                    if len(actual) > MAX_INPUT_TOKENS:
                        raise ExtractionError("input_exceeds_6000_tokens:" + str(record.get("record_id", record.get("sample_id"))), results)
                    texts.append(text)
                    metadata.append({"messages_sha256": objsha(messages), "token_ids_sha256": objsha(actual), "input_tokens": len(actual)})
                inputs = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False).to(device)
                for i, meta in enumerate(metadata):
                    ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
                    if objsha(ids) != meta["token_ids_sha256"]:
                        raise ExtractionError("batched_tokenization_mismatch", results)
                with torch.inference_mode():
                    out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                                         pad_token_id=tok.pad_token_id, use_cache=True)
                if (out.shape[0] != len(batch) or out.shape[1] < inputs["input_ids"].shape[1] or
                        not torch.equal(out[:, :inputs["input_ids"].shape[1]], inputs["input_ids"])):
                    raise ExtractionError("generation_input_prefix_mismatch", results)
                tokens = out[:, inputs["input_ids"].shape[1]:]
                decoded = tok.batch_decode(tokens, skip_special_tokens=True)
                eos = set(self._eos_token_ids)
                if len(decoded) != len(batch) or len(tokens) != len(batch):
                    raise ExtractionError("generation_batch_size_mismatch", results)
                batch_failed = False
                for (record, _, input_hash, prompt_hash), text, ids, meta in zip(batch, decoded, tokens.tolist(), metadata):
                    ended = any(token in eos for token in ids)
                    first_eos = next((i for i, token in enumerate(ids) if token in eos), None)
                    output_ids = ids if first_eos is None else ids[:first_eos + 1]
                    if not output_ids or len(output_ids) > MAX_NEW_TOKENS:
                        raise ExtractionError("generation_output_length_invalid", results)
                    if first_eos is not None and any(token != tok.pad_token_id for token in ids[first_eos + 1:]):
                        raise ExtractionError("generation_nonpadding_after_eos", results)
                    result = {
                        "sample_id": record["sample_id"],
                        "model": record.get("origin_model", "custom"),
                        "record_id": record.get("record_id", record.get("sample_id")),
                        "input_sha256": input_hash, "prompt_sha256": prompt_hash,
                        "raw_extraction": text, "eos_observed": ended, "decision": None,
                        "output_token_ids": output_ids, "output_tokens": len(output_ids),
                        "eos_token_ids": list(self._eos_token_ids), "input_prefix_mismatch_count": 0,
                        "extractor_identity": extraction_identity,
                        "processing_status": "completed" if ended else "generation_truncated", **meta,
                    }
                    if ended:
                        try:
                            result["decision"] = parse_short(text, record)
                        except (ValueError, TypeError) as exc:
                            result.update(processing_status="invalid_output", validation_error=str(exc))
                    if result["processing_status"] != "completed":
                        batch_failed = True
                    results.append(result)
                if batch_failed:
                    raise ExtractionError("llm_processing_failed; inspect exception.results", results)
            except ExtractionError:
                raise
            except Exception as exc:
                raise ExtractionError("inference_failed: " + str(exc), results) from exc
        return results
