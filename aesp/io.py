"""Strict JSONL input and indented JSON output; existing files are not overwritten."""

import hashlib
import json
import math
from pathlib import Path


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON value: {value}")
    return number


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant,
                      parse_float=_float)


def read_json(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL records must be objects")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            yield row


def indexed(rows, label="records"):
    result = {}
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError(f"{label}: sample_id must be a nonempty string")
        if sid in result:
            raise ValueError(f"{label}: duplicate sample_id {sid}")
        result[sid] = row
    if not result:
        raise ValueError(f"{label}: empty input")
    return result


def aligned(reference, other, label):
    if set(reference) != set(other):
        missing = sorted(set(reference) - set(other))[:5]
        extra = sorted(set(other) - set(reference))[:5]
        raise ValueError(f"{label}: sample IDs differ; missing={missing}, extra={extra}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
