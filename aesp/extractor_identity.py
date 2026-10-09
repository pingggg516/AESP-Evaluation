"""Content identity of the frozen answer extractor; standard library only.

This module never downloads or loads a model. The bundled manifest pins official
HF LFS SHA-256 identifiers and SHA-256 hashes of pinned ordinary file bytes.
Call ``verify_checkpoint(path)`` to read and verify every required local file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
from typing import Any

MODEL_ID = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
MANIFEST_SHA256 = "acb999b5a8322d12e8cae7617a569d187da756efa04c4e098cbb60cfa6d19324"
MANIFEST_PATH = Path(__file__).resolve().parent / "assets" / "extractor_checkpoint.json"


class CheckpointIdentityError(ValueError):
    """Missing, modified, or ambiguous extractor loading files."""


def load_manifest() -> dict[str, Any]:
    """Read the shipped immutable file manifest, checking its own SHA-256."""
    try:
        raw = MANIFEST_PATH.read_bytes()
    except OSError as exc:
        raise CheckpointIdentityError("extractor_manifest_unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise CheckpointIdentityError("extractor_manifest_sha256_mismatch")
    manifest = json.loads(raw)
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise CheckpointIdentityError("extractor_manifest_identity_mismatch")
    return manifest


def unverified_identity() -> dict[str, Any]:
    """Expected identity is not evidence that any local checkpoint matches it."""
    load_manifest()
    return {"status": "unverified", "model_id": MODEL_ID,
            "revision": MODEL_REVISION, "manifest_sha256": MANIFEST_SHA256}


def validate_verified_identity(report: Any) -> dict[str, Any]:
    """Check a verification report against every frozen manifest field.

    This checks report consistency; it does not hash a checkpoint or authenticate
    arbitrary user-supplied reports. Only ``verify_checkpoint`` reads the actual
    bytes. The loader uses this guard when reusing its already verified model.
    A detached report is returned; malformed or incomplete reports are rejected.
    """
    manifest = load_manifest()
    hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    sizes = {item["path"]: item["size"] for item in manifest["files"]}
    expected = {"status": "verified", "model_id": MODEL_ID,
                "revision": MODEL_REVISION, "manifest_sha256": MANIFEST_SHA256,
                "verification": "all_required_file_bytes_sha256",
                "file_count": len(hashes), "total_bytes": sum(sizes.values()),
                "file_sha256": hashes, "file_size": sizes,
                "source_api_url": manifest["source_api_url"]}
    if type(report) is not dict or set(report) != set(expected) | {"checkpoint_path"}:
        raise CheckpointIdentityError("checkpoint_verification_report_invalid_schema")
    for key, wanted in expected.items():
        actual = report[key]
        if type(actual) is not type(wanted) or actual != wanted:
            raise CheckpointIdentityError("checkpoint_verification_report_mismatch:" + key)
    if any(type(value) is not int for value in report["file_size"].values()):
        raise CheckpointIdentityError("checkpoint_verification_report_mismatch:file_size")
    path = report["checkpoint_path"]
    if (not isinstance(path, str) or not path.strip() or "\x00" in path or
            not (PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute())):
        raise CheckpointIdentityError("checkpoint_verification_report_invalid_path")
    return json.loads(json.dumps(report, allow_nan=False))


def _inventory(directory: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    required = {entry["path"] for entry in manifest["files"]}
    safe_files = set(manifest["allowed_non_loading_files"])
    safe_dirs = set(manifest["allowed_non_loading_directories"])
    try:
        children = {item.name: item for item in directory.iterdir()}
    except OSError as exc:
        raise CheckpointIdentityError("checkpoint_directory_unreadable:" + str(directory)) from exc
    missing = sorted(required - children.keys())
    if missing:
        raise CheckpointIdentityError("checkpoint_missing_files:" + ",".join(missing))
    # Reject every unknown top-level entry, including alternative model files,
    # adapters, chat_templates/, tokenizer sidecars, and executable code. Only
    # documentation and HF/Git bookkeeping directories are irrelevant to this
    # fixed trust_remote_code=False, root-directory loading path.
    extra = []
    for name, item in children.items():
        if name in required:
            continue
        if name in safe_files and item.is_file():
            continue
        if name in safe_dirs and item.is_dir():
            continue
        extra.append(name)
    if extra:
        raise CheckpointIdentityError("checkpoint_unexpected_entries:" + ",".join(sorted(extra)))
    return children


def _file_state(path: Path) -> tuple[str, int, int, int, int]:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise CheckpointIdentityError("checkpoint_file_unreadable:" + path.name) from exc
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointIdentityError("checkpoint_not_regular_file:" + path.name)
    return str(resolved), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def verify_checkpoint(path: str | Path) -> dict[str, Any]:
    """Verify all local loading bytes against the frozen official manifest.

    HF snapshot symlinks are supported: their target bytes are verified. Nothing
    is inferred from a directory name, snapshot name, or a revision argument.
    No alternate manifest can be supplied through this public API. A successful
    report describes the files read now; callers must not mutate them while a
    subsequent model load is in progress.
    """
    manifest = load_manifest()
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise CheckpointIdentityError("checkpoint_directory_missing:" + str(directory))
    directory = directory.resolve()
    children = _inventory(directory, manifest)
    states = {}
    hashes = {}
    sizes = {}
    for entry in manifest["files"]:
        name = entry["path"]
        source = children[name]
        before = _file_state(source)
        if before[3] != entry["size"]:
            raise CheckpointIdentityError("checkpoint_size_mismatch:" + name)
        digest = hashlib.sha256()
        read_bytes = 0
        try:
            with source.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    read_bytes += len(chunk)
        except OSError as exc:
            raise CheckpointIdentityError("checkpoint_file_unreadable:" + name) from exc
        if before != _file_state(source) or read_bytes != entry["size"]:
            raise CheckpointIdentityError("checkpoint_changed_during_verification:" + name)
        if digest.hexdigest() != entry["sha256"]:
            raise CheckpointIdentityError("checkpoint_sha256_mismatch:" + name)
        states[name] = before
        hashes[name] = digest.hexdigest()
        sizes[name] = read_bytes
    # Detect an earlier file or the set of loading files changing while later
    # weight shards were being read. Verification never repairs the directory.
    _inventory(directory, manifest)
    for name, before in states.items():
        if before != _file_state(directory / name):
            raise CheckpointIdentityError("checkpoint_changed_during_verification:" + name)
    return {"status": "verified", "model_id": MODEL_ID,
            "revision": MODEL_REVISION, "manifest_sha256": MANIFEST_SHA256,
            "verification": "all_required_file_bytes_sha256",
            "checkpoint_path": str(directory), "file_count": len(hashes),
            "total_bytes": sum(sizes.values()), "file_sha256": hashes,
            "file_size": sizes, "source_api_url": manifest["source_api_url"]}
