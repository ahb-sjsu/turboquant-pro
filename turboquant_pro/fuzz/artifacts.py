"""Deterministic, fail-closed replay-bundle storage for retrieval fuzzing.

The fuzzer's execution engine is intentionally separate from this module.  This
layer makes the durable boundary explicit before candidates can be generated:
every payload is content-addressed, JSON is canonical, and a bundle only becomes
visible once its manifest has been fully written and verified.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_SCHEMA = "turboquant-pro/fuzz-replay-bundle"
BUNDLE_SCHEMA_VERSION = 1
_MANIFEST_NAME = "bundle.json"
_CHECKSUMS_NAME = "checksums.txt"
_COMMIT_NAME = "COMMIT"
_MAX_ARRAY_BYTES = 1 << 30


class ReplayBundleError(ValueError):
    """A replay bundle is missing, unsafe, corrupt, or incompatible."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value deterministically, rejecting non-finite values."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReplayBundleError("bundle metadata must be finite JSON") from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_name(name: str, suffix: str) -> str:
    if not isinstance(name, str):
        raise ReplayBundleError(f"unsafe bundle payload name {name!r}")
    path = Path(name)
    if (
        not name
        or path.name != name
        or name in {".", ".."}
        or not name.endswith(suffix)
    ):
        raise ReplayBundleError(f"unsafe bundle payload name {name!r}")
    return name


def _npz_bytes(array: np.ndarray) -> bytes:
    """Write a one-array NPZ with a fixed ZIP timestamp for byte stability."""
    values = np.asarray(array)
    if values.dtype.hasobject or not np.issubdtype(values.dtype, np.number):
        raise ReplayBundleError("replay arrays must use a numeric non-object dtype")
    if not np.isfinite(values).all():
        raise ReplayBundleError("replay arrays must be finite")
    npy = io.BytesIO()
    np.lib.format.write_array(npy, values, allow_pickle=False)
    info = zipfile.ZipInfo("data.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr(info, npy.getvalue())
    return archive.getvalue()


def _array_from_npz(payload: bytes, name: str) -> np.ndarray:
    if len(payload) > _MAX_ARRAY_BYTES:
        raise ReplayBundleError(f"array payload {name!r} exceeds size limit")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != ["data"]:
                raise ReplayBundleError(
                    f"array payload {name!r} must contain only data.npy"
                )
            values = np.asarray(archive["data"])
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ReplayBundleError(f"invalid NPZ payload {name!r}") from error
    if (
        values.dtype.hasobject
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
    ):
        raise ReplayBundleError(f"invalid array contents in {name!r}")
    return values


def _read_canonical_json(path: Path, name: str) -> Any:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayBundleError(f"invalid JSON payload {name!r}") from error
    if canonical_json_bytes(value) != payload:
        raise ReplayBundleError(f"non-canonical JSON payload {name!r}")
    return value


def write_replay_bundle(
    destination: str | Path,
    *,
    case: Mapping[str, Any],
    geometry: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    documents: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Atomically create a deterministic replay bundle at a new directory.

    ``case.json`` and ``geometry.json`` are mandatory.  Additional documents
    use explicit ``.json`` names, while arrays use stem names and are persisted
    as one-array ``<stem>.npz`` payloads.  A destination must not already exist,
    which prevents a fuzzer campaign from accidentally replacing user evidence.
    """
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise ReplayBundleError(f"replay bundle destination already exists: {target}")
    if not arrays:
        raise ReplayBundleError("replay bundle needs at least one array payload")
    extra_documents = documents or {}
    payloads: dict[str, bytes] = {
        "case.json": canonical_json_bytes(dict(case)),
        "geometry.json": canonical_json_bytes(dict(geometry)),
    }
    for name, document in extra_documents.items():
        safe_name = _safe_name(name, ".json")
        if safe_name in payloads:
            raise ReplayBundleError(f"duplicate replay document name {safe_name!r}")
        payloads[safe_name] = canonical_json_bytes(dict(document))
    for stem, array in arrays.items():
        if not isinstance(stem, str) or not stem or Path(stem).name != stem:
            raise ReplayBundleError(f"unsafe replay array name {stem!r}")
        payloads[f"{stem}.npz"] = _npz_bytes(array)

    checksums = "".join(
        f"{_sha256(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("ascii")
    payloads[_CHECKSUMS_NAME] = checksums
    files = [
        {"path": name, "bytes": len(payloads[name]), "sha256": _sha256(payloads[name])}
        for name in sorted(payloads)
    ]
    from turboquant_pro import __version__

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "tool_version": __version__,
        "files": files,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise ReplayBundleError(f"replay bundle parent does not exist: {parent}")
    temporary = parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        for name, payload in payloads.items():
            (temporary / name).write_bytes(payload)
        (temporary / _MANIFEST_NAME).write_bytes(manifest_bytes)
        (temporary / _COMMIT_NAME).write_text(
            _sha256(manifest_bytes) + "\n", encoding="ascii"
        )
        os.replace(temporary, target)
    except OSError as error:
        raise ReplayBundleError(f"could not write replay bundle: {error}") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def load_replay_bundle(source: str | Path) -> dict[str, Any]:
    """Validate and load a replay bundle without evaluating any retrieval path."""
    root = Path(source)
    if not root.is_dir() or root.is_symlink():
        raise ReplayBundleError("replay bundle must be a regular directory")
    expected_control = {_MANIFEST_NAME, _COMMIT_NAME}
    if not expected_control.issubset({item.name for item in root.iterdir()}):
        raise ReplayBundleError("replay bundle is incomplete")
    manifest = _read_canonical_json(root / _MANIFEST_NAME, _MANIFEST_NAME)
    if not isinstance(manifest, dict):
        raise ReplayBundleError("replay bundle manifest must be an object")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ReplayBundleError("unsupported replay bundle schema")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ReplayBundleError("unsupported replay bundle schema version")
    try:
        commit = (root / _COMMIT_NAME).read_text(encoding="ascii")
    except OSError as error:
        raise ReplayBundleError("could not read replay bundle commit marker") from error
    if commit != _sha256(canonical_json_bytes(manifest)) + "\n":
        raise ReplayBundleError("replay bundle commit marker does not match manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ReplayBundleError("replay bundle manifest has no payload files")
    listed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReplayBundleError("invalid replay bundle file entry")
        name = entry.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in listed:
            raise ReplayBundleError("unsafe or duplicate replay bundle path")
        if not isinstance(entry.get("bytes"), int) or not isinstance(
            entry.get("sha256"), str
        ):
            raise ReplayBundleError("invalid replay bundle checksum entry")
        listed[name] = entry
    actual = {item.name for item in root.iterdir()}
    if actual != set(listed) | expected_control:
        raise ReplayBundleError("replay bundle has missing or unexpected files")
    for name, entry in listed.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ReplayBundleError(f"unsafe replay bundle payload {name!r}")
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or _sha256(payload) != entry["sha256"]:
            raise ReplayBundleError(f"checksum mismatch for replay payload {name!r}")
    checksums = (root / _CHECKSUMS_NAME).read_bytes()
    expected_checksums = "".join(
        f"{listed[name]['sha256']}  {name}\n"
        for name in sorted(listed)
        if name != _CHECKSUMS_NAME
    ).encode("ascii")
    if checksums != expected_checksums:
        raise ReplayBundleError("replay bundle checksums file does not match manifest")
    required = {"case.json", "geometry.json"}
    if not required.issubset(listed):
        raise ReplayBundleError("replay bundle is missing required metadata")
    documents = {
        name: _read_canonical_json(root / name, name)
        for name in sorted(listed)
        if name.endswith(".json")
    }
    arrays = {
        name.removesuffix(".npz"): _array_from_npz((root / name).read_bytes(), name)
        for name in sorted(listed)
        if name.endswith(".npz")
    }
    if not arrays:
        raise ReplayBundleError("replay bundle is missing array payloads")
    return {"manifest": manifest, "documents": documents, "arrays": arrays}
