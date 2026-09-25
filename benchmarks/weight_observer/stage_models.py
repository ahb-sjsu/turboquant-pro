"""Stage one model's weights onto the volume from a 1-CPU, 2 GiB pod (vendored from GET G3c).

    python stage_models.py --model-id Qwen/Qwen2.5-1.5B-Instruct --revision REV --dest /data/models/qwen1p5b

Two earlier attempts failed on memory, both recorded in PREREG-G3C.md: the Hugging Face
chunked downloader fetching seven files at once was OOM-killed at 2 GiB, and buffered writes to
the Ceph volume let unflushed pages pile up against the limit and crawl at 0.26 MB/s. So every
file goes to the pod's local disk first, one at a time with the chunked downloader off, and is
then copied to the volume in 64 MB pieces, each flushed and dropped from the page cache before
the next. Ceph takes direct writes at about 21 MB/s on these nodes. A manifest records the
resolved revision and each file's size and sha256, and the read speed of the largest file is
measured last, because the GPU pods will load from here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

PATTERNS = (".json", ".safetensors", ".txt", ".model", ".jinja", ".tiktoken")
CHUNK = 64 << 20


def copy_bounded(src: Path, dst: Path) -> tuple[int, str]:
    """Copy with bounded page cache: write a chunk, fsync, drop it, next chunk. Returns size, sha256."""
    h = hashlib.sha256()
    n = 0
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
            fo.flush()
            os.fsync(fo.fileno())
            os.posix_fadvise(fo.fileno(), n, len(b), os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fi.fileno(), n, len(b), os.POSIX_FADV_DONTNEED)
            h.update(b)
            n += len(b)
    os.replace(tmp, dst)
    return n, h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "STAGED.json").exists():
        print("already staged", dest)
        return 0
    api = HfApi()
    info = api.model_info(a.model_id, revision=a.revision)
    files = [
        f
        for f in api.list_repo_files(a.model_id, revision=info.sha)
        if f.endswith(PATTERNS)
    ]
    local = Path("/tmp/dl")
    manifest = {"model_id": a.model_id, "revision": info.sha, "files": {}}
    t0 = time.time()
    total = 0
    for f in sorted(files):
        p = Path(
            hf_hub_download(a.model_id, f, revision=info.sha, local_dir=str(local))
        )
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        size, sha = copy_bounded(p, dest / f)
        manifest["files"][f] = {"bytes": size, "sha256": sha}
        total += size
        p.unlink()
        print(
            f"{f} {size / 1e6:.0f} MB written at {size / 1e6 / max(time.time() - t1, 1e-3):.1f} MB/s",
            flush=True,
        )
    shutil.rmtree(local, ignore_errors=True)
    manifest["seconds"] = round(time.time() - t0, 1)
    manifest["bytes"] = total
    big = max(manifest["files"], key=lambda k: manifest["files"][k]["bytes"])
    t1 = time.time()
    n = 0
    with open(dest / big, "rb") as fi:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            n += len(b)
            os.posix_fadvise(fi.fileno(), n - len(b), len(b), os.POSIX_FADV_DONTNEED)
    manifest["read_mb_per_s"] = round(n / 1e6 / max(time.time() - t1, 1e-3), 1)
    json.dump(manifest, open(dest / "STAGED.json.part", "w"), indent=1)
    os.replace(dest / "STAGED.json.part", dest / "STAGED.json")
    print(
        "STAGED",
        a.model_id,
        info.sha,
        f"{total / 1e9:.2f} GB in {manifest['seconds']} s, read {manifest['read_mb_per_s']} MB/s",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
