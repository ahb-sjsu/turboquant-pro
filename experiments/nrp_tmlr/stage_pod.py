# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Stage one model onto a volume set, in a small CPU pod (1 CPU, 2 GiB).

Each file of the plan's checkpoint is downloaded from the Hub at the plan's revision,
one at a time, and checked against the hash Atlas's cache holds for it. A weight file
is then written into the stripes (stripe.py) and deleted; the small files go to the
config directory. Last, every stripe must hash to the reference stripe written on
Atlas from the same checkpoint (prep.py stripes), and the read speed a GPU pod will see
(all stripes at once) is measured. A finished source file is marked, so a pod that
dies resumes where it stopped.

    python stage_pod.py --plan plan_mistral.json --root mistral --vols /w0,/w1,/w2,/w3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import prep  # noqa: E402
import stripe  # noqa: E402

CHUNK = 64 << 20


def file_hashes(path: str) -> tuple[str, str]:
    """(sha256, git blob sha1) of a file, read in pieces."""
    h = hashlib.sha256()
    g = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                return h.hexdigest(), g.hexdigest()
            h.update(b)
            g.update(b)


def fetch(repo: str, rev: str, name: str, want: str, tmp: str) -> str:
    from huggingface_hub import hf_hub_download

    for attempt in range(3):
        try:
            p = hf_hub_download(repo, name, revision=rev, local_dir=tmp)
            break
        except Exception as e:  # a transient Hub error: try again, then fail
            print(f"  {name}: try {attempt + 1} failed ({type(e).__name__}: {e})")
            if attempt == 2:
                raise
    sha, sha1 = file_hashes(p)
    if want not in (sha, sha1):
        raise SystemExit(f"{name}: sha256 {sha} / git sha1 {sha1}, expected {want}")
    return p


def read_speed(paths: list[str]) -> float:
    def read(path):
        n = 0
        with open(path, "rb", buffering=0) as f:
            while True:
                b = f.read(CHUNK)
                if not b:
                    return n
                if stripe.DONTNEED is not None:
                    os.posix_fadvise(f.fileno(), n, len(b), stripe.DONTNEED)
                n += len(b)

    t = time.time()
    with ThreadPoolExecutor(len(paths)) as ex:
        n = sum(ex.map(read, paths))
    return n / 1e6 / max(time.time() - t, 1e-3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--root", required=True, help="directory name on every volume")
    ap.add_argument("--vols", required=True)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    vols = a.vols.split(",")
    if len(vols) != plan["k"]:
        raise SystemExit(f"{len(vols)} volumes for {plan['k']} stripes")
    dests = [os.path.join(v, a.root, "s") for v in vols]
    mdir = os.path.join(vols[0], a.root, "model")
    done = os.path.join(vols[0], a.root, ".done")
    staged = os.path.join(vols[0], a.root, "STAGED.json")
    if os.path.exists(staged):
        print("already staged", staged)
        return 0
    os.makedirs(done, exist_ok=True)
    t0 = time.time()
    small = tempfile.mkdtemp(prefix="small-")
    for name, want in plan["source_hashes"].items():
        if name == "model.safetensors.index.json":
            continue  # replaced by the stripes' own index
        if name.endswith(".safetensors"):
            if os.path.exists(os.path.join(done, name)):
                print(f"  {name}: striped earlier")
                continue
            tmp = tempfile.mkdtemp(prefix="dl-")
            t1 = time.time()
            p = fetch(plan["repo"], plan["revision"], name, want, tmp)
            size = os.path.getsize(p)
            t2 = time.time()
            n = stripe.write(plan, p, dests)
            shutil.rmtree(tmp, ignore_errors=True)
            open(os.path.join(done, name), "w").write(want)
            print(
                f"  {name}: {size / 1e9:.2f} GB downloaded and verified in"
                f" {t2 - t1:.0f}s, {n / 1e9:.2f} GB striped in {time.time() - t2:.0f}s",
                flush=True,
            )
        elif "config" not in plan:  # Gemma's text config comes from the plan
            fetch(plan["repo"], plan["revision"], name, want, small)
    prep.write_model_dir(plan, small, mdir)
    got = stripe.stripe_hashes(plan, dests)
    if got != plan["sha256"]:
        raise SystemExit(f"stripes differ from Atlas's: {got} vs {plan['sha256']}")
    paths = [os.path.join(d, s["file"]) for d, s in zip(dests, plan["stripes"])]
    mbps = read_speed(paths)
    rec = {
        "repo": plan["repo"],
        "revision": plan["revision"],
        "stripes_sha256": got,
        "seconds": round(time.time() - t0, 1),
        "read_mb_per_s_all_stripes": round(mbps, 1),
    }
    json.dump(rec, open(staged + ".part", "w"), indent=1)
    os.replace(staged + ".part", staged)
    print("STAGED", plan["repo"], json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
