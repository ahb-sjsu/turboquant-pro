#!/usr/bin/env python3
"""Offline structural validation for the NRP Job manifests.

Used when a live API server is unreachable (e.g. the kube context requires an
interactive OIDC browser login, so `kubectl apply --dry-run=server` cannot run
non-interactively). This is NOT a substitute for server-side dry-run -- it does
not check against the cluster's OpenAPI schema or admission webhooks. It does
catch YAML errors and missing/misshaped required fields before you launch.

Usage:  python validate.py job_baseline.yaml job_latency.yaml
Note:   templates may still contain __PLACEHOLDER__ tokens; those are reported,
        not failed (launch.sh substitutes them at apply time).
"""
import sys

import yaml

REQUIRED_JOB = ["apiVersion", "kind", "metadata", "spec"]


def check(path):
    errs, warns = [], []
    with open(path) as f:
        text = f.read()
    try:
        docs = [d for d in yaml.safe_load_all(text) if d]
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"], []
    if not docs:
        return ["no documents parsed"], []
    for doc in docs:
        for k in REQUIRED_JOB:
            if k not in doc:
                errs.append(f"missing top-level '{k}'")
        if doc.get("kind") != "Job":
            warns.append(f"kind={doc.get('kind')} (expected Job)")
        spec = doc.get("spec", {})
        tmpl = spec.get("template", {}).get("spec", {})
        conts = tmpl.get("containers", [])
        if not conts:
            errs.append("spec.template.spec.containers empty")
        for c in conts:
            if "image" not in c:
                errs.append(f"container {c.get('name')} missing image")
            res = c.get("resources", {})
            gpu = res.get("limits", {}).get("nvidia.com/gpu")
            if gpu is None:
                warns.append(f"container {c.get('name')} has no nvidia.com/gpu limit")
        if tmpl.get("restartPolicy") not in ("Never", "OnFailure"):
            errs.append(f"restartPolicy={tmpl.get('restartPolicy')} (Job needs Never/OnFailure)")
        if "activeDeadlineSeconds" not in spec:
            warns.append("no activeDeadlineSeconds (NRP etiquette: set a wall-clock cap)")
        # PVC mount sanity
        vols = {v["name"] for v in tmpl.get("volumes", [])}
        for c in conts:
            for m in c.get("volumeMounts", []):
                if m["name"] not in vols:
                    errs.append(f"mount '{m['name']}' has no matching volume")
    if "__" in text:
        warns.append("contains __PLACEHOLDER__ tokens (template -- substitute before apply)")
    return errs, warns


def main():
    rc = 0
    for path in sys.argv[1:]:
        errs, warns = check(path)
        status = "FAIL" if errs else "OK"
        print(f"[{status}] {path}")
        for w in warns:
            print(f"    warn: {w}")
        for e in errs:
            print(f"    ERROR: {e}")
        if errs:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
