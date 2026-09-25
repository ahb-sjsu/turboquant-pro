"""Stage WikiText-2 (raw) onto the volume as train.txt and test.txt, from a CPU pod.

    python -m weight_observer.stage_text --dest /data/wo/text

The same text the kvquant perplexity harness reads (wikitext-2-raw-v1, lines joined by a
blank line), so the calibration and evaluation splits are the standard ones.
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    if os.path.exists(os.path.join(a.dest, "test.txt")):
        print("text exists")
        return 0
    from datasets import load_dataset

    for split in ("train", "test"):
        d = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        tmp = os.path.join(a.dest, f"{split}.txt.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n\n".join(d["text"]))
        os.replace(tmp, os.path.join(a.dest, f"{split}.txt"))
    print("TEXT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
