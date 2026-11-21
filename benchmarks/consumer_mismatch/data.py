"""The three consumers of Part IV: text domains, staged as train.txt and test.txt each.

    python -m consumer_mismatch.data --dest /archive/experiments/consumer-mismatch/text

- ``wiki``     WikiText-2 raw (the text of Parts III and III-c), standard splits.
- ``code``     Python source, codeparrot/codeparrot-clean-valid. The first 60% of files is
               train and the rest test, in the dataset's own order.
- ``reviews``  IMDB movie reviews, standard train and test splits, in the dataset's order.

Documents are joined by a blank line, as in Part III. Each split is far longer than the
token counts drawn from it (section 1 of the prereg), so the draws are disjoint and never wrap.
"""

from __future__ import annotations

import argparse
import os

DOMAINS = ("wiki", "code", "reviews")


def _write(dest: str, name: str, docs) -> None:
    tmp = os.path.join(dest, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n\n".join(d for d in docs if d and d.strip()))
    os.replace(tmp, os.path.join(dest, name))


def stage(dest: str) -> None:
    from datasets import load_dataset

    for dom in DOMAINS:
        out = os.path.join(dest, dom)
        if os.path.exists(os.path.join(out, "test.txt")):
            print(f"{dom}: exists")
            continue
        os.makedirs(out, exist_ok=True)
        if dom == "wiki":
            for sp in ("train", "test"):
                _write(
                    out,
                    f"{sp}.txt",
                    load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)["text"],
                )
        elif dom == "code":
            d = load_dataset("codeparrot/codeparrot-clean-valid", split="train")[
                "content"
            ]
            cut = int(0.6 * len(d))
            _write(out, "train.txt", d[:cut])
            _write(out, "test.txt", d[cut:])
        else:
            for sp in ("train", "test"):
                _write(
                    out, f"{sp}.txt", load_dataset("stanfordnlp/imdb", split=sp)["text"]
                )
        print(f"{dom}: staged")


def sequences(tok, path: str, seq: int, start: int, count: int) -> list:
    """Token sequences ``start .. start+count-1`` of length ``seq`` from one file, in order.
    Raises if the file is too short, so a draw never silently shrinks."""
    ids = tok(open(path, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    n = len(ids) // seq
    if start + count > n:
        raise SystemExit(f"{path}: {n} sequences of {seq}, need {start + count}")
    return [ids[i * seq : (i + 1) * seq] for i in range(start, start + count)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    stage(ap.parse_args().dest)
