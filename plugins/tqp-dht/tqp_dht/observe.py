# tqp-dht: a BitTorrent DHT source for the TurboQuant Pro console
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Observation Theory, measured on a live Kademlia router.

The common object of Observation Theory (readscope ``PRINCIPLES.md``, v1.0) is a
consumer C reading a representation x through a read operator P_C. Here:

- the **representation** is a 160-bit DHT id (a target: an info hash, a node);
- the **consumer** is Kademlia routing: C(x) = the known node XOR-nearest to x,
  over the node ids this daemon has actually seen;
- the **read operator** is measured blind, as readscope does it: flip each of
  the 160 bits of x and ask whether C's decision changes. The diagonal of P_C
  at bit i is the probability, over a probing distribution D, that flipping bit
  i changes the routing decision. Routing reads the leading bits and nothing
  past about log2 N of them; the rest is ker P_C, the nuisance.

Each principle has one measurement here, each a pure function of the node set,
the probing distribution and a seeded generator, so each is tested against a
prediction stated before it is run:

- P1 consumer relativity: the read spectrum against an isotropic observer's,
  and the flip: two codes of equal reconstruction error ranked oppositely by
  two consumers (routing reads prefixes; a sharder reads suffixes);
- P2 measure dependence: the same probe under different distributions D;
- P3 observation complexity: blind identification costs d = 160 flips per
  sample, and probing k < d bits recovers k/d of the read mass, not more;
- P4 temporal nonstationarity: the read spectrum as a process, and the price of
  using an old one (history kept by the caller);
- P5 metric consequence: predicted damage (delta' P delta) against measured
  damage, for a graded consumer (the XOR distance to the nearest node) and the
  selection itself, which P5 excludes; perturbations inside the kernel should
  meet silence.

The nearest node is found exactly: ids are split into three words (bits 0-63,
64-127, 128-159) and compared lexicographically on their XOR with the target.
"""

from __future__ import annotations

import numpy as np

D_BITS = 160
_W = (64, 64, 32)  # word widths, most significant first


# ------------------------------------------------------------------ ids
def pack(ids) -> np.ndarray:
    """20-byte ids as an (n, 3) uint64 array of their three words."""
    out = np.zeros((len(ids), 3), dtype=np.uint64)
    for r, b in enumerate(ids):
        v = int.from_bytes(b, "big")
        out[r, 0] = v >> 96
        out[r, 1] = (v >> 32) & 0xFFFFFFFFFFFFFFFF
        out[r, 2] = v & 0xFFFFFFFF
    return out


def bit_mask(i: int) -> np.ndarray:
    """The three words of the id with only bit i set (bit 0 most significant)."""
    m = np.zeros(3, dtype=np.uint64)
    if i < 64:
        m[0] = np.uint64(1) << np.uint64(63 - i)
    elif i < 128:
        m[1] = np.uint64(1) << np.uint64(127 - i)
    else:
        m[2] = np.uint64(1) << np.uint64(159 - i)
    return m


_MASKS = np.stack([bit_mask(i) for i in range(D_BITS)])  # (160, 3)


def random_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    lo = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    hi = rng.integers(
        0, np.iinfo(np.uint64).max, size=(n, 2), dtype=np.uint64, endpoint=True
    )
    return np.column_stack([hi, lo])


def near(rng: np.random.Generator, center: np.ndarray, n: int, bits: int) -> np.ndarray:
    """n ids sharing at least ``bits`` leading bits with ``center``."""
    ids = random_ids(rng, n)
    keep = np.zeros(3, dtype=np.uint64)
    for i in range(bits):
        keep |= _MASKS[i]
    return (ids & ~keep) | (center & keep)


def nearest(nodes: np.ndarray, target: np.ndarray) -> int:
    """Row of the node XOR-nearest to ``target``: exact, word by word."""
    x = nodes ^ target
    cand = np.flatnonzero(x[:, 0] == x[:, 0].min())
    for w in (1, 2):
        if len(cand) == 1:
            break
        col = x[cand, w]
        cand = cand[col == col.min()]
    return int(cand[0])


def xor_bits(a: np.ndarray, b: np.ndarray) -> int:
    """Shared leading bits of two ids (160 when equal)."""
    x = a ^ b
    for w, width in enumerate(_W):
        if x[w]:
            return sum(_W[:w]) + width - int(x[w]).bit_length()
    return D_BITS


# ------------------------------------------------------------------ P1, P2
def _decisions(nodes: np.ndarray, t: np.ndarray) -> np.ndarray:
    """For one target: 161 routing decisions, unflipped then with each bit i
    flipped, exact. A flip in the top word needs its own argmin over that word;
    a flip below it can only move the decision among the nodes that tie with
    the target on the whole top word, so those are resolved on that tie set."""
    out = np.empty(D_BITS + 1, dtype=np.int64)
    out[0] = nearest(nodes, t)
    h = nodes[:, 0] ^ t[0]
    for i in range(64):
        hi = h ^ _MASKS[i][0]
        cand = np.flatnonzero(hi == hi.min())
        if len(cand) == 1:
            out[1 + i] = cand[0]
        else:
            out[1 + i] = cand[nearest(nodes[cand], t ^ _MASKS[i])]
    tie = np.flatnonzero(h == h.min())  # the only nodes a low-bit flip can choose
    sub = nodes[tie]
    for i in range(64, D_BITS):
        out[1 + i] = tie[nearest(sub, t ^ _MASKS[i])] if len(tie) > 1 else tie[0]
    return out


def read_spectrum(nodes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """diag P_C, measured: per bit, the share of targets whose routing decision
    changes when that bit alone is flipped."""
    hits = np.zeros(D_BITS)
    for t in targets:
        d = _decisions(nodes, t)
        hits += d[1:] != d[0]
    return hits / max(len(targets), 1)


def read_depth(spectrum: np.ndarray) -> float:
    """tr P_C: the expected number of bits whose flip changes the decision."""
    return float(np.sum(spectrum))


def knee(spectrum: np.ndarray) -> int:
    """The first bit read less than half the time: where the kernel begins."""
    below = np.flatnonzero(spectrum < 0.5)
    return int(below[0]) if len(below) else D_BITS


def flip(
    nodes: np.ndarray, targets: np.ndarray, rng: np.random.Generator, keep: int = 32
) -> dict:
    """P1's flip: code A keeps an id's top ``keep`` bits, code B its bottom
    ``keep`` bits, each randomising the rest, so both have the same
    reconstruction error (160 - keep random bits). Routing (reads prefixes)
    and a sharder that buckets ids by their low bits rank the codes oppositely."""
    top = np.zeros(3, dtype=np.uint64)
    bottom = np.zeros(3, dtype=np.uint64)
    for i in range(keep):
        top |= _MASKS[i]
        bottom |= _MASKS[D_BITS - 1 - i]
    noise = random_ids(rng, len(targets))
    code_a = (targets & top) | (noise & ~top)
    code_b = (targets & bottom) | (noise & ~bottom)
    shard = 1 << 16  # the sharder: the id's low 16 bits choose one of 65536 shards

    def routing(codes):
        return float(
            np.mean(
                [
                    nearest(nodes, c) == nearest(nodes, t)
                    for c, t in zip(codes, targets, strict=True)
                ]
            )
        )

    def sharding(codes):
        return float(np.mean((codes[:, 2] % shard) == (targets[:, 2] % shard)))

    return {
        "keep_bits": keep,
        "reconstruction_error_bits": D_BITS - keep,
        "routing": {"A_top": routing(code_a), "B_bottom": routing(code_b)},
        "sharding": {"A_top": sharding(code_a), "B_bottom": sharding(code_b)},
    }


# ------------------------------------------------------------------ P3
def blind_budget(
    spectrum: np.ndarray,
    rng: np.random.Generator,
    ks=(5, 10, 20, 40, 80, 120, 160),
    trials: int = 200,
) -> list:
    """The share of the read mass found when only k of the d bits are probed,
    the k chosen blind (at random): blind identification cannot aim at the read
    subspace before it has found it, so it recovers about k/d, not more."""
    mass = float(np.sum(spectrum)) or 1.0
    out = []
    for k in ks:
        got = [
            float(np.sum(spectrum[rng.choice(D_BITS, size=k, replace=False)])) / mass
            for _ in range(trials)
        ]
        out.append({"k": k, "recovered": float(np.mean(got)), "k_over_d": k / D_BITS})
    return out


# ------------------------------------------------------------------ P4
def staleness(history: list) -> list:
    """For each earlier spectrum, its distance (L1, in bits) from the latest:
    the price of using a read operator identified that long ago."""
    if len(history) < 2:
        return []
    t_now, now = history[-1]
    return [
        {"age_s": t_now - t, "l1_bits": float(np.sum(np.abs(np.asarray(s) - now)))}
        for t, s in history[:-1]
    ]


# ------------------------------------------------------------------ P5
def consequence(
    nodes: np.ndarray,
    targets: np.ndarray,
    spectrum: np.ndarray,
    rng: np.random.Generator,
    per_target: int = 4,
) -> dict:
    """Predicted damage delta' P delta (diagonal P: the sum of the flipped bits'
    read probabilities) against measured damage, for random multi-bit
    perturbations, and for perturbations confined to the kernel (bits read
    never, spectrum == 0), which should meet silence."""
    kernel = np.flatnonzero(spectrum == 0)
    rows, silent = [], {"trials": 0, "changed": 0}
    for t in targets:
        base = nearest(nodes, t)
        base_d = xor_bits(nodes[base], t)
        for _ in range(per_target):
            bits = rng.choice(D_BITS, size=int(rng.integers(1, 6)), replace=False)
            p = t.copy()
            for i in bits:
                p ^= _MASKS[i]
            new = nearest(nodes, p)
            rows.append(
                {
                    "predicted": float(np.sum(spectrum[bits])),
                    "selection": float(new != base),
                    "graded": abs(xor_bits(nodes[new], p) - base_d),
                }
            )
            if len(kernel):
                q = t.copy()
                for i in rng.choice(kernel, size=min(3, len(kernel)), replace=False):
                    q ^= _MASKS[i]
                silent["trials"] += 1
                silent["changed"] += int(nearest(nodes, q) != base)

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        if np.std(ra) == 0 or np.std(rb) == 0:
            return None
        return float(np.corrcoef(ra, rb)[0, 1])

    pred = np.array([r["predicted"] for r in rows])
    return {
        "n": len(rows),
        "rho_graded": spearman(pred, np.array([r["graded"] for r in rows])),
        "rho_selection": spearman(pred, np.array([r["selection"] for r in rows])),
        "kernel_silence": silent,
    }
