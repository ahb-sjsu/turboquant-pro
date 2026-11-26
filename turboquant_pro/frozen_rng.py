# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""The frozen random generator that TQE1 rotations are drawn from.

A TQE1 record stores a seed, not its rotation: every decode regenerates the rotation
from the seed (``docs/FORMAT_SPEC.md``, "Rotation derivation"). Up to 2.0.0a3 the
draws came from ``np.random.default_rng(seed)``, and NumPy's policy for its random
module (NEP 19) allows ``Generator`` streams to change in a feature release. This
module fixes the draws inside turboquant-pro. It reproduces, bit for bit, the
``default_rng`` operations the rotations use, so a stored record decodes to the same
vector whatever NumPy is installed:

- ``SeedSequence(seed)`` mixing, for a non-negative integer seed;
- ``PCG64``: the 128-bit LCG, its XSL-RR output, and the buffer that gives a 32-bit
  request the low half of a 64-bit output and keeps the high half for the next one;
- ``standard_normal``: NumPy's 256-layer ziggurat with NumPy's tables
  (:mod:`turboquant_pro._ziggurat_tables`);
- ``choice([-1.0, 1.0], size)``: Lemire's bounded draw over two values, which takes
  one 32-bit draw per value and gives -1.0 when its top bit is 0 (:meth:`signs`);
- ``permutation(n)``: Fisher-Yates from the last element down, each index drawn by
  masked rejection on 32-bit draws.

The algorithms are ported from NumPy 2.4.6 (``bit_generator.pyx``, ``pcg64.h``,
``distributions.c``, ``_generator.pyx``). ``rust/tqformat/src/rng.rs`` states the same
algorithms in Rust, and both are pinned by the same known answers.

How it stays fast: the 128-bit state is a Python int. A long run of 64-bit outputs
is taken from NumPy's C implementation of the same PCG64, set to this generator's
state, but only after it is checked to agree (``_checked_numpy_pcg64``); this is the
bit generator, not NumPy's ``Generator`` streams, and nothing else is taken from
NumPy's random module. Without it the outputs are computed with NumPy arrays: the
state ``k`` steps ahead is ``A_k * s + C_k (mod 2**128)``, so a block of outputs is
one vectorized multiply-add on 64-bit halves (uint64 arithmetic wraps mod 2**64),
bit-identical and about ten times slower (1.3 s against 0.12 s for the 17 million
outputs of a 4096-dimension rotation, measured on one Xeon E5-2690 v3 core). The
ziggurat's common case
(one draw, accepted) is evaluated on whole arrays; only its rejections, under 1% of
draws, are walked one at a time. The
rejection tests call ``math.exp`` and ``math.log1p``, the C library functions that
NumPy's C code calls.
"""

from __future__ import annotations

import math
import operator

import numpy as np

from ._ziggurat_tables import (
    FI_DOUBLE,
    KI_DOUBLE,
    WI_DOUBLE,
    ZIGGURAT_NOR_INV_R,
    ZIGGURAT_NOR_R,
)

__all__ = ["FrozenGenerator", "default_rng"]

_KI = np.array(KI_DOUBLE, dtype=np.uint64)
_WI = np.array(WI_DOUBLE, dtype=np.float64)
_FI = np.array(FI_DOUBLE, dtype=np.float64)


# ---- BEGIN frozen generator: the same text in turboquant_pro/frozen_rng.py and
# ---- contrib/tqe1_reader.py (tests/test_frozen_rng.py compares the two). It reads
# ---- the ziggurat tables _KI, _WI, _FI, ZIGGURAT_NOR_R, ZIGGURAT_NOR_INV_R.
_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1
_M128 = (1 << 128) - 1

# SeedSequence (bit_generator.pyx)
_INIT_A = 0x43B0D7E5
_MULT_A = 0x931E8875
_INIT_B = 0x8B51F9DD
_MULT_B = 0x58F38DED
_MIX_MULT_L = 0xCA01F9DD
_MIX_MULT_R = 0x4973F715
_XSHIFT = 16
_POOL_SIZE = 4

# PCG64 (pcg64.h): PCG_DEFAULT_MULTIPLIER_128
_PCG_MULT = (2549297995355413924 << 64) | 4865540595714422341

_TWO_M53 = 1.0 / 9007199254740992.0  # next_double: (next_uint64 >> 11) * 2**-53
_WI_SIGNED = np.concatenate([_WI, -_WI])  # indexed by a draw's low 9 bits
_KI_TWICE = np.concatenate([_KI, _KI])

# Vectorized output: blocks of _BLOCK outputs share one start state; _ROWS blocks
# are computed per array operation.
_BLOCK = 1024
_ROWS = 64
_U32 = np.uint64(0xFFFFFFFF)
_SH32 = np.uint64(32)


# ------------------------------------------------------------------ SeedSequence
def _hashmix(value: int, hc: int) -> tuple[int, int]:
    value = (value ^ hc) & _M32
    hc = (hc * _MULT_A) & _M32
    value = (value * hc) & _M32
    return value ^ (value >> _XSHIFT), hc


def _mix(x: int, y: int) -> int:
    r = (_MIX_MULT_L * x - _MIX_MULT_R * y) & _M32
    return r ^ (r >> _XSHIFT)


def _seed_pool(seed: int) -> list[int]:
    """``SeedSequence(seed).pool`` for a non-negative int seed and no spawn key."""
    words = [0] if seed == 0 else []
    n = seed
    while n > 0:  # _int_to_uint32_array: 32-bit words, lowest first
        words.append(n & _M32)
        n >>= 32
    hc = _INIT_A
    pool = []
    for i in range(_POOL_SIZE):
        v, hc = _hashmix(words[i] if i < len(words) else 0, hc)
        pool.append(v)
    for src in range(_POOL_SIZE):
        for dst in range(_POOL_SIZE):
            if src != dst:
                h, hc = _hashmix(pool[src], hc)
                pool[dst] = _mix(pool[dst], h)
    for w in words[_POOL_SIZE:]:
        for dst in range(_POOL_SIZE):
            h, hc = _hashmix(w, hc)
            pool[dst] = _mix(pool[dst], h)
    return pool


def _generate_state_u64(pool: list[int], n_words: int) -> list[int]:
    """``SeedSequence.generate_state(n_words, np.uint64)``."""
    hc = _INIT_B
    w = []
    for i in range(2 * n_words):
        d = pool[i % _POOL_SIZE] ^ hc
        hc = (hc * _MULT_B) & _M32
        d = (d * hc) & _M32
        w.append(d ^ (d >> _XSHIFT))
    return [w[2 * k] | (w[2 * k + 1] << 32) for k in range(n_words)]


# ------------------------------------------------------------------ PCG64
def _output(state: int) -> int:
    """XSL-RR: the 64-bit output of a 128-bit state."""
    v = ((state >> 64) ^ state) & _M64
    r = state >> 122
    return ((v >> r) | (v << ((-r) & 63))) & _M64


def _jump(steps: int, inc: int) -> tuple[int, int]:
    """``(A, C)`` with ``state_after_steps = (A * state + C) mod 2**128``."""
    acc_mult, acc_plus = 1, 0
    cur_mult, cur_plus = _PCG_MULT, inc
    while steps > 0:
        if steps & 1:
            acc_mult = (acc_mult * cur_mult) & _M128
            acc_plus = (acc_plus * cur_mult + cur_plus) & _M128
        cur_plus = ((cur_mult + 1) * cur_plus) & _M128
        cur_mult = (cur_mult * cur_mult) & _M128
        steps >>= 1
    return acc_mult, acc_plus


def _halves(values: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """128-bit ints as two uint64 arrays: the high and the low 64 bits."""
    return (
        np.array([v >> 64 for v in values], dtype=np.uint64),
        np.array([v & _M64 for v in values], dtype=np.uint64),
    )


def _block_outputs(s_hi, s_lo, a_hi, a_lo, a0, a1, c_hi, c_lo) -> np.ndarray:
    """Outputs of the states ``a * s + c (mod 2**128)``, broadcast.

    ``s_hi``, ``s_lo`` are ``(rows, 1)`` block start states; ``a_*`` and ``c_*`` are
    ``(1, B)``, the multiplier and increment ``r`` steps ahead (``r = 1..B``), with
    ``a0``, ``a1`` the 32-bit halves of ``a_lo``. uint64 arithmetic wraps mod 2**64,
    which gives the low 64 bits of every product directly; only the high half of
    ``a_lo * s_lo`` is assembled from 32-bit pieces, whose partial sums fit.
    """
    m = _U32
    s0 = s_lo & m
    s1 = s_lo >> _SH32
    lo = a_lo * s_lo + c_lo
    carry = lo < c_lo
    p00 = a0 * s0
    p01 = a0 * s1
    p10 = a1 * s0
    mid = (p00 >> _SH32) + (p01 & m) + (p10 & m)
    hi = a1 * s1 + (p01 >> _SH32) + (p10 >> _SH32) + (mid >> _SH32)  # mulhi
    hi += a_lo * s_hi + a_hi * s_lo + c_hi
    hi += carry
    v = hi ^ lo
    r = hi >> np.uint64(58)  # state >> 122
    return (v >> r) | (v << ((np.uint64(64) - r) & np.uint64(63)))


# When True, long runs of raw 64-bit outputs come from NumPy's PCG64 bit generator,
# set to this generator's own state, provided it agrees with the PCG64 above (see
# _checked_numpy_pcg64); otherwise from _block_outputs. Seeding and every
# distribution stay here either way. Tests set it to False to check the pure path.
ACCELERATE = True
_CHECKED_OUTPUTS = 16


def _checked_numpy_pcg64(state: int, inc: int, n: int) -> np.ndarray | None:
    """``n`` outputs from ``np.random.PCG64`` set to ``(state, inc)``, if it agrees.

    PCG64 is a fixed published algorithm (a 128-bit LCG with the XSL-RR output), and
    NumPy's C implementation of it is far faster than the array code above. It is
    used only when its first outputs and its last output (computed here by jumping
    ahead) equal this module's; otherwise this returns None and the pure
    implementation is used, so decoding never depends on it.
    """
    try:
        bg = np.random.PCG64()
        bg.state = {
            "bit_generator": "PCG64",
            "state": {"state": state, "inc": inc},
            "has_uint32": 0,
            "uinteger": 0,
        }
        raw = np.asarray(bg.random_raw(n), dtype=np.uint64)
    except Exception:  # noqa: BLE001 - any failure means: do not use it
        return None
    if raw.shape != (n,):
        return None
    s = state
    for k in range(min(n, _CHECKED_OUTPUTS)):
        s = (s * _PCG_MULT + inc) & _M128
        if int(raw[k]) != _output(s):
            return None
    a, c = _jump(n, inc)
    if int(raw[-1]) != _output((a * state + c) & _M128):
        return None
    return raw


# ------------------------------------------------------------------ ziggurat
def _ziggurat(u: np.ndarray, n: int) -> tuple[np.ndarray, int] | None:
    """The first ``n`` normals NumPy's ziggurat makes from the 64-bit draws ``u``.

    Returns ``(normals, draws_used)``, or ``None`` when ``u`` is too short.
    ``random_standard_normal`` takes one draw per normal unless the draw falls
    outside its layer's rectangle: then a wedge test takes one more draw (and on a
    reject the loop starts again with a fresh draw), and the base layer's tail takes
    pairs of draws until one is accepted.
    """
    m = len(u)
    x = np.empty(m, dtype=np.float64)
    fast = np.empty(m, dtype=bool)
    step = 1 << 16
    for a in range(0, m, step):
        uc = u[a : a + step]
        # bits 0-7 pick the layer and bit 8 the sign, so one lookup in the 512-entry
        # signed table gives +-wi[idx] (rabs * -w is exactly -(rabs * w))
        layer = (uc & np.uint64(0x1FF)).astype(np.intp)
        rabs = (uc >> np.uint64(9)) & np.uint64(0x000FFFFFFFFFFFFF)
        np.multiply(rabs.astype(np.float64), _WI_SIGNED[layer], out=x[a : a + step])
        np.less(rabs, _KI_TWICE[layer], out=fast[a : a + step])

    slow = np.flatnonzero(~fast)
    idx_s = (u[slow] & np.uint64(0xFF)).astype(np.intp)
    # Every slow draw's wedge test, as if it started a sample; the walk below uses
    # only the ones that do. idx 0 (the tail) reads _FI[-1] here and is not used.
    nxt = u[np.minimum(slow + 1, m - 1)]
    d = (nxt >> np.uint64(11)).astype(np.float64) * _TWO_M53
    lhs = (_FI[idx_s - 1] - _FI[idx_s]) * d + _FI[idx_s]
    xs = x[slow]
    arg = (-0.5 * xs) * xs
    rhs = [math.exp(v) for v in arg.tolist()]
    wedge_ok = [lo < hi for lo, hi in zip(lhs.tolist(), rhs)]

    drop: list[int] = []  # draws that do not start an emitted normal
    tails: dict[int, float] = {}
    ends: dict[int, int] = {}  # slow start -> one past its last draw
    limit = m
    p = 0  # the first draw not yet consumed
    for j, f in enumerate(slow.tolist()):
        if f < p:
            continue  # consumed by the previous sample
        if idx_s[j]:
            if f + 1 >= m:
                limit = f
                break
            drop.append(f + 1)
            if wedge_ok[j]:
                ends[f] = f + 2
            else:
                drop.append(f)  # rejected: the loop restarts at f + 2
            p = f + 2
            continue
        k = f + 1
        while k + 1 < m:
            xx = -ZIGGURAT_NOR_INV_R * math.log1p(-((int(u[k]) >> 11) * _TWO_M53))
            yy = -math.log1p(-((int(u[k + 1]) >> 11) * _TWO_M53))
            k += 2
            if yy + yy > xx * xx:
                break
        else:
            limit = f
            break
        tail = ZIGGURAT_NOR_R + xx
        tails[f] = -tail if (int(u[f]) >> 17) & 1 else tail
        drop.extend(range(f + 1, k))
        ends[f] = k
        p = k

    # Every draw below `limit` that is not in `drop` (ascending) starts an emitted
    # normal. The n-th of them is the least `last` with last - #(drops <= last) = n - 1.
    dropped = np.array(drop, dtype=np.intp)
    if n == 0:
        return np.zeros(0), 0
    if limit - len(dropped) < n:
        return None
    last = n - 1
    while True:
        seen = int(np.searchsorted(dropped, last, side="right"))
        if n - 1 + seen == last:
            break
        last = n - 1 + seen
    keep = np.ones(last + 1, dtype=bool)
    keep[dropped[:seen]] = False
    out = x[: last + 1][keep]
    for f, v in tails.items():
        if f <= last:
            out[f - int(np.searchsorted(dropped, f))] = v
    return out, ends.get(last, last + 1)


# ------------------------------------------------------------------ generator
class FrozenGenerator:
    """``np.random.default_rng(seed)``, frozen, for the draws TQE1 rotations make.

    Only the operations the format needs are provided, each bit-identical to its
    ``Generator`` counterpart: :meth:`standard_normal`, :meth:`signs`
    (``choice([-1.0, 1.0], size=n)``) and :meth:`permutation`.
    """

    def __init__(self, seed: int) -> None:
        seed = operator.index(seed)
        if seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed}")
        v = _generate_state_u64(_seed_pool(seed), 4)
        initstate = (v[0] << 64) | v[1]
        initseq = (v[2] << 64) | v[3]
        # pcg_setseq_128_srandom_r
        self._inc = ((initseq << 1) | 1) & _M128
        state = self._inc
        state = (state + initstate) & _M128
        self._state = (state * _PCG_MULT + self._inc) & _M128
        self._has_u32 = False
        self._u32 = 0
        self._tables: tuple | None = None

    # 64-bit draws ------------------------------------------------------------
    def _peek_u64(self, n: int) -> np.ndarray:
        """The next ``n`` 64-bit outputs, without advancing."""
        if n < 2 * _BLOCK:
            out, s = [], self._state
            for _ in range(n):
                s = (s * _PCG_MULT + self._inc) & _M128
                out.append(_output(s))
            return np.array(out, dtype=np.uint64)
        if ACCELERATE:
            raw = _checked_numpy_pcg64(self._state, self._inc, n)
            if raw is not None:
                return raw
        if self._tables is None:
            mults, incs, am, ac = [], [], _PCG_MULT, self._inc
            for _ in range(_BLOCK):  # r = 1..B steps ahead
                mults.append(am)
                incs.append(ac)
                am, ac = (am * _PCG_MULT) & _M128, (ac * _PCG_MULT + self._inc) & _M128
            a_hi, a_lo = _halves(mults)
            c_hi, c_lo = _halves(incs)
            row = (a_hi, a_lo, a_lo & _U32, a_lo >> _SH32, c_hi, c_lo)
            self._tables = (tuple(t[None, :] for t in row), mults[-1], incs[-1])
        row, a_b, c_b = self._tables
        blocks = -(-n // _BLOCK)
        starts, s = [], self._state
        for _ in range(blocks):
            starts.append(s)
            s = (a_b * s + c_b) & _M128
        out = np.empty((blocks, _BLOCK), dtype=np.uint64)
        for r in range(0, blocks, _ROWS):
            s_hi, s_lo = _halves(starts[r : r + _ROWS])
            out[r : r + _ROWS] = _block_outputs(s_hi[:, None], s_lo[:, None], *row)
        return out.reshape(-1)[:n]

    def _advance(self, n: int) -> None:
        a, c = _jump(n, self._inc)
        self._state = (a * self._state + c) & _M128

    def _take_u64(self, n: int) -> np.ndarray:
        out = self._peek_u64(n)
        self._advance(n)
        return out

    # 32-bit draws: the low half of a 64-bit output first, the high half buffered
    def _take_u32(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.uint64)
        head = []
        if self._has_u32:
            head = [self._u32]
            self._has_u32 = False
            n -= 1
        raw = self._take_u64((n + 1) // 2)
        pairs = np.empty(2 * len(raw), dtype=np.uint64)
        pairs[0::2] = raw & _U32
        pairs[1::2] = raw >> _SH32
        if n % 2:
            self._has_u32 = True
            self._u32 = int(pairs[-1])
            pairs = pairs[:-1]
        return np.concatenate([np.array(head, dtype=np.uint64), pairs])

    # Generator methods ---------------------------------------------------------
    def standard_normal(self, size: int | tuple[int, ...]) -> np.ndarray:
        """``Generator.standard_normal(size)``: float64, filled in C order."""
        shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        n = math.prod(shape)
        # Rejections take about 2.2% more draws than normals (measured: 2.20% at
        # 4096 x 4096, at most 3.4% at n = 1000); a short buffer is retried, doubled.
        m = n + n // 25 + 64
        while True:
            res = _ziggurat(self._peek_u64(m), n)
            if res is not None:
                break
            m *= 2
        values, used = res
        self._advance(used)
        return values.reshape(shape)

    def signs(self, n: int) -> np.ndarray:
        """``Generator.choice([-1.0, 1.0], size=n)``: float64 values of -1.0 and 1.0."""
        top = self._take_u32(n) >> np.uint64(31)
        return np.where(top == 0, -1.0, 1.0)

    def permutation(self, n: int) -> np.ndarray:
        """``Generator.permutation(n)``: ``shuffle(arange(n))``, as int64."""
        saved = (self._state, self._has_u32, self._u32)
        budget = 2 * n + 64
        while True:
            draws = self._take_u32(budget).tolist()
            perm = list(range(n))
            t = 0
            try:
                for i in range(n - 1, 0, -1):
                    mask = (1 << i.bit_length()) - 1  # random_interval(i)
                    v = draws[t] & mask
                    t += 1
                    while v > i:
                        v = draws[t] & mask
                        t += 1
                    perm[i], perm[v] = perm[v], perm[i]
            except IndexError:
                self._state, self._has_u32, self._u32 = saved
                budget *= 2
                continue
            # consume exactly the t draws used, as NumPy does
            self._state, self._has_u32, self._u32 = saved
            self._take_u32(t)
            return np.array(perm, dtype=np.int64)


def default_rng(seed: int) -> FrozenGenerator:
    """The frozen counterpart of ``np.random.default_rng(seed)`` for an int seed."""
    return FrozenGenerator(seed)


# ---- END frozen generator
