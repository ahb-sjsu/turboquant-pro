"""The frozen rotation generator (``turboquant_pro.frozen_rng``, issue #343).

Three kinds of check:

1. Known answers that do not depend on the installed NumPy. They are the values in
   ``rust/tqformat/src/kat.rs`` (written by NumPy 2.4.6), so the Python generator,
   the Rust reader and the stored records all answer to the same numbers.
2. Agreement with the installed ``np.random.default_rng``, draw for draw. If a NumPy
   release changes its streams, these fail. turboquant-pro itself is then unaffected
   (it no longer draws rotations from NumPy), but the change is worth knowing.
3. The copy of the generator in ``contrib/tqe1_reader.py`` is the same text, and
   ``TurboQuantPGVector`` builds its rotations from it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from turboquant_pro import _ziggurat_tables as zt
from turboquant_pro import frozen_rng
from turboquant_pro.frozen_rng import default_rng
from turboquant_pro.pgvector import TurboQuantPGVector

ROOT = pathlib.Path(__file__).parents[1]

# rust/tqformat/src/kat.rs
SEEDSEQ_0_POOL = [4265667335, 1328953910, 1320288413, 3365567143]
SEEDSEQ_42_STATE_U64 = [
    11465652750463011511,
    15382171918060459190,
    9018504550953525431,
    3703499796004394495,
]
PCG64_42_RAW = [14276969152011380360, 8095878257575067585, 15838336090824644132]
NORMAL_42_BITS = [
    4599160914825247851,
    13830734528060758654,
    4604934682996863824,
    4606647073357791030,
    13834837537376030863,
    13831915351169297352,
]
CHOICE_7_9 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
PERM_7_10 = [8, 0, 7, 1, 3, 6, 2, 4, 5, 9]
CHOICE_11_5 = [-1.0, -1.0, 1.0, -1.0, 1.0]
PERM_11_12_AFTER = [2, 8, 7, 10, 1, 9, 6, 5, 11, 4, 0, 3]
NORMAL_2026_2M_XOR_SUM = (121253686475944714, 4642226106182676400)

SEEDS = [0, 1, 7, 42, 2**32 - 1, 2**32, 2**64 + 3, 2**128 + 7]


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


# ------------------------------------------------------------------ known answers
def test_seed_sequence_known_answers():
    assert frozen_rng._seed_pool(0) == SEEDSEQ_0_POOL
    state = frozen_rng._generate_state_u64(frozen_rng._seed_pool(42), 4)
    assert state == SEEDSEQ_42_STATE_U64


def test_pcg64_known_answers():
    assert default_rng(42)._take_u64(3).tolist() == PCG64_42_RAW


def test_normal_sign_and_permutation_known_answers():
    assert _bits(default_rng(42).standard_normal(6)).tolist() == NORMAL_42_BITS
    assert default_rng(7).signs(9).tolist() == CHOICE_7_9
    assert default_rng(7).permutation(10).tolist() == PERM_7_10
    g = default_rng(11)  # 5 signs leave half a draw buffered for the permutation
    assert g.signs(5).tolist() == CHOICE_11_5
    assert g.permutation(12).tolist() == PERM_11_12_AFTER


def test_two_million_normals_known_answer():
    """Exercises the vectorized path, the wedge and about 500 tail draws."""
    z = default_rng(2026).standard_normal(2_000_000)
    xor = int(np.bitwise_xor.reduce(z.view(np.uint64)))
    total = 0.0
    for v in z.tolist():  # sequential addition, as kat.rs was summed
        total += v
    total_bits = int(np.array([total]).view(np.uint64)[0])
    assert (xor, total_bits) == NORMAL_2026_2M_XOR_SUM


def test_vectorized_outputs_match_the_scalar_recurrence():
    g = default_rng(123)
    fast = g._peek_u64(5 * frozen_rng._BLOCK + 17)
    s, slow = g._state, []
    for _ in range(len(fast)):
        s = (s * frozen_rng._PCG_MULT + g._inc) & frozen_rng._M128
        slow.append(frozen_rng._output(s))
    assert fast.tolist() == slow


def test_pure_path_matches_the_accelerated_one(monkeypatch):
    """Without NumPy's PCG64 the array implementation gives the same draws."""
    n = 6 * frozen_rng._BLOCK + 5
    fast = default_rng(31)._peek_u64(n)
    monkeypatch.setattr(frozen_rng, "ACCELERATE", False)
    assert default_rng(31)._peek_u64(n).tolist() == fast.tolist()
    z = default_rng(2026).standard_normal(300_001)
    monkeypatch.setattr(frozen_rng, "ACCELERATE", True)
    assert np.array_equal(_bits(z), _bits(default_rng(2026).standard_normal(300_001)))


def test_a_disagreeing_numpy_pcg64_is_not_used(monkeypatch):
    """If NumPy's PCG64 ever drew differently, the frozen draws would not change."""
    real = np.random.PCG64

    class Shifted:
        def __init__(self):
            self._bg = real()

        @property
        def state(self):
            return self._bg.state

        @state.setter
        def state(self, value):
            value = dict(value, state=dict(value["state"]))
            value["state"]["state"] += 1  # one step off
            self._bg.state = value

        def random_raw(self, n):
            return self._bg.random_raw(n)

    expected = default_rng(77)._peek_u64(5000).tolist()
    monkeypatch.setattr(frozen_rng.np.random, "PCG64", Shifted)
    g = default_rng(77)
    assert frozen_rng._checked_numpy_pcg64(g._state, g._inc, 5000) is None
    assert g._peek_u64(5000).tolist() == expected


def test_jump_ahead_matches_stepping():
    g = default_rng(5)
    s = g._state
    for _ in range(1000):
        s = (s * frozen_rng._PCG_MULT + g._inc) & frozen_rng._M128
    g._advance(1000)
    assert g._state == s


def test_ziggurat_tables_are_numpys():
    assert len(zt.KI_DOUBLE) == len(zt.WI_DOUBLE) == len(zt.FI_DOUBLE) == 256
    assert zt.KI_DOUBLE[0] == 0x000EF33D8025EF6A and zt.KI_DOUBLE[1] == 0
    assert zt.FI_DOUBLE[0] == 1.0
    assert zt.ZIGGURAT_NOR_R == 3.6541528853610088


def test_bad_seeds_are_refused():
    with pytest.raises(ValueError):
        default_rng(-1)
    with pytest.raises(TypeError):
        default_rng(1.5)


# ------------------------------------------------------------------ live NumPy
DIVERGED = (
    f"NumPy {np.__version__} no longer draws what TQE1 rotations are defined by "
    "(FORMAT_SPEC 'Rotation derivation'). turboquant-pro is unaffected, since it "
    "draws from frozen_rng, but code that rebuilds rotations with "
    "np.random.default_rng would now decode stored records differently."
)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [0, 1, 5, 2047, 2048, 2049, 5000, 300_001])
def test_normals_match_numpy(seed, n):
    ours = default_rng(seed).standard_normal(n)
    theirs = np.random.default_rng(seed).standard_normal(n)
    assert np.array_equal(_bits(ours), _bits(theirs)), DIVERGED


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [1, 2, 9, 64, 4097, 4100, 5003, 65535])
def test_signs_then_permutation_match_numpy(seed, n):
    """The order the structured rotation draws in; odd n carries a buffered half."""
    g, h = default_rng(seed), np.random.default_rng(seed)
    assert np.array_equal(g.signs(n), h.choice([-1.0, 1.0], size=n)), DIVERGED
    assert np.array_equal(g.permutation(n), h.permutation(n)), DIVERGED


@pytest.mark.parametrize("seed", [3, 2**32 - 1])
def test_interleaved_draws_keep_numpys_position(seed):
    g, h = default_rng(seed), np.random.default_rng(seed)
    for step in range(3):
        n = 1001 + 2 * step
        assert np.array_equal(
            _bits(g.standard_normal(n)), _bits(h.standard_normal(n))
        ), DIVERGED
        assert np.array_equal(g.signs(7), h.choice([-1.0, 1.0], size=7)), DIVERGED
        assert np.array_equal(g.permutation(13), h.permutation(13)), DIVERGED
        assert np.array_equal(
            _bits(g.standard_normal((3, 5))), _bits(h.standard_normal((3, 5)))
        ), DIVERGED


@pytest.mark.parametrize("seed", [0, 4242])
def test_dense_rotation_draws_at_4096_match_numpy(seed):
    """The largest dense draw a TQE1 record makes: 4096 x 4096 normals."""
    ours = default_rng(seed).standard_normal((4096, 4096))
    theirs = np.random.default_rng(seed).standard_normal((4096, 4096))
    assert np.array_equal(_bits(ours), _bits(theirs)), DIVERGED


# ------------------------------------------------------------------ users of it
@pytest.mark.parametrize(
    "dim,rotation", [(64, "qr"), (256, "hadamard"), (4100, "qr"), (5003, "qr")]
)
def test_pgvector_rotation_is_the_frozen_draw(dim, rotation):
    q = TurboQuantPGVector(dim=dim, bits=3, seed=99, rotation=rotation)
    g = default_rng(99)
    if rotation == "hadamard" or dim > 4096:
        assert np.array_equal(q._sign_flip, g.signs(dim).astype(np.float32))
        if dim > 4096:
            assert np.array_equal(q._perm, g.permutation(dim))
    else:
        qq, _ = np.linalg.qr(g.standard_normal((dim, dim)).astype(np.float32))
        assert np.array_equal(q._Pi, qq)


def _load_contrib_reader():
    path = ROOT / "contrib" / "tqe1_reader.py"
    spec = importlib.util.spec_from_file_location("tqe1_reader_frozen_check", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def _generator_block(path: pathlib.Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    begin = [i for i, ln in enumerate(lines) if ln.startswith("# ---- BEGIN frozen")]
    end = [i for i, ln in enumerate(lines) if ln.startswith("# ---- END frozen")]
    assert len(begin) == 1 and len(end) == 1, path
    return lines[begin[0] : end[0] + 1]


def test_contrib_reader_holds_the_same_generator():
    pkg = _generator_block(ROOT / "turboquant_pro" / "frozen_rng.py")
    contrib = _generator_block(ROOT / "contrib" / "tqe1_reader.py")
    assert contrib == pkg, (
        "contrib/tqe1_reader.py's frozen generator differs from "
        "turboquant_pro/frozen_rng.py's; copy the block between the BEGIN and END "
        "markers verbatim"
    )
    mod = _load_contrib_reader()
    assert mod._KI.tolist() == list(zt.KI_DOUBLE)
    assert _bits(mod._WI).tolist() == _bits(np.array(zt.WI_DOUBLE)).tolist()
    assert _bits(mod._FI).tolist() == _bits(np.array(zt.FI_DOUBLE)).tolist()
    assert mod.ZIGGURAT_NOR_R == zt.ZIGGURAT_NOR_R
    assert mod.ZIGGURAT_NOR_INV_R == zt.ZIGGURAT_NOR_INV_R
    g = mod.default_rng(2026)
    z = g.standard_normal(200_000)
    assert np.array_equal(_bits(z), _bits(default_rng(2026).standard_normal(200_000)))
