"""The Observation Theory instrument on synthetic node sets, where its answers
are known before it is run: exact against brute force, the knee at log2 N,
one more bit of depth per doubling, blind to relabelling, the flip."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tqp_dht import observe as O


def _ints(ids: np.ndarray) -> list:
    return [(int(r[0]) << 96) | (int(r[1]) << 32) | int(r[2]) for r in ids]


def test_the_routing_decisions_are_exact_against_brute_force():
    rng = np.random.default_rng(1)
    nodes = O.random_ids(rng, 300)
    # ties on the top word, so the low words must decide
    nodes[:40, 0] = nodes[0, 0]
    targets = np.vstack([O.random_ids(rng, 20), nodes[:5] ^ np.uint64(3)])
    ni = _ints(nodes)
    for t in targets:
        ti = _ints(t[None, :])[0]
        got = O._decisions(nodes, t)
        for i in range(-1, O.D_BITS):
            x = ti if i < 0 else ti ^ (1 << (159 - i))
            want = min(range(len(ni)), key=lambda r: ni[r] ^ x)
            assert got[1 + i] == want, (i, got[1 + i], want)


def test_pack_and_shared_prefix_agree_with_the_bytes():
    a, b = bytes(20), bytes([0, 0, 0x10]) + bytes(17)
    p = O.pack([a, b])
    assert O.xor_bits(p[0], p[1]) == 19 and O.xor_bits(p[0], p[0]) == 160
    assert _ints(p) == [0, int.from_bytes(b, "big")]


@pytest.fixture(scope="module")
def uniform():
    rng = np.random.default_rng(7)
    out = {}
    for n in (4096, 8192):
        nodes = O.random_ids(rng, n)
        out[n] = (nodes, O.read_spectrum(nodes, O.random_ids(rng, 256)))
    return out


def test_routing_reads_the_leading_bits_and_stops_near_log2_n(uniform):
    nodes, s = uniform[4096]
    lg = math.log2(4096)
    assert abs(O.knee(s) - lg) <= 3
    assert s[: int(lg) - 4].min() >= 0.9  # read almost always
    assert s[int(lg) + 6 :].max() <= 0.05  # the kernel: never read
    assert np.all(s[64:] == 0)  # no top-word ties among random ids


def test_doubling_the_network_reads_one_more_bit(uniform):
    """Metamorphic: tr P_C grows by log2 2 = 1 bit when N doubles."""
    d1, d2 = O.read_depth(uniform[4096][1]), O.read_depth(uniform[8192][1])
    assert abs((d2 - d1) - 1.0) <= 0.4


def test_relabelling_every_id_by_one_mask_changes_nothing():
    rng = np.random.default_rng(3)
    nodes, targets = O.random_ids(rng, 1000), O.random_ids(rng, 64)
    mask = O.random_ids(rng, 1)[0]
    assert np.array_equal(
        O.read_spectrum(nodes, targets), O.read_spectrum(nodes ^ mask, targets ^ mask)
    )


def test_targets_near_a_node_share_its_leading_bits():
    rng = np.random.default_rng(4)
    c = O.random_ids(rng, 1)[0]
    assert all(O.xor_bits(x, c) >= 16 for x in O.near(rng, c, 50, 16))


def test_the_flip_two_codes_equal_in_error_ranked_oppositely():
    rng = np.random.default_rng(5)
    nodes, targets = O.random_ids(rng, 4096), O.random_ids(rng, 200)
    f = O.flip(nodes, targets, rng, keep=32)
    assert f["reconstruction_error_bits"] == 128  # the same for both codes
    assert f["routing"]["A_top"] >= 0.95 and f["routing"]["B_bottom"] <= 0.05
    assert f["sharding"]["B_bottom"] == 1.0 and f["sharding"]["A_top"] <= 0.05


def test_blind_probing_recovers_k_over_d_of_the_read_mass(uniform):
    """An identity for blind (random) probe sets, printed as such: nothing
    cheaper than d probes finds the read subspace before it is known."""
    rng = np.random.default_rng(6)
    for row in O.blind_budget(uniform[4096][1], rng, trials=400):
        assert abs(row["recovered"] - row["k_over_d"]) <= 0.03


def test_perturbations_in_the_kernel_meet_silence(uniform):
    nodes, s = uniform[4096]
    rng = np.random.default_rng(8)
    r = O.consequence(nodes, O.random_ids(rng, 64), s, rng)
    assert r["kernel_silence"]["trials"] == 64 * 4  # one per perturbation
    assert r["kernel_silence"]["changed"] == 0
    assert r["rho_graded"] is not None and r["rho_graded"] > 0.3


def test_staleness_is_the_distance_from_the_latest_spectrum():
    a, b = np.zeros(160), np.ones(160)
    st = O.staleness([(0.0, a), (10.0, a), (20.0, b)])
    assert [x["age_s"] for x in st] == [20.0, 10.0] and st[0]["l1_bits"] == 160.0
    assert O.staleness([(0.0, a)]) == []
