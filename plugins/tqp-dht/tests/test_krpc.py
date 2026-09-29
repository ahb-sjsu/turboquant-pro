"""KRPC reading and lookup reconstruction, on packets built by hand."""

from __future__ import annotations

import pytest

from tqp_dht.krpc import (
    BencodeError,
    LookupTracer,
    bdecode,
    bencode,
    compact_nodes,
    shared_prefix,
)

OURS = bytes(20)  # our node id: all zero bits


def nid(prefix_bits: int, target: bytes, salt: int = 0) -> bytes:
    """An id sharing exactly ``prefix_bits`` leading bits with ``target``."""
    t = int.from_bytes(target, "big")
    if prefix_bits >= 160:
        return target
    flip = 1 << (159 - prefix_bits)  # the first differing bit
    low = salt % (1 << max(0, 159 - prefix_bits)) if prefix_bits < 159 else 0
    v = (t ^ flip) & ~(flip - 1) | (low & (flip - 1))
    return v.to_bytes(20, "big")


def compact(ids) -> bytes:
    return b"".join(i + bytes([10, 0, 0, 1]) + (6881).to_bytes(2, "big") for i in ids)


def query(tid, target, method=b"find_node", sender=OURS):
    key = b"target" if method == b"find_node" else b"info_hash"
    return bencode(
        {b"t": tid, b"y": b"q", b"q": method, b"a": {b"id": sender, key: target}}
    )


def response(tid, responder, nodes=()):
    r = {b"id": responder}
    if nodes:
        r[b"nodes"] = compact(nodes)
    return bencode({b"t": tid, b"y": b"r", b"r": r})


# ------------------------------------------------------------------ bencode
def test_bencode_round_trips_and_decoding_is_strict():
    v = {b"a": [1, -2, b"x", {b"k": b""}], b"z": 0}
    assert bdecode(bencode(v)) == v
    for bad in (b"i03e", b"i-0e", b"ie", b"4:abc", b"i1ee", b"di1ei2ee", b"x"):
        with pytest.raises(BencodeError):
            bdecode(bad)


def test_shared_prefix_counts_leading_equal_bits():
    t = bytes(range(20))
    assert shared_prefix(t, t) == 160
    for k in (0, 1, 7, 8, 63, 159):
        assert shared_prefix(nid(k, t), t) == k
    assert shared_prefix(b"\x80" + bytes(19), bytes(20)) == 0


def test_compact_nodes_are_26_bytes_each():
    ids = [bytes([i]) * 20 for i in range(3)]
    got = compact_nodes(compact(ids))
    assert [g[0] for g in got] == ids and got[0][1:] == ("10.0.0.1", 6881)
    with pytest.raises(BencodeError):
        compact_nodes(b"x" * 25)


# ------------------------------------------------------------------ the tracer
def test_a_lookup_records_the_best_prefix_after_each_response():
    target = bytes([0xAB]) * 20
    tr = LookupTracer({OURS})
    tr.observe(query(b"t1", target), 0.0)
    tr.observe(query(b"t2", target), 0.1)
    tr.observe(response(b"t1", nid(3, target), [nid(9, target), nid(5, target)]), 0.2)
    tr.observe(response(b"t2", nid(4, target), [nid(7, target)]), 0.3)  # no better
    tr.observe(query(b"t3", target), 0.4)
    tr.observe(response(b"t3", nid(21, target), [nid(18, target)]), 0.5)
    (lk,) = tr.lookups()
    assert lk["method"] == "find_node" and lk["target"] == target.hex()
    assert lk["queries"] == 3 and lk["responses"] == 3
    assert lk["curve"] == [9, 9, 21] and lk["best_prefix"] == 21 and lk["live"]
    assert tr.counts["responses_paired"] == 3 and tr.counts["queries_out"] == 3


def test_what_cannot_be_paired_is_counted_not_guessed():
    target = bytes([1]) * 20
    tr = LookupTracer({OURS}, timeout_s=5.0)
    tr.observe(response(b"zz", nid(30, target)), 0.0)  # no such query
    tr.observe(query(b"t1", target), 1.0)
    tr.observe(response(b"t1", nid(30, target)), 7.0)  # too late
    tr.observe(query(b"q9", target, sender=bytes([7]) * 20), 7.1)  # someone else's
    tr.observe(b"not bencode", 7.2)
    assert tr.counts["responses_unpaired"] == 2 and tr.counts["responses_paired"] == 0
    assert tr.counts["queries_in"] == 1 and tr.counts["undecodable"] == 1
    assert tr.lookups() == []  # one query to a target is a probe, not a lookup


def test_get_peers_lookups_use_the_info_hash_and_idle_ones_finish():
    ih = bytes([0x5A]) * 20
    tr = LookupTracer({OURS}, idle_s=10.0)
    tr.observe(query(b"g1", ih, method=b"get_peers"), 0.0)
    tr.observe(query(b"g2", ih, method=b"get_peers"), 0.1)
    tr.observe(query(b"g3", ih, method=b"get_peers"), 0.2)
    tr.observe(response(b"g1", nid(12, ih)), 0.5)
    tr.observe(query(b"f1", bytes(20)), 20.0)  # a later packet: the first goes idle
    tr.observe(query(b"f2", bytes(20)), 20.1)
    tr.observe(query(b"f3", bytes(20)), 20.2)
    looks = tr.lookups()
    assert [x["method"] for x in looks] == ["find_node", "get_peers"]
    assert looks[0]["live"] and not looks[1]["live"] and looks[1]["curve"] == [12]


def test_ids_chosen_near_the_target_are_counted_apart():
    """A node sharing 144 bits with an info hash was seen on the live DHT: no
    random id does that. It is counted as near-target, and the convergence and
    the best prefix are those of the ids that could have come about by chance."""
    target = bytes([0x7A]) * 20
    tr = LookupTracer({OURS})
    tr.observe(query(b"a", target), 0.0)
    tr.observe(query(b"b", target), 0.1)
    tr.observe(query(b"c", target), 0.15)
    tr.observe(response(b"a", nid(9, target), [nid(144, target), nid(41, target)]), 0.2)
    tr.observe(response(b"b", nid(40, target)), 0.3)  # 40 is still plausible
    (lk,) = tr.lookups()
    assert lk["curve"] == [9, 40] and lk["best_prefix"] == 40
    assert lk["near_target"] == 2


def test_one_query_probes_are_counted_not_listed():
    """libtorrent refreshes a bucket with a single get_peers to one node: a
    probe, not a walk toward a target."""
    tr = LookupTracer({OURS}, idle_s=10.0)
    for i in range(5):
        tr.observe(query(bytes([i]), bytes([i]) * 20, method=b"get_peers"), i)
    tr.observe(query(b"x", bytes([9]) * 20), 30.0)  # the five go idle
    assert tr.lookups() == [] and tr.counts["probes"] == 5


def test_relabelling_every_id_by_one_mask_changes_no_curve():
    """Metamorphic: XOR with a common mask preserves every pairwise XOR, so every
    shared prefix, so every curve."""
    mask = bytes(range(40, 60))

    def x(b):
        return bytes(p ^ q for p, q in zip(b, mask, strict=True))

    target = bytes([0x33]) * 20
    steps = [(3, [9, 5]), (4, [7]), (21, [18, 2])]
    curves = []
    for relabel in (lambda b: b, x):
        tr = LookupTracer({relabel(OURS)})
        for i, (r, ns) in enumerate(steps):
            tid = bytes([i])
            tr.observe(query(tid, relabel(target), sender=relabel(OURS)), i)
            nodes = [relabel(nid(k, target, salt=i)) for k in ns]
            tr.observe(response(tid, relabel(nid(r, target)), nodes), i + 0.5)
        curves.append(tr.lookups()[0]["curve"])
    assert curves[0] == curves[1] == [9, 9, 21]
