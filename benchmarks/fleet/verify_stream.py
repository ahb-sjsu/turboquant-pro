# Verify the streaming build path against the block path, at fleet dims.
#
# The invariant that MUST hold bitwise is the corpus: concat(gen_block_bands)
# == gen_block, byte for byte, or the 1T points stop being comparable with the
# published 1B/10B/100B runs.
#
# The shard files themselves are held to the honest, measured standard: BLAS
# summation order is shape-dependent (one 5M-row GEMM vs band GEMMs), so the
# projection wobbles by a few float32 ULP — the same class of variation a
# different thread count or node already produces between two block builds.
# Consequences, measured 2026-08-04 on shards 1/7/199 at 300k rows:
#   - codes: 0-3 slots per 7.2M flip, every one by exactly +-1 level
#     (a value within an ULP of a searchsorted boundary lands either side)
#   - cnorm/vrnorm: <=5 ULP, <=2.2e-6 relative
# Gates below allow 10x margin on those measurements and nothing more.
#
# Run on Atlas: PYTHONPATH=<dir with fleet_common + turboquant_pro> python3 verify_stream.py
import json
import os
import shutil
import sys
import tempfile

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")

from fleet_common import gen_block, gen_block_bands  # noqa: E402

from turboquant_pro import ShardedIndex, TQEIndex  # noqa: E402
from turboquant_pro.index_file import read_container  # noqa: E402

ROWS = 300_000
SHARDS = [1, 7, 199]

work = tempfile.mkdtemp(prefix="tqp_stream_verify_")
try:
    # Basis: shard 0 at the fleet's config (dim 32 -> out 24, 4 bits).
    b0 = gen_block(0, rows=ROWS)
    base_dir = os.path.join(work, "base")
    ShardedIndex.write_shard(base_dir, b0, 0, 0, output_dim=24, bits=4)
    basis = os.path.join(base_dir, "shard_00000.tqe")

    for g in SHARDS:
        blk = gen_block(g, rows=ROWS)
        cat = np.concatenate(list(gen_block_bands(g, rows=ROWS)))
        assert cat.dtype == blk.dtype and cat.shape == blk.shape
        assert cat.tobytes() == blk.tobytes(), f"g={g}: band concat != gen_block"

        d_blk, d_str = os.path.join(work, f"b{g}"), os.path.join(work, f"s{g}")
        mb = ShardedIndex.write_shard(
            d_blk,
            blk,
            1,
            ids=np.arange(g * ROWS, (g + 1) * ROWS, dtype=np.int64),
            basis_from=basis,
            keep_originals=False,
        )
        ms = ShardedIndex.write_shard_streaming(
            d_str,
            gen_block_bands(g, rows=ROWS),
            1,
            ids_start=g * ROWS,
            basis_from=basis,
        )
        assert mb == ms, f"g={g}: metas differ: {mb} vs {ms}"
        pa, pb = (os.path.join(d, "shard_00001.tqe") for d in (d_blk, d_str))
        _, sa = read_container(pa)
        _, sb = read_container(pb)
        assert sa.keys() == sb.keys(), f"g={g}: section sets differ"
        ja, jb = (json.loads(x.pop("meta")) for x in (sa, sb))
        ja.pop("created_utc"), jb.pop("created_utc")
        assert ja == jb, f"g={g}: meta differs: {ja} vs {jb}"
        for name in (
            "pca_mean",
            "pca_components",
            "pca_eigenvalues",
            "pca_all_eigenvalues",
        ):
            assert sa[name] == sb[name], f"g={g}: section {name!r} differs"

        ca = TQEIndex.open(pa)._adc._codes.astype(np.int16)
        cb = TQEIndex.open(pb)._adc._codes.astype(np.int16)
        d = ca - cb
        n_flip = int((d != 0).sum())
        flip_rows = (d != 0).any(axis=1)
        assert np.abs(d).max(initial=0) <= 1, f"g={g}: code flip beyond +-1 level"
        assert n_flip / d.size <= 5e-6, f"g={g}: {n_flip}/{d.size} code flips"
        print(f"g={g}: codes flipped={n_flip}/{d.size} (all +-1)", flush=True)

        # A flipped row's vrnorm is the norm of ITS OWN reconstruction — each
        # file is self-consistent — so those rows are excluded from the ULP
        # gate rather than pretending they should match.
        for name in ("cnorm", "vrnorm"):
            va = np.frombuffer(sa[name], dtype=np.float32)
            vb = np.frombuffer(sb[name], dtype=np.float32)
            assert va.shape == vb.shape, f"g={g}: {name} lengths differ"
            ulp = np.abs(
                va.view(np.int32).astype(np.int64) - vb.view(np.int32).astype(np.int64)
            )[~flip_rows]
            frac = float((ulp > 0).mean())
            print(
                f"g={g}: {name} (unflipped rows) max_ulp={int(ulp.max())} "
                f"frac_diff={frac:.2e}",
                flush=True,
            )
            assert ulp.max() <= 50, f"g={g}: {name} differs beyond 50 ULP"
        for d_ in (d_blk, d_str):
            shutil.rmtree(d_)
        print(f"g={g}: OK (corpus byte-equal; codes/norms within gates)", flush=True)
    print("VERIFY_STREAM_OK")
finally:
    shutil.rmtree(work, ignore_errors=True)
    sys.stdout.flush()
