"""Write src/kat.rs: known answers from NumPy for the random-stream port in src/rng.rs.

    python tools/make_kat.py src/kat.rs

Run it with the NumPy whose streams the port must reproduce; the file records the
version. The values are the draws TQE1 decoding depends on: SeedSequence mixing,
PCG64 output, the ziggurat normal (2,000,000 draws, so the tail and wedge branches are
exercised), choice of signs, and permutation, including the 32-bit buffer that one
leaves for the other.
"""

import sys

import numpy as np
from numpy.random import PCG64, SeedSequence


def arr(name, rtype, values):
    vals = ", ".join(values)
    return f"pub(crate) const {name}: [{rtype}; {len(values)}] = [{vals}];"


lines = [
    f"// Known answers from NumPy {np.__version__}, written by tools/make_kat.py."
    " Do not edit by hand.",
    "#![allow(clippy::unreadable_literal)]",
    "",
]
lines.append(arr("SEEDSEQ_0_POOL", "u32", [str(int(v)) for v in SeedSequence(0).pool]))
lines.append(
    arr(
        "SEEDSEQ_42_STATE_U64",
        "u64",
        [str(int(v)) for v in SeedSequence(42).generate_state(4, np.uint64)],
    )
)
lines.append(arr("PCG64_42_RAW", "u64", [str(int(v)) for v in PCG64(42).random_raw(3)]))
z = np.random.default_rng(42).standard_normal(6)
lines.append(arr("NORMAL_42_BITS", "u64", [str(int(v)) for v in z.view(np.uint64)]))
c = np.random.default_rng(7).choice([-1.0, 1.0], size=9)
lines.append(arr("CHOICE_7_9", "f64", [repr(float(v)) for v in c]))
p = np.random.default_rng(7).permutation(10)
lines.append(arr("PERM_7_10", "usize", [str(int(v)) for v in p]))
g = np.random.default_rng(11)
s = g.choice([-1.0, 1.0], size=5)
p2 = g.permutation(12)
lines.append(arr("CHOICE_11_5", "f64", [repr(float(v)) for v in s]))
lines.append(arr("PERM_11_12_AFTER", "usize", [str(int(v)) for v in p2]))
big = np.random.default_rng(2026).standard_normal(2_000_000)
xor = int(np.bitwise_xor.reduce(big.view(np.uint64)))
total = 0.0
for v in big.tolist():  # sequential double addition, as the Rust test sums
    total += v
total_bits = int(np.array([total], dtype=np.float64).view(np.uint64)[0])
lines.append(
    f"pub(crate) const NORMAL_2026_2M_XOR_SUM: (u64, u64) = ({xor}, {total_bits});"
)
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("wrote", sys.argv[1], "with NumPy", np.__version__)
