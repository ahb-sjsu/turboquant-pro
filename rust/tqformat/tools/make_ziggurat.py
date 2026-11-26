"""Copy NumPy's three double-precision normal ziggurat tables into Rust, textually.

    python tools/make_ziggurat.py ziggurat_constants.h src/ziggurat.rs

Every constant stays the literal the C compiler saw: both languages round a decimal
literal to the nearest double, so the tables are bit-identical.
"""

import re
import sys

src = open(sys.argv[1], encoding="utf-8").read()


def table(name):
    m = re.search(r"static const (\w+) " + name + r"\[\] = \{(.*?)\};", src, re.S)
    ctype, body = m.group(1), m.group(2)
    vals = [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]
    assert len(vals) == 256, (name, len(vals))
    return ctype, vals


def rust_literal(ctype, v):
    if ctype == "uint64_t":
        v = re.sub(r"(ULL|UL|U)$", "", v, flags=re.I)
        return v.lower() if v.lower().startswith("0x") else v
    # double: keep the decimal text exactly; Rust needs a digit after a trailing dot
    if v.endswith("."):
        v += "0"
    return v


out = [
    "// The double-precision normal ziggurat tables of NumPy, copied textually from",
    "// numpy/random/src/distributions/ziggurat_constants.h (NumPy v2.4.6) by",
    "// rust/tqformat/tools/make_ziggurat.py. Do not edit by hand.",
    "//",
    "// Copyright (c) 2005-2025, NumPy Developers. All rights reserved.",
    "// Licensed under the BSD 3-Clause License; see NOTICE in this crate.",
    "",
    "#![allow(clippy::unreadable_literal, clippy::excessive_precision)]",
    "",
]
for name, rtype in (("ki_double", "u64"), ("wi_double", "f64"), ("fi_double", "f64")):
    ctype, vals = table(name)
    out.append(f"pub(crate) static {name.upper()}: [{rtype}; 256] = [")
    for v in vals:
        out.append(f"    {rust_literal(ctype, v)},")
    out.append("];")
    out.append("")
r = re.search(r"static const double ziggurat_nor_r = ([0-9.eE+-]+);", src).group(1)
inv = re.search(
    r"static const double ziggurat_nor_inv_r =\s*([0-9.eE+-]+);", src
).group(1)
out.append(f"pub(crate) const ZIGGURAT_NOR_R: f64 = {r};")
out.append(f"pub(crate) const ZIGGURAT_NOR_INV_R: f64 = {inv};")
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote", sys.argv[2])
