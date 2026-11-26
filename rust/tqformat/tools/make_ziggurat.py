"""Copy NumPy's three double-precision normal ziggurat tables into Rust or Python.

    python tools/make_ziggurat.py ziggurat_constants.h src/ziggurat.rs
    python tools/make_ziggurat.py ziggurat_constants.h _ziggurat_tables.py

The output language follows the file's extension. Every constant stays the literal the
C compiler saw: all three languages round a decimal literal to the nearest double, so
the tables are bit-identical to NumPy's.
"""

import re
import sys

src = open(sys.argv[1], encoding="utf-8").read()
dst = sys.argv[2]

NUMPY_LICENSE = """Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""


def table(name):
    m = re.search(r"static const (\w+) " + name + r"\[\] = \{(.*?)\};", src, re.S)
    ctype, body = m.group(1), m.group(2)
    vals = [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]
    assert len(vals) == 256, (name, len(vals))
    return ctype, vals


def literal(ctype, v, upper_hex=False):
    if ctype == "uint64_t":
        v = re.sub(r"(ULL|UL|U)$", "", v, flags=re.I)
        if v.lower().startswith("0x"):
            # rustfmt leaves hex alone (lowercase here); black writes 0x + uppercase
            return "0x" + (v[2:].upper() if upper_hex else v[2:].lower())
        return v
    # double: keep the decimal text exactly; a trailing dot needs a digit after it
    if v.endswith("."):
        v += "0"
    if upper_hex:
        # black's normal form for Python, so the generated file is already formatted:
        # lowercase e, no '+' in the exponent (the value does not change)
        v = v.replace("E", "e").replace("e+", "e")
    return v


TABLES = (("ki_double", "u64"), ("wi_double", "f64"), ("fi_double", "f64"))
r = re.search(r"static const double ziggurat_nor_r = ([0-9.eE+-]+);", src).group(1)
inv = re.search(
    r"static const double ziggurat_nor_inv_r =\s*([0-9.eE+-]+);", src
).group(1)

if dst.endswith(".rs"):
    out = [
        "// The double-precision normal ziggurat tables of NumPy, copied textually"
        " from",
        "// numpy/random/src/distributions/ziggurat_constants.h (NumPy v2.4.6) by",
        "// rust/tqformat/tools/make_ziggurat.py. Do not edit by hand.",
        "//",
        "// Copyright (c) 2005-2025, NumPy Developers. All rights reserved.",
        "// Licensed under the BSD 3-Clause License; see NOTICE in this crate.",
        "",
        "#![allow(clippy::unreadable_literal, clippy::excessive_precision)]",
        "",
    ]
    for name, rtype in TABLES:
        ctype, vals = table(name)
        out.append(f"pub(crate) static {name.upper()}: [{rtype}; 256] = [")
        out += [f"    {literal(ctype, v)}," for v in vals]
        out += ["];", ""]
    out.append(f"pub(crate) const ZIGGURAT_NOR_R: f64 = {r};")
    out.append(f"pub(crate) const ZIGGURAT_NOR_INV_R: f64 = {inv};")
elif dst.endswith(".py"):
    out = [
        '"""NumPy\'s double-precision normal ziggurat tables, for frozen_rng.',
        "",
        "Copied textually from numpy/random/src/distributions/ziggurat_constants.h",
        "(NumPy v2.4.6) by rust/tqformat/tools/make_ziggurat.py. Do not edit by hand.",
        "",
        "These tables are part of the TQE1 decoding contract: the rotation of a record",
        "is built from normals drawn with them (docs/FORMAT_SPEC.md, Rotation",
        "derivation). They are NumPy's, under NumPy's licence:",
        "",
        *NUMPY_LICENSE.splitlines(),
        '"""',
        "",
    ]
    for name, _ in TABLES:
        ctype, vals = table(name)
        out.append(f"{name.upper()} = (")
        out += [f"    {literal(ctype, v, upper_hex=True)}," for v in vals]
        out += [")", ""]
    out.append(f"ZIGGURAT_NOR_R = {r}")
    out.append(f"ZIGGURAT_NOR_INV_R = {inv}")
else:
    raise SystemExit("output must end in .rs or .py")
open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("wrote", dst)
