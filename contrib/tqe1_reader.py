#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""tqe1_reader — a single-file, dependency-free reference reader for TQE1.

This file is the "safetensors-style" portability proof for the TQE1 record
format (``docs/FORMAT_SPEC.md``): it decodes any TQE1 v1/v2/v3 record or batch
file using **only the Python standard library and numpy** — no turboquant-pro
import. Copy this one file into your project (or vendor it) and you can read
TQE1 forever: the rotation's random draws come from a frozen generator in this
file (a verbatim copy of ``turboquant_pro/frozen_rng.py``'s), not from
``numpy.random``'s Generator, whose streams numpy may change between releases
(numpy's PCG64 bit generator only speeds up the raw stream, after a check that it
agrees with the frozen one). It is
validated against the in-tree implementation by the golden-corpus conformance
suite (``tests/golden/tqe1/`` + ``tests/test_tqe1_golden.py``), and the copy of
the generator by ``tests/test_frozen_rng.py``.

Every constant below is normative and mirrors a section of FORMAT_SPEC.md;
section references are given inline. The decode contract:

    reconstruction = norm * (codebook(bits, cb)[indices] @ R(seed, rotation))

with indices bit-unpacked LSB-first and R the seed-deterministic orthogonal
rotation. Records are self-describing: no out-of-band metadata is required.

CLI usage::

    python tqe1_reader.py file.tqe            # summary of each record
    python tqe1_reader.py file.tqe --out x.npy  # decode all records -> (n, dim)

Library usage::

    from tqe1_reader import read_records, decode
    for rec in read_records(open("file.tqe", "rb").read()):
        vec = decode(rec)
"""

from __future__ import annotations

import argparse
import math
import operator
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"TQE1"
_HEADER_V1 = struct.Struct("<4sBBHIfI")  # SPEC "Record layout", v1 (20 bytes)
_HEADER_V2 = struct.Struct("<4sBBHIfBI")  # SPEC "Record layout", v2 (21 bytes)
_HEADER_V3 = struct.Struct("<4sBBHIfBBI")  # SPEC "Record layout", v3 (22 bytes)
_ROTATIONS = {0: "qr", 1: "hadamard"}  # SPEC v2/v3 `rotation` byte
_CODEBOOK_IDS = {0: "legacy", 1: "lloyd-max"}  # SPEC v3 `codebook` byte
_VALID_BITS = (2, 3, 4)

# SPEC "Codebooks": fixed centroids for a unit-Gaussian coordinate, scaled by
# 1/sqrt(dim) at decode time. "legacy" (v1, v2, and v3 with codebook 0) is the
# historical table; it is NOT the Lloyd-Max quantizer at 3 and 4 bits.
_LEGACY = {
    2: np.array([-1.510, -0.453, 0.453, 1.510]),
    3: np.array([-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]),
    4: np.array(
        [
            -2.401,
            -1.844,
            -1.437,
            -1.099,
            -0.800,
            -0.524,
            -0.262,
            -0.066,
            0.066,
            0.262,
            0.524,
            0.800,
            1.099,
            1.437,
            1.844,
            2.401,
        ]
    ),
}


def _sym(pos):
    pos = np.asarray(pos, dtype=np.float64)
    return np.concatenate([-pos[::-1], pos])


# "lloyd-max" (v3 with codebook 1): the Lloyd-Max quantizer for N(0, 1).
_LLOYD_MAX = {
    2: _sym([0.452780, 1.510418]),
    3: _sym([0.245094, 0.756005, 1.343909, 2.151946]),
    4: _sym(
        [0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590]
    ),
}
_CODEBOOKS = {"legacy": _LEGACY, "lloyd-max": _LLOYD_MAX}


# ------------------------------------- rotation draws (SPEC "Rotation derivation")
# The rotation is drawn from a frozen copy of the generator that
# np.random.default_rng(seed) is, not from numpy.random, so this reader does not
# depend on numpy's random streams. NumPy's double-precision normal ziggurat
# tables (ki, wi, fi: 256 entries each) are below as little-endian 64-bit words,
# from numpy/random/src/distributions/ziggurat_constants.h (NumPy v2.4.6); they
# and the algorithms ported from NumPy are under NumPy's licence:
#
# Copyright (c) 2005-2025, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
_ZIGGURAT_HEX = (
    "6aef25803df30e000000000000000000a8c6fb98be080c004281bdfa54a30d00"
    "eaeec17ef6510e007ef7d3e955b20e00b9ca7e814bef0e00aa44fa0a47190f00"
    "18cbff61ed370f005c256195464f0f0096a31be4a5610f00a49653757a700f00"
    "9a4428ecb27c0f00d357630cf1860f00de258357a68f0f00dad04dc724970f00"
    "09f5db07a99d0f0074fa81f560a30f00f84b5bde6fa80f00dc54d360f1ac0f00"
    "0fb91867fbb00f00c674538d9fb40f0077fe6623ecb70f000ee5a1e9ecba0f00"
    "ed0b049dabbd0f00576cff6030c00f0048a2371082c20f00d15be27aa6c40f00"
    "31ee7a97a2c60f00a49628a97ac80f0085de4b5e32ca0f001a2302e9cccb0f00"
    "c439f8124dcd0f0099ec8f4db5ce0f0030c91dbf07d00f00e6c4d64d46d10f00"
    "50f4e2a872d20f001ec9f04f8ed30f0078b490999ad40f00530f92b898d50f00"
    "ec998ec089d60f0032e8c8a96ed70f00e8087b5448d80f008c2cad8b17d90f00"
    "d2ada707ddd90f008c5e107099da0f00202ec05d4ddb0f00d0fc5b5cf9db0f00"
    "7d9ab9eb9ddc0f009d7218813bdd0f00902f3488d2dd0f00649f366463de0f00"
    "4e518d70eede0f002eb4a60174df0f0040ed9965f4df0f00f224bce46fe00f00"
    "58a225c2e6e00f004cb8283c59e10f00993fbc8cc7e10f00aa1cdbe931e20f00"
    "911bda8598e20f008641b58ffbe20f004a8d55335be30f002a00d099b7e30f00"
    "7fad9ee910e40f003477d44667e40f005c094cd3bae40f002495d2ae0be50f00"
    "78bc4ef759e50f001212e4c8a5e50f008986133eefe50f007810d96f36e60f00"
    "78d5c6757be60f00aa111e66bee60f00f2f4e555ffe60f0002a700593ee70f00"
    "399e3e827be70f00a27070e3b6e70f004342778df0e70f008cf0539028e80f00"
    "3a1735fb5ee80f00640884dc93e80f00bccef041c7e80f00f64e7d38f9e80f00"
    "1d9b87cc29e90f00ea88d30959e90f00a29a93fb86e90f00664871acb3e90f00"
    "d5b69426dfe90f007ce6ab7309ea0f00a466f19c32ea0f002c9532ab5aea0f00"
    "1a74d5a681ea0f00f01cde97a7ea0f0020d9f385ccea0f003ce66578f0ea0f00"
    "13ec2f7613eb0f004a2afe8535eb0f00b46231ae56eb0f00fa84e2f476eb0f00"
    "1420e65f96eb0f007c9dcff4b4eb0f00d049f4b8d2eb0f003e2e6eb1efeb0f00"
    "e8bd1ee30bec0f00155ab15227ec0f00d3af9d0442ec0f0096f129fd5bec0f00"
    "f4ee6c4075ec0f00b40c50d28dec0f00121f91b6a5ec0f00fe27c4f0bcec0f00"
    "15fb5484d3ec0f00b3c88874e9ec0f00b7917fc4feec0f002885357713ed0f00"
    "0349848f27ed0f004c2f24103bed0f006e58adfb4ded0f00ddc3985460ed0f00"
    "e84f411d72ed0f0082a9e45783ed0f00c82ca40694ed0f0004b7852ba4ed0f00"
    "b46a74c8b3ed0f00526641dfc2ed0f00526ea471d1ed0f00d38a3c81dfed0f00"
    "8099900feded0f0014d40f1efaed0f00c44b12ae06ee0f00065ad9c012ee0f00"
    "e00690571eee0f0024654b7329ee0f00bce40a1534ee0f003c9bb83d3eee0f00"
    "f48229ee47ee0f0086b01d2751ee0f00417f40e959ee0f002eb4283562ee0f00"
    "f197580b6aee0f007a073e6c71ee0f00827b325878ee0f00ba067bcf7eee0f00"
    "b24a48d284ee0f004363b6608aee0f0051c8cc7a8fee0f00da257e2094ee0f00"
    "ea29a85198ee0f005c48130e9cee0f00f47372559fee0f00aecc6227a2ee0f00"
    "ac426b83a4ee0f00712dfc68a6ee0f00fad66ed7a7ee0f000afa04cea8ee0f00"
    "3b33e84ba9ee0f0010642950a9ee0f005e07c0d9a8ee0f00547689e7a7ee0f00"
    "241d4878a6ee0f00839ea28aa4ee0f00dae4221da2ee0f002420352e9fee0f00"
    "2eaf26bc9bee0f00e4f224c597ee0f003a0a3c4793ee0f00167555408eee0f00"
    "7a9c36ae88ee0f00fd3d7f8e82ee0f0088b8a7de7bee0f00ff37ff9b74ee0f00"
    "5ebda9c36cee0f007e009e5264ee0f008828a3455bee0f00b6574e9951ee0f00"
    "cf06004a47ee0f00502ce1533cee0f00d82ae0b230ee0f000582ad6224ee0f00"
    "5a3cb85e17ee0f0047142aa209ee0f00cc49e327fbed0f006c2176eaebed0f00"
    "7e0422e4dbed0f00d339ce0ecbed0f00f42c0464b9ed0f00c938e9dca6ed0f00"
    "8de9377293ed0f0036a8381c7fed0f002bc0b9d269ed0f0000ae068d53ed0f00"
    "22a4de413ced0f00d82f6ae723ed0f0044e62f730aed0f0034fe07daefec0f00"
    "b8b70e10d4ec0f00b46e9508b7ec0f00c13012b698ec0f0078a90d0a79ec0f00"
    "fe310ff557ec0f0062c9866635ec0f0035b3b44c11ec0f00d06f8e94ebeb0f00"
    "92b6a029c4eb0f00dc0ceef59aeb0f004285c9e16feb0f009e1fadd342eb0f00"
    "4b2d0bb013eb0f00e9021a59e2ea0f00572299aeaeea0f0026e38e8d78ea0f00"
    "e573fdcf3fea0f00f6d98d4c04ea0f003b562fd6c5e90f00a447a93b84e90f00"
    "28471d473fe90f00d6c576bdf6e80f00e6e8c45daae80f00eab17ae059e80f00"
    "40a990f604e80f00c0338248abe70f00a56a1f754ce70f0002a22a10e8e60f00"
    "d8abb6a07de60f007e30389f0ce60f0042f7387394e50f008072977014e50f00"
    "58f436d48be40f00371efdbff9e30f009cb1ee355de30f00fee42f12b5e20f00"
    "5755990300e20f00148378823ce10f00b067eec468e00f00aa712bb082df0f00"
    "aafe7ec587de0f00fd3bc60975dd0f0013bf29e546dc0f0082022ef8f8da0f00"
    "75bab2e185d90f0004cf48efe6d70f000b65bdad13d60f0012f0e24901d40f00"
    "acc7b4a7a1d10f009e1f7604e2ce0f00b2115ed8a8cb0f00222dcd6ed2c70f00"
    "ed221e2f2bc30f003ab8c08165bd0f00345400c406b60f0074282a5840ac0f00"
    "9845011e979e0f00fc1da448fa890f002c30f0f7c5660f004a1c334b5a1a0f00"
    "79d915783b49cf3cc6f6fde30b8d8b3cb45b2c3caf50923c613b4438b97c953c"
    "0ca72fe8fc01983cbcd04c2e0c239a3cf761382f4d009c3c7472745a2fac9d3c"
    "c3d54c2d48329f3cadbb8e27324da03c435d023b05f5a03c77364197a692a13c"
    "f51a7a8fa227a23c80d863382eb5a23cf59157c03f3ca33c2fb1a2c19ebda33c"
    "559bff8def39a43ca7fe3d36bbb1a43c74d31a627525a53c96ce07a78095a53c"
    "ea7ed9cf3102a63c3d7ca361d26ba63c70050092a2d2a63ca6f846d3da36a73c"
    "772ab310ad98a73c43f546ad45f8a73c770a4353cc55a83c9a767b9e64b1a83c"
    "98cf4ea92e0ba93cea1e2c824763a93c46c5388ec9b9a93c2ca7a4dccc0eaa3c"
    "59cd776d6762aa3c3016106eadb4aa3c9c6c136db105ab3c297a42878455ab3c"
    "3a9f528e36a4ab3c3282bf2ad6f1ab3cf34e59f9703eac3c613b32a5138aac3c"
    "8b2672fec9d4ac3c48b7800e9f1ead3c101fe4299d67ad3cc3b82300ceafad3c"
    "5376f1a93af7ad3cfeedd2b5eb3dae3c006f7a33e983ae3cce82f9bd3ac9ae3c"
    "2662f084e70daf3c88f6d854f651af3caed7879e6d95af3cac2efa7d53d8af3c"
    "ec3442e0560db03c9a8f39f5402eb03cfca5169eea4eb03c10a0725b566fb03c"
    "0bf47190868fb03c1361bc847dafb03c7fcc4b663dcfb03c6b08164bc8eeb03c"
    "ee159532200eb13cbe0f3107472db13c41918e9f3e4cb13c1e20c4bf086bb13c"
    "34da781aa789b13c886dee511ba8b13ccb2af8f866c6b13c2ed4e0938be4b13c"
    "9fa040998a02b23ce9c6c4726520b23c1fc3e97d1d3eb23cfb6ba90cb45bb23c"
    "7fd31d662a79b23c1bd719c78196b23cda2eb862bbb3b23c53b8e162d8d0b23c"
    "8ea9cbe8d9edb23cd7486e0dc10ab33c30b9f4e18e27b33ca15e26704444b33c"
    "d552cabae260b33c6a5805be6a7db33c64b2b26fdd99b33c033db8bf3bb6b33c"
    "e01d569886d2b33c835a72debeeeb33c749ee071e50ab43c5d74a62dfb26b43c"
    "a4303ce80043b43c5dc7ca73f75eb43c36c3669edf7ab43c2f8f4832ba96b43c"
    "5d4102f687b2b43cdc11b3ac49ceb43c05a6381600eab43c62555eefab05b53c"
    "5a8b0af24d21b53c4f666ad5e63cb53cc8b21b4e7758b53c785f550e0074b53c"
    "14850ec6818fb53c591b2423fdaab53c3d737dd172c6b53cd38c2f7be3e1b53c"
    "385e9fc84ffdb53cc31fa360b818b63ca2b0a2e81d34b63c0b26b704814fb63c"
    "7296c957e26ab63c3731b1834286b63cb1b25029a2a1b63cbb43b3e801bdb63c"
    "52d3286162d8b63c54f86131c4f3b63ceb688bf7270fb73cc61469518e2ab73c"
    "dcee70dcf745b73c1f73e5356561b73c49f4effad67cb73c93bdbac84d98b73c"
    "09148b3ccab3b73cfb22dbf34ccfb73ce7de738cd6eab73c1fea86a46706b83c"
    "7686c8da0022b83c159f89cea23db83cbdf5d11f4e59b83cc57e7a6f0375b83c"
    "2df7475fc390b83c43c005928eacb83c9c0ca1ab65c8b83c276a445149e4b83c"
    "8fb573293a00b93c478328dc381cb93cfc0aef124638b93c8aa203796254b93c"
    "eed570bb8e70b93c312a2e89cb8cb93cbf993f9319a9b93c2cd9d58c79c5b93c"
    "11746f2bece1b93c4ad2fa2672feb93c9236f9390c1bba3c5bc8a221bb37ba3c"
    "88bb0b9e7f54ba3ca4a94a725a71ba3c3d31a0644c8eba3c08f19f3e56abba3c"
    "cef55acd78c8ba3c36b38be1b4e5ba3c1aa1c34f0b03bb3c5b989af07c20bb3c"
    "000ce0a00a3ebb3c033dce41b55bbb3c27893fb97d79bb3c3cf7e5f16497bb3c"
    "6e2585db6bb5bb3ca2c02e6b93d3bb3c83ae819bdcf1bb3ca016ec6c4810bc3c"
    "2d7af0e5d72ebc3c1c0d6e138c4dbc3c0587ec08666cbc3c17a6ebe0668bbc3c"
    "aba236bd8faabc3c90d63bc7e1c9bc3c37e068305ee9bc3c6e8f8b320609bd3c"
    "20ef3710db28bd3c47c63315de48bd3c23f1e7961069bd3ca5fbd7f47389bd3c"
    "706e209909aabd3c0e49fcf8d2cabd3c372e5295d1ebbd3c1cd249fb060dbe3c"
    "f646eac4742ebe3c88d1c1991c50be3c25fe972f0072be3c0abf2a4b2194be3c"
    "086ff7c081b6be3c3aa7107623d9be3ca9ec016108fcbe3c2153c28a321fbf3c"
    "6d4db70fa442bf3c6801c9205f66bf3c82978904668abf3cbf227118bbaebf3c"
    "85e72fd260d3bf3c0bf618c159f8bf3c75a0d347d40ec03c47c98f02a821c03c"
    "ab02a983a934c03cc7f53e4eda47c03c7eb3adf63b5bc03c6826a723d06ec03c"
    "172e638f9882c03c54a2e8089796c03cc4c07175cdaac03c48d4eed13dbfc03c"
    "303daa34ead3c03c936511cfd4e8c03cb69fa6effffdc03c417020046e13c13c"
    "355dbb9b2129c13c6d09c4691d3fc13c3b2e60486455c13cf3ee9d3bf96bc13c"
    "6112d274df82c13caceb4e561a9ac13c8e2f7f77adb1c13c94a671a99cc9c13c"
    "39aee4fbebe1c13c01d9e2c29ffac13c81cc049dbc13c23ceed36f7a472dc23c"
    "249caca44547c23ce05876c7bc61c23c2e59a8fab27cc23c780e77cd2e98c23c"
    "520a2a5337b4c23c97db9631d4d0c23cf578a9b10deec23ceeae56d2ec0bc33c"
    "a3a4685e7b2ac33ca312ae05c449c33c40a8337ad269c33c0a415692b38ac33c"
    "fa88ae7075acc33ca60417b327cfc33c75f460aadbf2c33cdae5b99ca417c43c"
    "945e5415983dc43c153aa744ce64c43cbc439c75628dc43c275a6b9d73b7c43c"
    "0289cd0d25e3c43c41ace9539f10c53c427e3a521140c53c1be44aa9b171c53c"
    "d98d718bc0a5c53cfed03a248adcc53c4c1e86cf6916c63cea6a007bce53c63c"
    "c3e59fbe4095c63c32e2098d6bdbc63c347a5ff02827c73c730609569579c73c"
    "8cced6f42dd4c73c34f229050339c83c147caabf0fabc83c96446f94e02ec93c"
    "ab574001eecbc93c5a779478dc8fca3cb1fd78381f98cb3c33ad0982b43bcd3c"
    "000000000000f03f87f079c96a44ef3f15a96c5b54b7ee3f77f027e0113fee3f"
    "95de04a76fd3ed3ff2bc57069270ed3fdc19a1784914ed3feb2da7a833bdec3f"
    "7f78a9ce5e6aec3feabaeed91c1bec3f82dce14eebceeb3f52f58f3a6585eb3f"
    "10dd34823a3eeb3fa2e86c3f2af9ea3f04257af1feb5ea3fe1c950d58b74ea3f"
    "0faff5fdaa34ea3fd81f65ee3bf6e93f8106248d22b9e93fc17a6157467de93f"
    "477a1bc29142e93f4f7131bdf108e93fa80ae64f55d0e83f02dfba48ad98e83f"
    "acbc37fceb61e83f6ecf560f052ce83fcbe2204bedf6e73f58689c779ac2e73f"
    "d5b0a03c038fe73f56d870071f5ce73f126d3ff4e529e73fee7aeaba50f8e63f"
    "895a639e58c7e63f2a3b515ef796e63f23e3922a2767e63f180c5598e237e63f"
    "652680982409e63f6aff4a6fe8dae53f895cc8ac29ade53f8f8d4c26e47fe53f"
    "469e8df01353e53fd56c655ab526e53f67b620e8c4fae43fc04e494f3fcfe43f"
    "7852dc7221a4e43f1250df5f6879e43f7936494a114fe43fe35f358a1925e43f"
    "825b58997efbe33fa331af103ed2e33f0ecd62a655a9e33fd500da2bc380e33f"
    "e950f58b8458e33f353a70c99730e33fef3864fdfa08e33fee3bea55ace1e23f"
    "4a95d714aabae23f15cd938ef293e23fed040529846de23f84db905a5d47e23f"
    "f2f72fa97c21e23f209692a9e0fbe13f699954fe87d6e13f11d13f5771b1e13f"
    "503c9b709b8ce13fda3986120568e13f9ca95e10ad43e13f381f3148921fe13f"
    "135932a2b3fbe03fa042411010d8e03faed9708da6b4e03f815d991d7691e03f"
    "363cf0cc7d6ee03f2e3fa6afbc4be03f2a828be13129e03fc4cab885dc06e03f"
    "a1bd7b8c77c9df3fca00a9a79d85df3ff37a2fcb2942df3f958f7e711affde3f"
    "541fbd206ebcde3fc5c34e6a237ade3f859b5fea3838de3f093a7647adf6dd3f"
    "b1560b327fb5dd3f33de2664ad74dd3f801002a13634dd3f6d5baeb419f4dc3f"
    "48a8c07355b4dc3fc7d700bbe874dc3fb82c1d6fd235dc3f176a617c11f7db3f"
    "916d71d6a4b8db3f1b1307788b7adb3fca31b362c43cdb3f5285a19e4effda3f"
    "9e5a5f3a29c2da3f80d8a44a5385da3f4dc020eacb48da3f3e844639920cda3f"
    "df931e5ea5d0d93fc6c018840495d93f939fe0dbae59d93f17cb339ba31ed93f"
    "15f1b9fce1e3d83f8891de3f69a9d83fb65aaca8386fd83fd90daa7f4f35d83f"
    "11d9b811adfbd73fb014f4af50c2d73feb5292af3989d73fedb1c7696750d73f"
    "4c61a93bd917d73faa4c12868edfd63f21de88ad86a7d63fe2cb251ac16fd63f"
    "15e57b373d38d63fc8d28074fa00d63f44c27643f8c9d53fbeeed6193693d53f"
    "00013d70b35cd53fed3b53c26f26d53f926dbf8e6af0d43fa29c1057a3bad43f"
    "d46aad9f1985d43ffe24c3efcc4fd43f197a35d1bc1ad43fdbd28ed0e8e5d33f"
    "ae43f17c50b1d33f79130868f37cd33f9ed1f925d148d33f2ff65a4de914d33f"
    "660721773be1d23fdd3f963ec7add23f1eb14d418c7ad23f89de171f8a47d23f"
    "9eccf779c014d23f168118f62ee2d13f50f0c239d5afd13fe85454edb27dd13f"
    "67ee34bbc74bd13f2324cf4f131ad13fc409875995e8d03fda42b2884db7d03f"
    "3643908f3b86d03fd9e942225f55d03f7e74c7f6b724d03fc593df898be8cf3f"
    "3532b88c1088cf3fd298e96cfe27cf3f449cc9a454c8ce3fdd3c28b21269ce3f"
    "84714516380ace3f0a90c755c4abcd3f4f51b2f8b64dcd3fcc6f5e8a0ff0cc3f"
    "53df7199cd92cc3f479dd8b7f035cc3fa118be7a78d9cb3faa31877a647dcb3f"
    "3ad1cc52b421cb3f071857a267c6ca3f7e26190b7e6bca3f3d7e2d32f710ca3f"
    "5afed2bfd2b6c93f277c6a5f105dc93f69fa74bfaf03c93f5b819291b0aac83f"
    "389a818a1252c83f75711f62d5f9c73f23a368d3f8a1c73fa6b57a9c7c4ac73f"
    "1647967e60f3c63f5cf2213ea49cc63f9cf1ada24746c63ff983f8764af0c53f"
    "6c1df388ac9ac53f3568c8a96d45c53fc11fe3ad8df0c43f2dcef56c0c9cc43f"
    "d57503c2e947c43fae31698b25f4c33feed7e8aabfa0c33f88abb405b84dc33f"
    "652a7c840efbc23f1a077a13c3a8c23fb75e83a2d556c23f343c18254605c23f"
    "427d759214b4c13f632da8e54063c13fb96ea21dcb12c13fba09523db3c2c03f"
    "85bfb84bf972c03f2a7d06549d23c03f2c226bcb3ea9bf3f1c0e5229ff0bbf3f"
    "4ba59af27b6fbe3f8fe87661b5d3bd3fe591bdb9ab38bd3f0a743b495f9ebc3f"
    "15100b68d004bc3f33e2f278ff6bbb3f33f6cae9ecd3ba3f8662ea33993cba3f"
    "195b9ddc04a6b93faba0a4753010b93f5228bf9d1c7bb83fd6ef3e01cae6b73f"
    "7611aa5a3953b73f4c4a69736bc0b63f184d8524612eb63fa46674571b9db53f"
    "ae2bfa069b0cb53f13221b40e17cb43f869a2623efedb33f703ed9e4c55fb33f"
    "11319bcf66d2b23f910ddd44d345b23f7d8997be0cbab13f9d17f2d0142fb13f"
    "2596152ceda4b03f97e4309e971bb03f356e6c2b2c26af3f8151b247d516ae3f"
    "62f1adfe2e09ad3f2c2a280f3efdab3f705f389007f3aa3f635529f990eaa93f"
    "abb5682ae0e3a83f1e27af77fbdea73f64d098b3e9dba63fd4adf23cb2daa53f"
    "5d27110e5ddba43fcbee98cef2dda33f97f43de87ce2a23fbc6a1f9f05e9a13f"
    "1180962e98f1a03fc4a518d781f89f3f758c82db1a129e3f1a09cd8319309c3f"
    "f8eb224e9f529a3f0ac100b6d179983f82bf0bf4daa5963f64b0fbf2ead6943f"
    "135eab8d380d933f123060340349913f49dd724f2a158f3fac8f4f278da48b3f"
    "78a48d0d0441883fe0cf1a4296eb843f922f952992a5813f3768ecf860e17c3f"
    "5db80cd9a89e763ffdb1b0031f8a703f67b0c1439f5f653f0ff7b9b605a6543f"
)
_ZW = np.frombuffer(bytes.fromhex("".join(_ZIGGURAT_HEX)), dtype="<u8")
_KI = _ZW[:256].astype(np.uint64)
_WI = _ZW[256:512].view("<f8").astype(np.float64)
_FI = _ZW[512:].view("<f8").astype(np.float64)
ZIGGURAT_NOR_R = 3.654152885361009
ZIGGURAT_NOR_INV_R = 0.2736612373297583


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


@dataclass
class TQERecord:
    """One parsed (not yet decoded) TQE1 record."""

    version: int
    bits: int
    dim: int
    seed: int
    norm: float
    rotation: str
    codes: bytes
    codebook: str = "legacy"


# ----------------------------------------------------------------- parsing
def parse_record(buf: bytes, offset: int = 0) -> tuple[TQERecord, int]:
    """Parse one record at ``offset``; returns ``(record, next_offset)``.

    Raises ``ValueError`` on bad magic, unknown version/rotation/bits, or
    truncation — the SPEC "Conformance" error cases.
    """
    if len(buf) < offset + 20:
        raise ValueError("buffer too small for a TQE header")
    if buf[offset : offset + 4] != MAGIC:
        raise ValueError(f"bad magic {buf[offset:offset + 4]!r}; expected {MAGIC!r}")
    version = buf[offset + 4]
    if version == 1:
        _, _, bits, dim, seed, norm, codelen = _HEADER_V1.unpack_from(buf, offset)
        rotation, hsize = "qr", _HEADER_V1.size
    elif version == 2:
        if len(buf) < offset + _HEADER_V2.size:
            raise ValueError("buffer too small for a TQE v2 header")
        _, _, bits, dim, seed, norm, rot, codelen = _HEADER_V2.unpack_from(buf, offset)
        if rot not in _ROTATIONS:
            raise ValueError(f"unknown rotation code {rot}")
        rotation, hsize = _ROTATIONS[rot], _HEADER_V2.size
    elif version == 3:
        if len(buf) < offset + _HEADER_V3.size:
            raise ValueError("buffer too small for a TQE v3 header")
        _, _, bits, dim, seed, norm, rot, cb, codelen = _HEADER_V3.unpack_from(
            buf, offset
        )
        if rot not in _ROTATIONS:
            raise ValueError(f"unknown rotation code {rot}")
        if cb not in _CODEBOOK_IDS:
            raise ValueError(f"unknown codebook code {cb}")
        rotation, hsize = _ROTATIONS[rot], _HEADER_V3.size
    else:
        raise ValueError(f"unsupported TQE format version {version}")
    if bits not in _VALID_BITS:
        raise ValueError(f"unsupported bits {bits}; expected one of {_VALID_BITS}")
    codebook = _CODEBOOK_IDS[cb] if version == 3 else "legacy"
    end = offset + hsize + codelen
    if len(buf) < end:
        raise ValueError("truncated TQE record (codes shorter than codelen)")
    rec = TQERecord(
        version=version,
        bits=bits,
        dim=dim,
        seed=seed,
        norm=norm,
        rotation=rotation,
        codes=buf[offset + hsize : end],
        codebook=codebook,
    )
    return rec, end


def read_records(buf: bytes) -> list[TQERecord]:
    """Parse a batch file: back-to-back records of any version."""
    out, off = [], 0
    while off < len(buf):
        rec, off = parse_record(buf, off)
        out.append(rec)
    return out


# ----------------------------------------------------------------- decoding
def _unpack_indices(codes: bytes, bits: int, dim: int) -> np.ndarray:
    """SPEC "Bit-packing": LSB-first indices, `dim` of them, `bits` each."""
    b = np.frombuffer(codes, dtype=np.uint8).astype(np.uint32)
    if bits == 2:
        vals = np.column_stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)]).ravel()
    elif bits == 3:
        trip = b.reshape(-1, 3)
        w = trip[:, 0] | (trip[:, 1] << 8) | (trip[:, 2] << 16)
        vals = np.column_stack([(w >> s) & 0x7 for s in range(0, 24, 3)]).ravel()
    else:  # bits == 4
        vals = np.column_stack([b & 0xF, (b >> 4) & 0xF]).ravel()
    return vals[:dim].astype(np.uint8)


def _fwht(a: np.ndarray) -> np.ndarray:
    """Unnormalized Fast Walsh-Hadamard Transform along the last axis."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    d = a.shape[-1]
    out = a.reshape(-1, d).copy()
    n, h = out.shape[0], 1
    while h < d:
        out = out.reshape(n, d // (2 * h), 2, h)
        x, y = out[:, :, 0, :], out[:, :, 1, :]
        out = np.stack((x + y, x - y), axis=2).reshape(n, d)
        h *= 2
    return out.reshape(a.shape)


def _unrotate(y: np.ndarray, dim: int, seed: int, rotation: str) -> np.ndarray:
    """SPEC "Decode algorithm" step 3: apply R(seed)^T.

    The rotation is fully determined by ``(dim, seed, rotation)``: SPEC "Rotation
    derivation" draws it from the frozen generator above, in the writer's order.
    """
    rng = default_rng(seed)
    if rotation == "hadamard":
        sign = rng.signs(dim).astype(np.float32)
        scale = np.float32(1.0 / math.sqrt(dim))
        return _fwht(y) * scale * sign
    if dim <= 4096:  # full QR of a seeded float32 Gaussian matrix
        g = rng.standard_normal((dim, dim)).astype(np.float32)
        q, _ = np.linalg.qr(g)
        return y @ q
    # structured rotation for very large dim: sign flip + permutation
    sign = rng.signs(dim).astype(np.float32)
    perm = rng.permutation(dim)
    inv_perm = np.argsort(perm)
    return y[..., inv_perm] / sign


def decode(rec: TQERecord) -> np.ndarray:
    """SPEC "Decode algorithm" steps 1-4: record -> float32 vector."""
    indices = _unpack_indices(rec.codes, rec.bits, rec.dim)
    table = _CODEBOOKS[rec.codebook][rec.bits]
    centroids = (table / math.sqrt(rec.dim)).astype(np.float32)
    y_hat = centroids[indices]
    x_hat = _unrotate(y_hat, rec.dim, rec.seed, rec.rotation)
    return (x_hat * np.float32(rec.norm)).astype(np.float32)


def decode_file(path: str) -> np.ndarray:
    """Decode every record in a batch file -> array of shape (n, dim)."""
    with open(path, "rb") as f:
        recs = read_records(f.read())
    return np.stack([decode(r) for r in recs])


# ---------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="a .tqe record/batch file")
    ap.add_argument("--out", help="write decoded vectors to this .npy")
    a = ap.parse_args(argv)
    with open(a.path, "rb") as f:
        recs = read_records(f.read())
    for i, r in enumerate(recs):
        print(
            f"[{i}] v{r.version} bits={r.bits} dim={r.dim} seed={r.seed} "
            f"rotation={r.rotation} codebook={r.codebook} norm={r.norm:.6g} "
            f"codes={len(r.codes)}B"
        )
    if a.out:
        np.save(a.out, np.stack([decode(r) for r in recs]))
        print(f"decoded {len(recs)} vectors -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
