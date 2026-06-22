"""Keys default to 4-bit in AutoConfig (attention scores amplify key-quant error)."""

from turboquant_pro.autoconfig import _TARGETS


def test_keys_4bit_except_extreme():
    for name, p in _TARGETS.items():
        if name == "extreme":
            continue
        assert p["key_bits"] == 4, f"{name} should default keys to 4-bit"


def test_compression_is_k4_v2():
    assert _TARGETS["compression"]["key_bits"] == 4
    assert _TARGETS["compression"]["value_bits"] == 2
