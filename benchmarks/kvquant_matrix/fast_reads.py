"""``FAST_LOAD=1``: from_pretrained reads each weight with one large pread, not through mmap.

transformers 5.5 opens every safetensors file with ``safe_open`` (an mmap) and hands its
four loader threads one lazy slice per tensor; ``tensor[...]`` then reads the tensor by
page faults. On NRP's linstor volumes that ran at ~140 MB/s however the files were laid
out: Mistral-7B's 14.5 GB took 106 s from the striped copy (2026-10-19), with the GPU idle,
and NRP deleted the head wave's Jobs before they started work. Here, while ``reads()`` is
entered, the ``safe_open`` that modeling_utils calls is wrapped so that ``tensor[...]`` on
a whole tensor reads its byte range with ``os.preadv`` into a CPU tensor of the file's
dtype and shape, and drops those pages from the cache. The bytes are the file's bytes and
everything else in from_pretrained (config, dtype casts, device placement, buffers, tied
weights, generation config, the loader's order and threads) runs unchanged; any other
use of the file (keys, metadata, partial slices) goes to the real ``safe_open``. Default
off: without FAST_LOAD=1, ``reads()`` changes nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct

import torch

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
CHUNK = 256 << 20  # bytes per preadv call
stats = {"tensors": 0, "bytes": 0}  # what reads() read (the tests check it was used)


def header(path: str):
    """(tensor entries, data start) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return head, 8 + n


class _Slice:
    """A lazy tensor: a whole-tensor ``[...]`` is one pread; anything else is the real slice."""

    def __init__(self, real, fd: int, entry: dict, start: int):
        self._real, self._fd = real, fd
        self._dtype = DTYPES[entry["dtype"]]
        self._shape = entry["shape"]
        b, e = entry["data_offsets"]
        self._begin, self._n = start + b, e - b

    def __getitem__(self, idx):
        if idx is not Ellipsis:
            return self._real[idx]
        out = torch.empty(self._n, dtype=torch.uint8)
        view = memoryview(out.numpy())
        done = 0
        while done < self._n:
            got = os.preadv(self._fd, [view[done:done + CHUNK]], self._begin + done)
            if got <= 0:
                raise OSError(f"short read at byte {self._begin + done}")
            done += got
        os.posix_fadvise(self._fd, self._begin, self._n, os.POSIX_FADV_DONTNEED)
        stats["tensors"] += 1
        stats["bytes"] += self._n
        if self._n == 0:
            return torch.empty(self._shape, dtype=self._dtype)
        return out.view(self._dtype).reshape(self._shape)

    def __getattr__(self, name):  # get_dtype, get_shape, ... from the real slice
        return getattr(self._real, name)


class _File:
    """The real safe_open handle, with get_slice returning pread-backed slices."""

    def __init__(self, real, path: str):
        self._real = real
        self._head, self._start = header(path)
        self._fd = os.open(path, os.O_RDONLY)

    def get_slice(self, name):
        return _Slice(self._real.get_slice(name), self._fd, self._head[name], self._start)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        os.close(self._fd)
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def reads():
    """Within this block, from_pretrained reads whole tensors by pread (FAST_LOAD=1)."""
    if os.environ.get("FAST_LOAD") != "1":
        yield
        return
    from transformers import modeling_utils as mu

    real = mu.safe_open

    def wrapped(path, *a, **kw):
        f = real(path, *a, **kw)
        framework = kw.get("framework", a[0] if a else None)
        device = kw.get("device", a[1] if len(a) > 1 else "cpu")
        return _File(f, path) if framework == "pt" and device == "cpu" else f

    mu.safe_open = wrapped
    try:
        yield
    finally:
        mu.safe_open = real
