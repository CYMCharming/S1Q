"""Compact signed 2/3-bit payloads for symmetric groupwise weight codes.

The on-disk order is a continuous little-endian bitstream: code zero starts
at the least significant bit of byte zero. The unused high bits of the last
byte are zero. Packing changes storage only; decoding still uses float GEMM.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def packed_nbytes(numel: int, bits: int) -> int:
    if numel < 0 or bits not in (2, 3):
        raise ValueError("Expected nonnegative code count and 2/3 weight bits.")
    return math.ceil(numel * bits / 8)


def pack_signed_codes(codes: Tensor, bits: int) -> Tensor:
    """Return a CPU uint8 payload without allocating a giant word matrix."""
    count = codes.numel()
    size = packed_nbytes(count, bits)
    if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Weight codes must use a signed integral dtype.")
    flat = codes.detach().reshape(-1).to(device="cpu")
    qmax = (1 << (bits - 1)) - 1
    if ((flat < -qmax) | (flat > qmax)).any():
        raise ValueError("Weight codes exceed the symmetric signed range.")
    packed = torch.empty(size, dtype=torch.uint8)
    chunk_size = 1 << 20  # Divisible by eight, so chunk boundaries are byte aligned.
    offsets = torch.arange(8, dtype=torch.int64) * bits
    for first in range(0, count, chunk_size):
        part = flat[first:first + chunk_size].to(torch.int64)
        if part.numel() % 8:
            part = F.pad(part, (0, 8 - part.numel() % 8))
        words = ((part.reshape(-1, 8) & ((1 << bits) - 1)) << offsets).sum(dim=1)
        encoded = torch.stack([(words >> (8 * byte)) & 255 for byte in range(bits)], dim=1)
        destination = first * bits // 8
        remaining = min(encoded.numel(), size - destination)
        packed[destination:destination + remaining] = encoded.reshape(-1)[:remaining].to(torch.uint8)
    return packed


def unpack_signed_codes(payload: Tensor, count: int, bits: int, *, validate_padding: bool = True) -> Tensor:
    """Decode a payload on its existing device; result is flat int8 codes."""
    expected = packed_nbytes(count, bits)
    if payload.dtype != torch.uint8 or payload.ndim != 1 or payload.numel() != expected:
        raise ValueError("Packed payload dtype or length does not match the declared shape.")
    if count == 0:
        return torch.empty(0, dtype=torch.int8, device=payload.device)
    if validate_padding and count * bits % 8:
        used = count * bits % 8
        if int(payload[-1].item()) >> used:
            raise ValueError("Unused high bits of the final payload byte must be zero.")
    starts = torch.arange(count, device=payload.device, dtype=torch.int64) * bits
    first_byte = starts // 8
    offset = starts % 8
    low = payload[first_byte].to(torch.int32) >> offset
    high = payload[(first_byte + 1).clamp_max(expected - 1)].to(torch.int32) << (8 - offset)
    unsigned = (low | high) & ((1 << bits) - 1)
    sign = 1 << (bits - 1)
    return torch.where(unsigned >= sign, unsigned - (1 << bits), unsigned).to(torch.int8)
