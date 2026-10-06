"""2/3-bit storage parity and malformed-payload checks."""

import math

import pytest
import torch

from s1q.bitpack import pack_signed_codes, packed_nbytes, unpack_signed_codes


@pytest.mark.parametrize("bits", (2, 3))
@pytest.mark.parametrize("count", (0, 1, 2, 3, 7, 8, 9, 127, 128, 129, 4097))
def test_round_trip(bits, count):
    qmax = (1 << (bits - 1)) - 1
    codes = torch.randint(-qmax, qmax + 1, (count,), dtype=torch.int8)
    packed = pack_signed_codes(codes, bits)
    assert packed.dtype == torch.uint8
    assert packed.numel() == packed_nbytes(count, bits)
    assert torch.equal(unpack_signed_codes(packed, count, bits), codes)


@pytest.mark.parametrize("bits", (2, 3))
def test_invalid_code_and_padding(bits):
    qmax = (1 << (bits - 1)) - 1
    with pytest.raises(ValueError, match="range"):
        pack_signed_codes(torch.tensor([-(qmax + 1)]), bits)
    # Validate in the source dtype: narrowing first would wrap 65537 to 1.
    with pytest.raises(ValueError, match="range"):
        pack_signed_codes(torch.tensor([65537], dtype=torch.int64), bits)
    good = pack_signed_codes(torch.tensor([qmax], dtype=torch.int8), bits)
    good[-1] |= 1 << bits
    with pytest.raises(ValueError, match="Unused high bits"):
        unpack_signed_codes(good, 1, bits)


def test_three_bit_stream_crosses_byte_boundary():
    codes = torch.tensor([-3, -2, -1, 0, 1, 2, 3, 0, -3], dtype=torch.int8)
    packed = pack_signed_codes(codes, 3)
    assert packed.numel() == math.ceil(codes.numel() * 3 / 8)
    assert torch.equal(unpack_signed_codes(packed, codes.numel(), 3), codes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_decode_matches_cpu():
    codes = torch.randint(-3, 4, (10001,), dtype=torch.int8)
    packed = pack_signed_codes(codes, 3)
    decoded = unpack_signed_codes(packed.cuda(), codes.numel(), 3)
    assert torch.equal(decoded.cpu(), codes)
