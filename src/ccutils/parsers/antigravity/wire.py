"""The one protobuf read Tier 1 makes: a top-level field's raw bytes.

The lake decodes nothing into its tables. It reads exactly one field, to
decide whether a new snapshot continues the old one: Step metadata field 1,
the step's creation time (docs/ANTIGRAVITY_CONTRACT.md claims 6 and 14). The
bytes are compared, never interpreted or stored. Full decoding is Phase 1's
job and will use the app's own descriptors, not this.
"""

from __future__ import annotations


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return result, i
        if shift > 63:
            raise ValueError("varint too long")


def field_bytes(buf: bytes, field_no: int) -> bytes | None:
    """The raw value bytes of top-level `field_no` (last occurrence wins), or None.

    For a varint field that is its encoded varint; for length-delimited, the
    payload; for fixed32/64, the four or eight bytes. Raises ValueError on
    wire data that does not parse, so a caller never compares garbage."""
    found = None
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        number, wire_type = key >> 3, key & 7
        start = i
        if wire_type == 0:
            _, i = _varint(buf, i)
        elif wire_type == 1:
            i += 8
        elif wire_type == 2:
            length, i = _varint(buf, i)
            start = i
            i += length
        elif wire_type == 5:
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        if i > n:
            raise ValueError("truncated field")
        if number == field_no:
            found = bytes(buf[start:i])
    return found
