"""Read Antigravity protobuf by field number.

Antigravity publishes no schema. The shaped layer reads fields by number,
and every reading (which number holds what) is a measured claim in
docs/ANTIGRAVITY_CONTRACT.md with a canary beside it; nothing here knows a
field's meaning. This module is only the wire format: it never guesses that
a length-delimited value is a string, a message or packed scalars, because
without a schema the three are indistinguishable. The caller, which knows
the claim it is reading under, says which.

The lake itself decodes nothing and uses `wire.field_bytes` alone; this is
for what is derived from the lake.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

Value = int | bytes


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


def fields(buf: bytes) -> Iterator[tuple[int, int, Value]]:
    """(field number, wire type, value) for each top-level field, in order.

    A varint is an int; a length-delimited or fixed-width value is its bytes.
    Raises ValueError on data that is not valid wire format, so a caller
    never reads a value out of garbage."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        number, wire_type = key >> 3, key & 7
        if number == 0:
            raise ValueError("field number 0")
        value: Value
        if wire_type == 0:
            value, i = _varint(buf, i)
        elif wire_type == 2:
            length, i = _varint(buf, i)
            if i + length > n:
                raise ValueError("truncated length-delimited field")
            value, i = bytes(buf[i:i + length]), i + length
        elif wire_type in (1, 5):
            width = 8 if wire_type == 1 else 4
            if i + width > n:
                raise ValueError("truncated fixed-width field")
            value, i = bytes(buf[i:i + width]), i + width
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        yield number, wire_type, value


def get(buf: Value | None, *path: int) -> Value | None:
    """The value at a field path; None when any hop is absent.

    Every hop but the last must be a sub-message. Protobuf's rule for a
    scalar that occurs twice applies: the last one wins. proto3 does not
    write a zero, so None means "absent or zero" for a number and the caller
    decides which the claim allows."""
    current = buf
    for number in path:
        if not isinstance(current, (bytes, bytearray)):
            return None
        found = None
        for n, _, value in fields(bytes(current)):
            if n == number:
                found = value
        if found is None:
            return None
        current = found
    return current


def get_all(buf: Value | None, *path: int) -> list[Value]:
    """Every occurrence of the last field of `path`, in order: a repeated field."""
    *hops, last = path
    parent = get(buf, *hops) if hops else buf
    if not isinstance(parent, (bytes, bytearray)):
        return []
    return [value for n, _, value in fields(bytes(parent)) if n == last]


def text(value: Value | None) -> str | None:
    """A length-delimited value as text, or None when it is not valid UTF-8."""
    if not isinstance(value, (bytes, bytearray)):
        return None
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError:
        return None


def signed(value: Value | None) -> int | None:
    """A varint read as int64: negatives are stored as their two's complement."""
    if not isinstance(value, int):
        return None
    return value - (1 << 64) if value >= 1 << 63 else value


def packed_varints(value: Value | None) -> list[int]:
    """A packed repeated varint field."""
    if not isinstance(value, (bytes, bytearray)):
        return []
    out, i, buf = [], 0, bytes(value)
    while i < len(buf):
        v, i = _varint(buf, i)
        out.append(v)
    return out


def timestamp(value: Value | None) -> datetime | None:
    """A google.protobuf.Timestamp message {1: seconds, 2: nanos}, in UTC.

    None when no seconds are stated: an empty message is not the epoch."""
    if not isinstance(value, (bytes, bytearray)):
        return None
    seconds: int | None = None
    nanos = 0
    for n, _, v in fields(bytes(value)):
        if isinstance(v, int) and n == 1:
            seconds = v
        elif isinstance(v, int) and n == 2:
            nanos = v
    if seconds is None:
        return None
    return _EPOCH + timedelta(seconds=signed(seconds) or 0, microseconds=nanos // 1000)
