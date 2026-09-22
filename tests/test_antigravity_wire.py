"""The test-side protobuf codec agrees with the protobuf encoding spec.

Fixtures build Antigravity blobs with this encoder and contract canaries read
real blobs with this decoder. If both were only checked against each other, a
shared misreading of the wire format would pass every test while every
fixture and every canary quietly measured the wrong field. The golden bytes
below are the worked examples from the protobuf encoding guide
(protobuf.dev/programming-guides/encoding).
"""

import pytest

from helpers_antigravity import (
    f_msg,
    f_str,
    f_timestamp,
    f_varint,
    get,
    parse,
    timestamp_seconds,
    varint,
)


class TestGoldenBytes:
    def test_varint_150(self):
        # "150 is encoded as 9601" -- the guide's varint example.
        assert varint(150) == bytes.fromhex("9601")

    def test_int32_field_1(self):
        # message Test1 { int32 a = 1; } with a = 150 -> 08 96 01
        assert f_varint(1, 150) == bytes.fromhex("089601")
        assert parse(bytes.fromhex("089601")) == [(1, 0, 150)]

    def test_string_field_2(self):
        # message Test2 { string b = 2; } with b = "testing" -> 12 07 74 65 73 74 69 6e 67
        golden = bytes.fromhex("120774657374696e67")
        assert f_str(2, "testing") == golden
        assert parse(golden) == [(2, 2, b"testing")]

    def test_embedded_message_field_3(self):
        # message Test3 { Test1 c = 3; } with c.a = 150 -> 1a 03 08 96 01
        golden = bytes.fromhex("1a03089601")
        assert f_msg(3, f_varint(1, 150)) == golden
        assert get(golden, 3, 1) == 150


class TestDecoder:
    def test_last_wins_for_repeated_scalar(self):
        # Protobuf semantics: a non-repeated scalar seen twice takes the last value.
        assert get(f_varint(1, 5) + f_varint(1, 9), 1) == 9

    def test_absent_path_is_none(self):
        assert get(f_varint(1, 5), 2) is None
        assert get(f_varint(1, 5), 1, 1) is None  # 1 is a varint, not a message

    def test_truncated_buffer_raises(self):
        with pytest.raises(ValueError):
            parse(f_str(2, "testing")[:-1])

    def test_timestamp(self):
        buf = f_timestamp(1, 1_790_000_000, 500_000_000)
        assert timestamp_seconds(get(buf, 1)) == pytest.approx(1_790_000_000.5)

    def test_zero_timestamp_omits_fields(self):
        # proto3 omits zero values, so an all-zero Timestamp is an empty message.
        assert f_timestamp(1, 0) == bytes.fromhex("0a00")


class TestLakeFieldReader:
    """`parsers/antigravity/wire.field_bytes`, the one protobuf read the lake
    makes, checked against the same golden bytes and against the test codec."""

    def test_golden_bytes(self):
        from ccutils.parsers.antigravity.wire import field_bytes
        assert field_bytes(bytes.fromhex("089601"), 1) == bytes.fromhex("9601")
        assert field_bytes(bytes.fromhex("120774657374696e67"), 2) == b"testing"
        assert field_bytes(bytes.fromhex("1a03089601"), 3) == bytes.fromhex("089601")

    def test_agrees_with_the_test_codec_on_a_step_metadata(self):
        from ccutils.parsers.antigravity.wire import field_bytes
        meta = f_timestamp(1, 1_790_000_000, 5) + f_str(12, "exec-1") + f_msg(4, f_str(1, "call"))
        assert field_bytes(meta, 1) == get(meta, 1)
        assert field_bytes(meta, 12) == b"exec-1"
        assert field_bytes(meta, 7) is None

    def test_last_wins_and_bad_wire_raises(self):
        from ccutils.parsers.antigravity.wire import field_bytes
        assert field_bytes(f_str(1, "a") + f_str(1, "b"), 1) == b"b"
        with pytest.raises(ValueError):
            field_bytes(f_str(2, "testing")[:-1], 2)
