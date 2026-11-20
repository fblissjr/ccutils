"""The reader that takes Antigravity protobuf apart by field number.

There is no published schema, so the shaped layer reads fields by number and
each reading is a measured claim in docs/ANTIGRAVITY_CONTRACT.md. This file
holds the reader itself: against the protobuf encoding guide's worked bytes
(an expected-value oracle), and against the test-side encoder the fixtures
are built with (agreement between two implementations is not correctness on
its own).
"""

from datetime import datetime, timezone

import pytest

from ccutils.parsers.antigravity import decode

from helpers_antigravity import f_bytes, f_msg, f_str, f_timestamp, f_varint


class TestGoldenBytes:
    def test_the_guides_worked_examples(self):
        # protobuf.dev/programming-guides/encoding: Test1 {a = 150}, Test2 {b = "testing"}
        assert list(decode.fields(bytes.fromhex("089601"))) == [(1, 0, 150)]
        assert list(decode.fields(bytes.fromhex("120774657374696e67"))) == [(2, 2, b"testing")]

    def test_an_embedded_message(self):
        # Test3 {c = Test1 {a = 150}} -> 1a 03 08 96 01
        assert decode.get(bytes.fromhex("1a03089601"), 3, 1) == 150

    def test_fixed_width_fields_are_returned_as_their_bytes(self):
        buf = bytes.fromhex("0d01000000") + bytes.fromhex("110100000000000000")   # 1: fixed32, 2: fixed64
        assert list(decode.fields(buf)) == [(1, 5, b"\x01\x00\x00\x00"), (2, 1, b"\x01" + b"\x00" * 7)]


class TestPaths:
    BUF = f_varint(1, 15) + f_msg(5, f_timestamp(1, 1_790_000_000, 5), f_str(12, "exec-1")) + f_msg(
        20, f_str(1, "hello"), f_msg(7, f_str(1, "call_1"), f_str(2, "run_command")),
        f_msg(7, f_str(1, "call_2"), f_str(2, "read_file")))

    def test_get_walks_nested_messages(self):
        assert decode.get(self.BUF, 1) == 15
        assert decode.get(self.BUF, 5, 12) == b"exec-1"
        assert decode.get(self.BUF, 20, 1) == b"hello"

    def test_an_absent_hop_is_none_not_an_error(self):
        assert decode.get(self.BUF, 9) is None
        assert decode.get(self.BUF, 5, 99, 1) is None
        assert decode.get(self.BUF, 1, 2) is None        # a varint has nothing inside it
        assert decode.get(None, 1) is None

    def test_a_repeated_field_is_read_whole_and_in_order(self):
        calls = decode.get_all(self.BUF, 20, 7)
        assert [decode.text(decode.get(c, 2)) for c in calls] == ["run_command", "read_file"]
        assert decode.get_all(self.BUF, 20, 99) == [] and decode.get_all(None, 1) == []

    def test_the_last_occurrence_of_a_scalar_wins(self):
        assert decode.get(f_varint(4, 3) + f_varint(4, 5), 4) == 5

    def test_garbage_raises_and_never_returns_a_value(self):
        with pytest.raises(ValueError):
            list(decode.fields(b"\x0a\x05ab"))           # length runs past the end
        with pytest.raises(ValueError):
            list(decode.fields(b"\x0b"))                 # wire type 3 (a group)


class TestValues:
    def test_text_is_strict_utf8(self):
        assert decode.text(b"caf\xc3\xa9") == "café"
        assert decode.text(b"\xff\xfe") is None          # bytes that are not text stay unread
        assert decode.text(None) is None and decode.text(15) is None

    def test_a_timestamp_message(self):
        when = decode.timestamp(decode.get(f_timestamp(1, 1_790_000_000, 500_000_000), 1))
        assert when == datetime(2026, 9, 21, 14, 13, 20, 500_000, tzinfo=timezone.utc)
        assert decode.timestamp(None) is None
        assert decode.timestamp(b"") is None             # seconds omitted: no time was stated

    def test_a_negative_number_stored_unsigned(self):
        # proto3 int64 -1 is ten bytes of varint; the CLI stores exit code -1 this way.
        raw = decode.get(f_varint(1, -1), 1)
        assert raw == (1 << 64) - 1 and decode.signed(raw) == -1
        assert decode.signed(127) == 127 and decode.signed(None) is None

    def test_packed_varints(self):
        packed = f_bytes(2, bytes.fromhex("03" "8e02" "9ea705"))   # the guide's [3, 270, 86942]
        assert decode.packed_varints(decode.get(packed, 2)) == [3, 270, 86942]
        assert decode.packed_varints(None) == []
