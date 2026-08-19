"""Tests for the Kafka engine-cycle envelope.

The envelope is the one place gateway data is framed by hand rather than by csp. Its shape has to
stay byte-compatible with csp's ``JSONTextMessageMapper``, because the engine-timestamp subscribe
path still reads it back through that mapper to recover ``csp_timestamp`` for
``tick_timestamp_from_field``.
"""

from datetime import datetime

import orjson

from csp_gateway.server.modules.kafka.kafka import (
    _ENVELOPE_ENCODING_FIELD,
    _ENVELOPE_TIMESTAMP_FIELD,
    _decode_envelope,
    _encode_envelope,
)

ENCODING = '{"foo":1.0}'
TIMESTAMP = datetime(2020, 1, 1, 0, 0, 1)
# csp writes DateTimeType.UINT64_MILLIS as epoch milliseconds, so this is what the mapper expects.
EPOCH_MILLIS = 1577836801000


def test_encodes_the_two_wire_fields():
    assert orjson.loads(_encode_envelope(ENCODING, TIMESTAMP)) == {
        _ENVELOPE_ENCODING_FIELD: ENCODING,
        _ENVELOPE_TIMESTAMP_FIELD: EPOCH_MILLIS,
    }


def test_timestamp_is_epoch_millis_not_a_string():
    # A string here would still round trip through _decode_envelope but would fail csp's parser.
    assert isinstance(orjson.loads(_encode_envelope(ENCODING, TIMESTAMP))[_ENVELOPE_TIMESTAMP_FIELD], int)


def test_round_trips():
    cycle = _decode_envelope(_encode_envelope(ENCODING, TIMESTAMP))
    assert cycle.encoding == ENCODING
    assert cycle.csp_timestamp == TIMESTAMP


def test_decodes_what_csps_mapper_would_produce():
    message = orjson.dumps({_ENVELOPE_ENCODING_FIELD: ENCODING, _ENVELOPE_TIMESTAMP_FIELD: EPOCH_MILLIS}).decode()
    cycle = _decode_envelope(message)
    assert cycle.encoding == ENCODING
    assert cycle.csp_timestamp == TIMESTAMP


def test_decodes_a_message_without_a_timestamp():
    message = orjson.dumps({_ENVELOPE_ENCODING_FIELD: ENCODING}).decode()
    assert _decode_envelope(message).csp_timestamp is None


def test_encoding_is_nested_as_a_string_not_inlined():
    # The payload is opaque to the envelope; inlining it would collide with the envelope's own keys.
    assert orjson.loads(_encode_envelope('{"csp_timestamp":"collide"}', TIMESTAMP))[_ENVELOPE_TIMESTAMP_FIELD] == EPOCH_MILLIS
