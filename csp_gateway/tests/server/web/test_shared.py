import json
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum

import pytest
import starlette.responses
from pydantic import BaseModel

from csp_gateway import GatewayStruct
from csp_gateway.server.demo import ExampleData
from csp_gateway.server.web import prepare_response
from csp_gateway.server.web.routes.shared import get_fully_qualified_type_name
from csp_gateway.utils.struct.base import type_adapter_for


def test_prepare_response():
    data = ExampleData()
    response = prepare_response(res=data, is_list_model=False, is_dict_basket=False, wrap_in_response=True)
    assert isinstance(response, starlette.responses.Response)
    response = prepare_response(res=data, is_list_model=False, is_dict_basket=False, wrap_in_response=False)
    assert json.loads(response) == [json.loads(data.type_adapter().dump_json(data))]


def test_prepare_response_list():
    data = ExampleData()
    response = prepare_response(res=[data], is_list_model=False, is_dict_basket=False, wrap_in_response=False)
    assert json.loads(response) == [json.loads(data.type_adapter().dump_json(data))]
    response = prepare_response(res=[data], is_list_model=True, is_dict_basket=False, wrap_in_response=False)
    assert json.loads(response) == [json.loads(data.type_adapter().dump_json(data))]
    response = prepare_response(res=(data,), is_list_model=True, is_dict_basket=False, wrap_in_response=False)
    assert json.loads(response) == [json.loads(data.type_adapter().dump_json(data))]


def test_prepare_response_dict():
    data = ExampleData()
    with pytest.raises(AttributeError):
        prepare_response(res={"foo": data}, is_list_model=False, is_dict_basket=False, wrap_in_response=False)

    response = prepare_response(res={"foo": data}, is_list_model=False, is_dict_basket=True, wrap_in_response=False)
    assert json.loads(response) == [json.loads(data.type_adapter().dump_json(data))]


def test_get_fully_qualified_type_name_with_model():
    """Test that get_fully_qualified_type_name returns the correct fully qualified type name."""
    fq_name = get_fully_qualified_type_name(ExampleData)
    assert fq_name == "csp_gateway.server.demo.omnibus.ExampleData"


def test_get_fully_qualified_type_name_with_list_model():
    """Test that get_fully_qualified_type_name extracts the type from a List."""
    fq_name = get_fully_qualified_type_name(list[ExampleData])
    assert fq_name == "csp_gateway.server.demo.omnibus.ExampleData"


def test_get_fully_qualified_type_name_with_none():
    """Test that get_fully_qualified_type_name returns empty string for None."""
    fq_name = get_fully_qualified_type_name(None)
    assert fq_name == ""


class _Color(Enum):
    RED = 1
    BLUE = 2


class _Inner(GatewayStruct):
    label: str = ""
    values: list[float] = []


class _Record(GatewayStruct):
    x: int = 0
    f: float = 0.0
    color: _Color = _Color.RED
    maybe: str | None = None
    aware: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    naive: datetime = datetime(2026, 1, 1)
    inner: _Inner = _Inner()
    inners: list[_Inner] = []


class _SubRecord(_Record):
    extra: str = "sub"


class _Model(BaseModel):
    a: int = 1


def _old_dump(res):
    """The per-record serialization prepare_response used before batching."""
    return b"[" + b",".join(type_adapter_for(r).dump_json(r) for r in res) + b"]"


def _records():
    floats = [1.5, float("nan"), float("inf"), float("-inf"), -0.0]
    return [
        _Record(
            x=i,
            f=floats[i % len(floats)],
            color=_Color.BLUE if i % 2 else _Color.RED,
            maybe=None if i % 3 else f"m{i}",
            aware=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-5))) + timedelta(seconds=i),
            naive=datetime(2026, 1, 1) + timedelta(microseconds=i),
            inner=_Inner(label=f"in{i}", values=[float(i), float("nan")]),
            inners=[_Inner(label=str(j)) for j in range(i % 3)],
        )
        for i in range(20)
    ]


@pytest.mark.parametrize(
    "res",
    [
        pytest.param(_records(), id="same_type"),
        pytest.param([_Record()], id="single"),
        pytest.param([_Record(), _SubRecord(extra="x"), _Record()], id="mixed_subclass"),
        pytest.param([_SubRecord(), _SubRecord()], id="all_subclass"),
        pytest.param([_Model(a=1), _Model(a=2)], id="plain_pydantic"),
        pytest.param([ExampleData(x=1), ExampleData(x=2)], id="example_data"),
    ],
)
def test_prepare_response_bytes_match_per_record(res):
    expected = _old_dump(res)
    response = prepare_response(res=res, is_list_model=True)
    assert response.body == expected
    assert prepare_response(res=res, is_list_model=True, wrap_in_response=False) == expected.decode()


def test_prepare_response_mixed_subclass_keeps_subclass_fields():
    body = json.loads(prepare_response(res=[_Record(), _SubRecord(extra="x")], is_list_model=True).body)
    assert "extra" not in body[0]
    assert body[1]["extra"] == "x"


def test_prepare_response_empty():
    assert prepare_response(res=[], is_list_model=True).body == b"[]"
    assert prepare_response(res=None, wrap_in_response=False) == "[]"
