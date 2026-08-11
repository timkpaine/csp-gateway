import pytest
from pydantic import TypeAdapter, ValidationError

from csp_gateway.utils.struct import GatewayStruct


def test_runs_on_pydantic_path():
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: "a<0" if s.a < 0 else None)
    assert TypeAdapter(S).validate_python({"a": 1}).a == 1
    with pytest.raises(ValidationError, match="a<0"):
        TypeAdapter(S).validate_python({"a": -1})


def test_mro_aggregation():
    class Base(GatewayStruct):
        a: int = 0

    class Child(Base):
        b: int = 0

    Base.add_validator(lambda s: "base-fail" if s.a == 1 else None)
    Child.add_validator(lambda s: "child-fail" if getattr(s, "b", 0) == 1 else None)

    with pytest.raises(ValidationError, match="base-fail"):
        TypeAdapter(Child).validate_python({"a": 1})
    with pytest.raises(ValidationError, match="child-fail"):
        TypeAdapter(Child).validate_python({"b": 1})
    # child validator does not leak upward to Base
    assert TypeAdapter(Base).validate_python({"a": 0}).a == 0


def test_isolation_between_sibling_classes():
    class A(GatewayStruct):
        x: int = 0

    class B(GatewayStruct):
        x: int = 0

    A.add_validator(lambda s: "only-A" if s.x == 5 else None)
    assert TypeAdapter(B).validate_python({"x": 5}).x == 5
    with pytest.raises(ValidationError, match="only-A"):
        TypeAdapter(A).validate_python({"x": 5})


def test_runs_even_when_after_hook_overridden_without_super():
    # REGRESSION: WT structs override _validate_gateway_struct_after WITHOUT super(). The registry must
    # STILL run because it executes in the wrap validator (_validate_gateway_struct), not the after hook.
    class S(GatewayStruct):
        a: int = 0

        @classmethod
        def _validate_gateway_struct_after(cls, val):
            return val

    ran = []
    S.add_validator(lambda s: ran.append(1) and None)
    TypeAdapter(S).validate_python({"a": 1})
    assert ran, "registry must run even when the after hook is overridden without super()"


def test_validator_runs_exactly_once():
    class S(GatewayStruct):
        a: int = 0

    count = []
    S.add_validator(lambda s: count.append(1) and None)
    TypeAdapter(S).validate_python({"a": 1})
    assert len(count) == 1, f"validator should run exactly once, ran {len(count)}"


def test_no_validators_is_noop():
    class S(GatewayStruct):
        a: int = 0

    assert TypeAdapter(S).validate_python({"a": 7}).a == 7


def test_add_validator_usable_as_decorator():
    class S(GatewayStruct):
        a: int = 0

    @S.add_validator
    def _v(s):
        return "bad-a" if s.a == 9 else None

    assert _v is not None
    with pytest.raises(ValidationError, match="bad-a"):
        TypeAdapter(S).validate_python({"a": 9})


def test_multiple_validators_all_run_first_failure_wins():
    class S(GatewayStruct):
        a: int = 0

    order = []
    S.add_validator(lambda s: (order.append("first"), "stop" if s.a == 1 else None)[1])
    S.add_validator(lambda s: (order.append("second"), None)[1])
    TypeAdapter(S).validate_python({"a": 0})
    assert order == ["first", "second"]
    order.clear()
    with pytest.raises(ValidationError, match="stop"):
        TypeAdapter(S).validate_python({"a": 1})
    assert order == ["first"]


def test_manual_call_on_csp_path():
    # Native/CSP construction does NOT auto-run validators; they can be invoked explicitly.
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: "neg" if s.a < 0 else None)
    obj = S(a=-1)
    with pytest.raises(ValueError, match="neg"):
        S._run_validators(obj)


class _VChild(GatewayStruct):
    v: int = 0


class _VParent(GatewayStruct):
    child: _VChild
    name: str = ""


def test_nested_struct_validator_fires_on_parent_validation():
    _VChild.add_validator(lambda c: "child-bad" if c.v < 0 else None)
    try:
        with pytest.raises(ValidationError, match="child-bad"):
            TypeAdapter(_VParent).validate_python({"name": "p", "child": {"v": -1}})
    finally:
        _VChild.clear_validators()


def test_raising_validator_behavior():
    # A validator should RETURN an error string. If it raises, pydantic wraps ValueError/AssertionError
    # into a ValidationError but lets other exceptions propagate raw -- document both.
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: (_ for _ in ()).throw(ValueError("as-value-error")))
    with pytest.raises(ValidationError, match="as-value-error"):
        TypeAdapter(S).validate_python({"a": 1})

    S.clear_validators()

    def _boom(s):
        raise RuntimeError("raw-runtime")

    S.add_validator(_boom)
    with pytest.raises(RuntimeError, match="raw-runtime"):
        TypeAdapter(S).validate_python({"a": 1})


def test_registry_runs_alongside_existing_after_hook():
    # A struct that overrides the after-hook to enforce its own rule AND has a registered validator:
    # BOTH must be enforced.
    class S(GatewayStruct):
        a: int = 0
        b: int = 0

        @classmethod
        def _validate_gateway_struct_after(cls, val):
            if val.a < 0:
                raise ValueError("after-hook: a<0")
            return val

    S.add_validator(lambda s: "registry: b<0" if s.b < 0 else None)
    # after-hook fires
    with pytest.raises(ValidationError, match="after-hook: a<0"):
        TypeAdapter(S).validate_python({"a": -1, "b": 0})
    # registry fires
    with pytest.raises(ValidationError, match="registry: b<0"):
        TypeAdapter(S).validate_python({"a": 0, "b": -1})
    # both pass
    assert TypeAdapter(S).validate_python({"a": 1, "b": 1}).a == 1
