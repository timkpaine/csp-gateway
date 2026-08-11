"""Tests for the unified GatewayStruct validator registry (``add_validator`` / ``clear_validators``).

Validators run automatically during pydantic validation (REST ``/send``, ``model_validate``/
``TypeAdapter``, JSON snapshot replay) and for nested structs when a containing struct is validated. A
validator ``fn(value) -> value`` returns the (possibly transformed) value and **raises** to reject it;
``mode="before"`` (alias ``"pre"``) runs on the raw input prior to construction, ``mode="after"`` (alias
``"post"``, default) on the constructed struct. They do NOT run on native ``MyStruct(...)`` construction.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from csp_gateway.utils.struct import GatewayStruct


def _bump(s, by=10):
    s2 = s.copy()
    s2.a = s.a + by
    return s2


def _set_a(s, a):
    s2 = s.copy()
    s2.a = a
    return s2


def _set_v(c, v):
    c2 = c.copy()
    c2.v = v
    return c2


def test_after_validator_can_transform_during_validation():
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: _set_a(s, s.a * 10), mode="after")
    assert TypeAdapter(S).validate_python({"a": 5}).a == 50


def test_before_validator_reshapes_raw_input():
    class S(GatewayStruct):
        a: int = 0

    # A "before" validator that accepts a legacy key `old_a` and folds it into `a` (like coercion).
    S.add_validator(lambda d: {**{k: v for k, v in d.items() if k != "old_a"}, "a": d["old_a"]} if "old_a" in d else d, mode="before")
    assert TypeAdapter(S).validate_python({"old_a": 7}).a == 7


def test_after_is_the_default_mode():
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: _bump(s))  # no mode -> "after"
    assert TypeAdapter(S).validate_python({"a": 1}).a == 11


def test_validator_rejects_by_raising():
    class S(GatewayStruct):
        a: int = 0

    def _reject_neg(s):
        if s.a < 0:
            raise ValueError("a<0")
        return s

    S.add_validator(_reject_neg)
    assert TypeAdapter(S).validate_python({"a": 1}).a == 1
    with pytest.raises(ValidationError, match="a<0"):
        TypeAdapter(S).validate_python({"a": -1})


def test_before_validator_rejects_by_raising():
    class S(GatewayStruct):
        a: int = 0

    def _reject(d):
        if isinstance(d, dict) and d.get("a", 0) < 0:
            raise ValueError("pre-reject")
        return d

    S.add_validator(_reject, mode="before")
    assert TypeAdapter(S).validate_python({"a": 1}).a == 1
    with pytest.raises(ValidationError, match="pre-reject"):
        TypeAdapter(S).validate_python({"a": -1})


def test_runs_on_pydantic_path_only_not_native_construction():
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: _set_a(s, 999), mode="after")
    assert TypeAdapter(S).validate_python({"a": 1}).a == 999  # pydantic path
    assert S(a=1).a == 1  # native construction bypasses validators


def test_pre_and_post_are_aliases_for_before_and_after():
    calls = []

    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda d: (calls.append("pre"), d)[1], mode="pre")
    S.add_validator(lambda s: (calls.append("post"), s)[1], mode="post")
    TypeAdapter(S).validate_python({"a": 1})
    assert calls == ["pre", "post"]  # pre==before (pre-construction), post==after


def test_mro_aggregation_before_then_after_base_then_child():
    calls = []

    class Base(GatewayStruct):
        a: int = 0

    class Child(Base):
        b: int = 0

    Base.add_validator(lambda d: (calls.append("pre-base"), d)[1], mode="before")
    Child.add_validator(lambda d: (calls.append("pre-child"), d)[1], mode="before")
    Base.add_validator(lambda s: (calls.append("post-base"), s)[1], mode="after")
    Child.add_validator(lambda s: (calls.append("post-child"), s)[1], mode="after")

    TypeAdapter(Child).validate_python({"a": 1, "b": 2})
    # before (base then child) run before construction; after (base then child) after.
    assert calls == ["pre-base", "pre-child", "post-base", "post-child"]


def test_child_validator_does_not_leak_to_base():
    class Base(GatewayStruct):
        a: int = 0

    class Child(Base):
        b: int = 0

    def _reject(s):
        raise ValueError("child-only")

    Child.add_validator(_reject)
    with pytest.raises(ValidationError, match="child-only"):
        TypeAdapter(Child).validate_python({"a": 0})
    assert TypeAdapter(Base).validate_python({"a": 0}).a == 0  # base unaffected


def test_isolation_between_sibling_classes():
    class A(GatewayStruct):
        x: int = 0

    class B(GatewayStruct):
        x: int = 0

    def _reject(s):
        raise ValueError("only-A")

    A.add_validator(_reject)
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
    S.add_validator(lambda s: (ran.append(1), s)[1])
    TypeAdapter(S).validate_python({"a": 1})
    assert ran, "registry must run even when the after hook is overridden without super()"


def test_validator_runs_exactly_once():
    class S(GatewayStruct):
        a: int = 0

    count = []
    S.add_validator(lambda s: (count.append(1), s)[1])
    TypeAdapter(S).validate_python({"a": 1})
    assert len(count) == 1


def test_after_validators_run_before_the_after_hook():
    # An after-validator may fix data that the after-hook then checks -> it must run first.
    class S(GatewayStruct):
        a: int = 0

        @classmethod
        def _validate_gateway_struct_after(cls, val):
            if val.a < 10:
                raise ValueError("hook: a<10")
            return val

    S.add_validator(lambda s: _set_a(s, s.a + 100), mode="after")
    assert TypeAdapter(S).validate_python({"a": 1}).a == 101  # bumped before the hook checked it


def test_after_hook_and_registered_validator_both_enforced():
    class S(GatewayStruct):
        a: int = 0
        b: int = 0

        @classmethod
        def _validate_gateway_struct_after(cls, val):
            if val.a < 0:
                raise ValueError("hook: a<0")
            return val

    def _reject_b(s):
        if s.b < 0:
            raise ValueError("validator: b<0")
        return s

    S.add_validator(_reject_b)
    with pytest.raises(ValidationError, match="validator: b<0"):
        TypeAdapter(S).validate_python({"a": 0, "b": -1})
    with pytest.raises(ValidationError, match="hook: a<0"):
        TypeAdapter(S).validate_python({"a": -1, "b": 0})
    assert TypeAdapter(S).validate_python({"a": 1, "b": 1}).a == 1


def test_multiple_after_validators_run_in_order_first_raise_wins():
    class S(GatewayStruct):
        a: int = 0

    order = []

    def _first(s):
        order.append("first")
        if s.a == 1:
            raise ValueError("stop")
        return s

    def _second(s):
        order.append("second")
        return s

    S.add_validator(_first)
    S.add_validator(_second)
    TypeAdapter(S).validate_python({"a": 0})
    assert order == ["first", "second"]
    order.clear()
    with pytest.raises(ValidationError, match="stop"):
        TypeAdapter(S).validate_python({"a": 1})
    assert order == ["first"]  # short-circuits on first raise


def test_after_validators_transform_then_reject():
    class S(GatewayStruct):
        a: int = 0

    def _require_ge_10(s):
        if s.a < 10:
            raise ValueError("too-small")
        return s

    # first bumps into range, second accepts
    S.add_validator(lambda s: _set_a(s, s.a + 100), mode="after")
    S.add_validator(_require_ge_10)
    assert TypeAdapter(S).validate_python({"a": 1}).a == 101

    # a transform that breaks the struct is caught by the reject validator
    S.clear_validators()
    S.add_validator(lambda s: _set_a(s, -1), mode="after")
    S.add_validator(_require_ge_10)
    with pytest.raises(ValidationError, match="too-small"):
        TypeAdapter(S).validate_python({"a": 500})


def test_no_validators_is_noop():
    class S(GatewayStruct):
        a: int = 0

    assert TypeAdapter(S).validate_python({"a": 7}).a == 7


def test_bare_decorator_registers_after_validator():
    class S(GatewayStruct):
        a: int = 0

    @S.add_validator
    def _v(s):
        return _bump(s, 1)

    assert _v is not None
    assert TypeAdapter(S).validate_python({"a": 1}).a == 2


def test_decorator_factory_with_mode():
    class S(GatewayStruct):
        a: int = 0

    @S.add_validator(mode="before")
    def _pre(d):
        return {**d, "a": d.get("a", 0) + 5}

    assert _pre is not None
    assert TypeAdapter(S).validate_python({"a": 1}).a == 6


def test_bound_method_validator_registered_by_another_object():
    # "some other object registers a validator and passes in a bound method that needs to be called"
    class Enricher:
        def __init__(self, bump):
            self.bump = bump

        def enrich(self, s):
            return _bump(s, self.bump)

    class S(GatewayStruct):
        a: int = 0

    S.add_validator(Enricher(bump=100).enrich, mode="after")  # bound method captured at registration
    assert TypeAdapter(S).validate_python({"a": 1}).a == 101


def test_manual_call_on_csp_path():
    # Native/CSP construction does NOT auto-run validators; they can be invoked explicitly.
    class S(GatewayStruct):
        a: int = 0

    def _reject_neg(s):
        if s.a < 0:
            raise ValueError("neg")
        return s

    S.add_validator(_reject_neg)
    assert S._run_validators(S(a=1)).a == 1
    with pytest.raises(ValueError, match="neg"):
        S._run_validators(S(a=-1))


class _VChild(GatewayStruct):
    v: int = 0


class _VParent(GatewayStruct):
    child: _VChild
    name: str = ""


def test_nested_struct_validator_rejects_on_parent_validation():
    def _reject(c):
        if c.v < 0:
            raise ValueError("child-bad")
        return c

    _VChild.add_validator(_reject)
    try:
        with pytest.raises(ValidationError, match="child-bad"):
            TypeAdapter(_VParent).validate_python({"name": "p", "child": {"v": -1}})
    finally:
        _VChild.clear_validators()


def test_nested_struct_after_validator_transforms_on_parent_validation():
    _VChild.add_validator(lambda c: _set_v(c, c.v * 10), mode="after")
    try:
        out = TypeAdapter(_VParent).validate_python({"name": "p", "child": {"v": 5}})
        assert out.child.v == 50
    finally:
        _VChild.clear_validators()


def test_valueerror_becomes_validationerror_other_exceptions_propagate():
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


def test_after_validator_returning_none_raises_clear_error():
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: None, mode="after")
    with pytest.raises(ValueError, match="returned None"):
        TypeAdapter(S).validate_python({"a": 1})


def test_before_validator_returning_none_raises_clear_error():
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda d: None, mode="before")
    with pytest.raises(ValueError, match="returned None"):
        TypeAdapter(S).validate_python({"a": 1})


def test_clear_validators_removes_all():
    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: _bump(s), mode="after")
    assert TypeAdapter(S).validate_python({"a": 1}).a == 11
    S.clear_validators()
    assert TypeAdapter(S).validate_python({"a": 1}).a == 1


def test_clear_validators_by_mode():
    calls = []

    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda d: (calls.append("pre"), d)[1], mode="before")
    S.add_validator(lambda s: (calls.append("post"), s)[1], mode="after")
    S.clear_validators(mode="before")
    calls.clear()
    TypeAdapter(S).validate_python({"a": 1})
    assert calls == ["post"]  # only the after validator remains


def test_clear_validators_by_mode_alias():
    calls = []

    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda d: (calls.append("pre"), d)[1], mode="before")
    S.add_validator(lambda s: (calls.append("post"), s)[1], mode="after")
    S.clear_validators(mode="post")  # alias for "after"
    calls.clear()
    TypeAdapter(S).validate_python({"a": 1})
    assert calls == ["pre"]  # only the before validator remains


def test_clear_validators_removes_own_but_not_inherited_or_sibling():
    # The refactor's central invariant: clearing one class must NOT remove inherited (parent) or
    # sibling registrations -- only the class's own.
    class Base(GatewayStruct):
        a: int = 0

    class Child(Base):
        b: int = 0

    class Sibling(GatewayStruct):
        a: int = 0

    def _base_reject(s):
        if s.a < 0:
            raise ValueError("base-reject")
        return s

    def _child_reject(s):
        if getattr(s, "b", 0) < 0:
            raise ValueError("child-reject")
        return s

    def _sibling_reject(s):
        if s.a < 0:
            raise ValueError("sibling-reject")
        return s

    Base.add_validator(_base_reject)
    Child.add_validator(_child_reject)
    Sibling.add_validator(_sibling_reject)

    # Child initially enforces its own rule (b<0).
    with pytest.raises(ValidationError, match="child-reject"):
        TypeAdapter(Child).validate_python({"a": 0, "b": -1})

    Child.clear_validators()

    # Child's OWN validator is gone: b<0 no longer rejected.
    assert TypeAdapter(Child).validate_python({"a": 0, "b": -1}).b == -1
    # The INHERITED base validator still runs on Child.
    with pytest.raises(ValidationError, match="base-reject"):
        TypeAdapter(Child).validate_python({"a": -1, "b": 0})
    # A SIBLING class is unaffected by clearing Child.
    with pytest.raises(ValidationError, match="sibling-reject"):
        TypeAdapter(Sibling).validate_python({"a": -1})


def test_add_validator_rejects_invalid_mode():
    class S(GatewayStruct):
        a: int = 0

    with pytest.raises(ValueError, match="mode must be"):
        S.add_validator(lambda s: s, mode="sideways")


def test_add_validator_rejects_noncallable():
    class S(GatewayStruct):
        a: int = 0

    with pytest.raises(TypeError, match="must be callable"):
        S.add_validator("not-callable")
    with pytest.raises(TypeError, match="must be callable"):
        S.add_validator("not-callable", mode="before")
