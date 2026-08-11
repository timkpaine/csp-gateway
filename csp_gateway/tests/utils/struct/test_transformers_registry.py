import pytest
from pydantic import TypeAdapter

from csp_gateway.utils.struct import GatewayStruct


def test_post_transformer_lambda_mutates_during_validation():
    class S(GatewayStruct):
        a: int = 0

    def _post(s):
        s2 = s.copy()
        s2.a = s.a * 10
        return s2

    S.add_transformer(_post, mode="after")
    out = TypeAdapter(S).validate_python({"a": 5})
    assert out.a == 50  # transformer ran during pydantic validation


def test_pre_transformer_reshapes_raw_input():
    class S(GatewayStruct):
        a: int = 0

    # A "before" transformer that accepts a legacy key `old_a` and folds it into `a` (like coercion).
    S.add_transformer(lambda d: {**{k: v for k, v in d.items() if k != "old_a"}, "a": d["old_a"]} if "old_a" in d else d, mode="before")
    out = TypeAdapter(S).validate_python({"old_a": 7})
    assert out.a == 7


def test_bound_method_transformer_registered_by_another_object():
    # "some other method adds a transformer and passes in a bound method that needs to be called"
    class Enricher:
        def __init__(self, bump):
            self.bump = bump

        def enrich(self, s):
            s2 = s.copy()
            s2.a = s.a + self.bump
            return s2

    class S(GatewayStruct):
        a: int = 0

    enricher = Enricher(bump=100)
    S.add_transformer(enricher.enrich, mode="after")  # bound method captured at registration
    out = TypeAdapter(S).validate_python({"a": 1})
    assert out.a == 101


def test_mro_aggregation_and_order_pre_then_post():
    calls = []

    class Base(GatewayStruct):
        a: int = 0

    class Child(Base):
        b: int = 0

    Base.add_transformer(lambda d: (calls.append("pre-base"), d)[1], mode="before")
    Child.add_transformer(lambda d: (calls.append("pre-child"), d)[1], mode="before")
    Base.add_transformer(lambda s: (calls.append("post-base"), s)[1], mode="after")
    Child.add_transformer(lambda s: (calls.append("post-child"), s)[1], mode="after")

    TypeAdapter(Child).validate_python({"a": 1, "b": 2})
    # pre (base then child) run before construction; post (base then child) after.
    assert calls == ["pre-base", "pre-child", "post-base", "post-child"]


def test_no_transformers_is_noop():
    class S(GatewayStruct):
        a: int = 0

    assert TypeAdapter(S).validate_python({"a": 3}).a == 3


def test_clear_transformers_removes_registrations():
    class S(GatewayStruct):
        a: int = 0

    S.add_transformer(lambda s: _bump(s), mode="after")
    assert TypeAdapter(S).validate_python({"a": 1}).a == 11  # transformer active
    S.clear_transformers()
    assert TypeAdapter(S).validate_python({"a": 1}).a == 1  # cleared -> no-op


def test_clear_transformers_by_mode():
    calls = []

    class S(GatewayStruct):
        a: int = 0

    S.add_transformer(lambda d: (calls.append("pre"), d)[1], mode="before")
    S.add_transformer(lambda s: (calls.append("post"), s)[1], mode="after")
    S.clear_transformers(mode="before")
    calls.clear()
    TypeAdapter(S).validate_python({"a": 1})
    assert calls == ["post"]  # only post remains


def test_clear_validators_removes_registrations():
    from pydantic import ValidationError

    class S(GatewayStruct):
        a: int = 0

    S.add_validator(lambda s: "bad" if s.a == 1 else None)
    with pytest.raises(ValidationError, match="bad"):
        TypeAdapter(S).validate_python({"a": 1})
    S.clear_validators()
    assert TypeAdapter(S).validate_python({"a": 1}).a == 1  # cleared


def _bump(s):
    s2 = s.copy()
    s2.a = s.a + 10
    return s2


class _Child(GatewayStruct):
    v: int = 0


class _Parent(GatewayStruct):
    child: _Child
    name: str = ""


def test_nested_struct_transformer_fires_on_parent_validation():
    # Registering on the nested struct type fires when a containing struct is validated.
    _Child.add_transformer(lambda c: _set_v(c, c.v * 10), mode="after")
    try:
        out = TypeAdapter(_Parent).validate_python({"name": "p", "child": {"v": 5}})
        assert out.child.v == 50
    finally:
        _Child.clear_transformers()


def test_post_transformer_returning_none_raises_clear_error():
    class S(GatewayStruct):
        a: int = 0

    S.add_transformer(lambda s: None, mode="after")
    with pytest.raises(ValueError, match="returned None"):
        TypeAdapter(S).validate_python({"a": 1})


def test_pre_transformer_returning_none_raises_clear_error():
    class S(GatewayStruct):
        a: int = 0

    S.add_transformer(lambda d: None, mode="before")
    with pytest.raises(ValueError, match="returned None"):
        TypeAdapter(S).validate_python({"a": 1})


def test_transform_then_validate_transformer_can_fix_invalid_struct():
    from pydantic import ValidationError

    class S(GatewayStruct):
        a: int = 0

    # validator requires a >= 10; transformer bumps a into range -> validation passes.
    S.add_validator(lambda s: "too-small" if s.a < 10 else None)
    S.add_transformer(lambda s: _set_a(s, s.a + 100), mode="after")
    out = TypeAdapter(S).validate_python({"a": 1})
    assert out.a == 101  # transformed before validated

    # And a transformer that breaks the struct is caught by the validator.
    S.clear_transformers()
    S.add_transformer(lambda s: _set_a(s, -1), mode="after")
    with pytest.raises(ValidationError, match="too-small"):
        TypeAdapter(S).validate_python({"a": 500})


def test_add_transformer_usable_as_decorator():
    class S(GatewayStruct):
        a: int = 0

    @S.add_transformer
    def _t(s):
        return _set_a(s, s.a + 1)

    assert _t is not None
    assert TypeAdapter(S).validate_python({"a": 1}).a == 2


def _set_v(c, v):
    c2 = c.copy()
    c2.v = v
    return c2


def _set_a(s, a):
    s2 = s.copy()
    s2.a = a
    return s2


def test_add_transformer_rejects_invalid_mode_and_noncallable():
    class S(GatewayStruct):
        a: int = 0

    with pytest.raises(ValueError, match="mode must be"):
        S.add_transformer(lambda s: s, mode="pre")  # typo for 'before'
    with pytest.raises(TypeError, match="must be callable"):
        S.add_transformer("not-callable", mode="after")


def test_add_validator_rejects_noncallable():
    class S(GatewayStruct):
        a: int = 0

    with pytest.raises(TypeError, match="must be callable"):
        S.add_validator("not-callable")
