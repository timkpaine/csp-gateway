from datetime import datetime, timezone
from typing import Any, TypeVar

import csp
from csp import Struct
from pydantic import ValidationInfo
from pydantic_core import CoreConfig, core_schema

from ..id_generator import get_counter
from .psp import PerspectiveUtilityMixin

IdType = str

__all__ = (
    "GatewayLookupMixin",
    "GatewayPydanticMixin",
    "GatewayStruct",
    "GatewayStructMixins",
    "IdType",
    "global_lookup",
    "is_gateway_struct_like",
)

T = TypeVar("T")

# Global registry: maps ID -> instance for all GatewayLookupMixin instances
_global_registry: dict[str, Any] = {}

# Class-specific registry: maps (class, ID) -> instance
_class_registry: dict[tuple, Any] = {}


def global_lookup(id: IdType, cls: type[T] | None = None) -> T | None:
    """Look up a GatewayStruct instance by ID.

    Args:
        id: The unique ID of the instance to look up.
        cls: Optional class to filter by. If provided, only returns
             instances of that specific class.

    Returns:
        The instance if found, None otherwise.
    """
    if cls is not None:
        return _class_registry.get((cls, id))
    return _global_registry.get(id)


class GatewayLookupMixin:
    # Shared global ID generator
    id_generator = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Use the single global generator for all classes
        if GatewayLookupMixin.id_generator is None:
            GatewayLookupMixin.id_generator = get_counter()
        cls.id_generator = GatewayLookupMixin.id_generator
        cls._include_in_lookup = True

    def __init__(self, **kwargs: Any) -> None:
        if "id" not in kwargs:
            kwargs["id"] = str(self.__class__.id_generator.next())
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now(timezone.utc)
        if getattr(self.__class__, "_include_in_lookup", True):
            # Insert into both global and class-specific registries
            _global_registry[kwargs["id"]] = self
            _class_registry[(self.__class__, kwargs["id"])] = self
        super().__init__(**kwargs)

    @classmethod
    def omit_from_lookup(cls, omit=True):
        cls._include_in_lookup = not omit

    @classmethod
    def included_in_lookup(cls):
        return cls._include_in_lookup

    @classmethod
    def generate_id(cls) -> str:
        return str(cls.id_generator.next())

    @classmethod
    def lookup(cls, id: IdType) -> Any | None:
        """Look up an instance by ID, scoped to this class.

        Args:
            id: The unique ID of the instance to look up.

        Returns:
            The instance if found in this class's registry, None otherwise.
        """
        return _class_registry.get((cls, id))


class GatewayPydanticMixin:
    # Class-level validator registries. Validators run automatically during pydantic validation (the wrap
    # validator ``_validate_gateway_struct`` invokes them) and are aggregated across the MRO, so base-class
    # and subclass validators both execute. They hold ARBITRARY callables (lambdas, bound methods) that
    # pydantic never sees directly. A validator ``fn(value) -> value`` receives the value and returns the
    # (possibly transformed) value; it RAISES to reject the input (surfaced by the REST API as a 422).
    # "before" (alias "pre") validators run on the raw input (typically a dict) prior to struct
    # construction -- useful for accepting legacy/aliased shapes; "after" (alias "post") validators run on
    # the constructed struct. Plain class attributes, not csp.Struct fields (no annotation).
    _pre_validators = []
    _post_validators = []

    # Accepted ``mode`` values -> internal registry attribute (both "before"/"pre" and "after"/"post").
    _VALIDATOR_MODE_ATTRS = {
        "before": "_pre_validators",
        "pre": "_pre_validators",
        "after": "_post_validators",
        "post": "_post_validators",
    }

    @classmethod
    def _validator_mode_attr(cls, mode):
        """Resolve a public ``mode`` ("before"/"pre"/"after"/"post") to its registry attribute name."""
        try:
            return cls._VALIDATOR_MODE_ATTRS[mode]
        except (KeyError, TypeError):
            raise ValueError(f"mode must be 'before'/'pre' or 'after'/'post'; got {mode!r}")

    @classmethod
    def add_validator(cls, fn=None, *, mode="after"):
        """Register a validator invoked during pydantic validation.

        A validator ``fn(value) -> value`` receives the value and returns the (possibly transformed)
        value; it **raises** (e.g. ``ValueError``) to reject the input -- surfaced by the REST API as a
        422. Returning ``None`` is treated as an error (a validator must return the value).

        ``mode="before"`` (alias ``"pre"``) runs on the raw input (usually a dict) before construction --
        useful for accepting legacy/aliased input shapes; ``mode="after"`` (alias ``"post"``, the default)
        runs on the constructed struct. ``fn`` may be any callable -- a lambda, or a bound method of some
        other object (e.g. an adapter holding a secmaster). Validators registered on base classes also run
        (aggregated across the MRO). Usable directly, as a bare decorator (``@Struct.add_validator``), or
        as a decorator factory (``@Struct.add_validator(mode="before")``). Returns ``fn``.
        """
        attr = cls._validator_mode_attr(mode)

        def _register(func):
            if not callable(func):
                raise TypeError(f"validator must be callable; got {func!r}")
            # Ensure THIS class has its OWN list rather than mutating an inherited (shared) one.
            if attr not in cls.__dict__:
                setattr(cls, attr, [])
            getattr(cls, attr).append(func)
            return func

        # Support ``add_validator(fn, ...)``/bare-decorator and the ``add_validator(mode=...)`` factory.
        if fn is None:
            return _register
        return _register(fn)

    @classmethod
    def _collect_validators(cls, attr):
        """Aggregate validators of one kind (``_pre_validators``/``_post_validators``) across the MRO."""
        collected = []
        for klass in reversed(cls.__mro__):
            collected.extend(klass.__dict__.get(attr, ()))
        return collected

    @classmethod
    def _run_validators(cls, val, *, mode="after"):
        """Run the registered validators of one kind on ``val``; return the (possibly transformed) value.

        Each validator must return the value (raising to reject); a ``None`` return is an error. Useful
        for manually validating a natively/CSP-constructed struct, which does not auto-run validators.
        """
        attr = cls._validator_mode_attr(mode)
        label = "before" if attr == "_pre_validators" else "after"
        for fn in cls._collect_validators(attr):
            val = fn(val)
            if val is None:
                raise ValueError(
                    f"{cls.__name__}: {label!r} validator {getattr(fn, '__name__', fn)!r} returned None; validators must return the (possibly transformed) value"
                )
        return val

    @classmethod
    def clear_validators(cls, *, mode=None):
        """Remove validators registered directly on THIS class (not inherited ones).

        ``mode=None`` clears both "before" and "after"; ``"before"``/``"pre"`` or ``"after"``/``"post"``
        clears just that kind. Useful for teardown/idempotency when validators are attached dynamically at
        gateway-build time.
        """
        if mode is None:
            cls._pre_validators = []
            cls._post_validators = []
        else:
            setattr(cls, cls._validator_mode_attr(mode), [])

    @classmethod
    def _validate_gateway_struct_after(cls, val):
        """Validate GatewayStruct after pydantic type validation.
        A validator attached to every GatewayStruct to allow for defining custom
        model-level after validators that run after pydantic type validation.
        If not defined on a child class, the parent's validator will be used.  If defined on a child class, the parent's validator will be ignored. Please call the parent's validator directly if you want to run both.

        This is meant to be mixed-in with csp.Struct's. We do not inherit from a csp.Struct
        since csp.Struct's do not support multiple inheritance with other csp.Struct's

        Args:
            cls: The class this validator is attached to
            val: The value to validate
        Returns:
            The validated value, possibly modified
        """
        return val

    @classmethod
    def type_adapter(cls):
        # NOTE: Only needed until csp>0.9 is released with this fix
        # We mangle ourselves, explicitly, to make sure that child Structs
        # will get their own type adapters.
        attr_name = f"_{cls.__name__}__pydantic_type_adapter"
        internal_type_adapter = getattr(cls, attr_name, None)
        if internal_type_adapter:
            return internal_type_adapter

        # Late import to avoid autogen issues
        from pydantic import TypeAdapter

        type_adapter = TypeAdapter(cls)
        setattr(cls, attr_name, type_adapter)
        return type_adapter

    @classmethod
    def _validate_gateway_struct(cls, val, handler, info: ValidationInfo):
        if isinstance(info.context, dict) and isinstance(val, dict):
            if info.context.get("force_new_id", False):
                # If we are forcing a new id, we need to remove the old one
                val.pop("id", None)
            if info.context.get("force_new_timestamp", False):
                # If we are forcing a new timestamp, we need to remove the old one
                val.pop("timestamp", None)
        # "before" validators reshape the raw input (e.g. legacy-field coercion) prior to construction.
        val = cls._run_validators(val, mode="before")
        csp_struct = handler(val)
        # "after" validators validate/normalize the constructed struct. They run in the wrap validator
        # (the robust funnel) rather than the after-hook: subclasses commonly OVERRIDE
        # _validate_gateway_struct_after WITHOUT super(), which would bypass the registry, whereas the wrap
        # validator is always inherited or overridden-with-super(). Registered validators (aggregated
        # across the MRO) therefore run for every struct, and BEFORE the after-hook so they may fix data
        # the hook then checks.
        csp_struct = cls._run_validators(csp_struct, mode="after")
        final = cls._validate_gateway_struct_after(csp_struct)
        return final

    @staticmethod
    def _get_pydantic_core_schema(struct_cls, source_type, handler):
        # Get parent schema - note the struct_cls parameter
        parent_schema = csp.Struct._get_pydantic_core_schema(struct_cls, source_type, handler)
        core_config = CoreConfig(coerce_numbers_to_str=True)
        # soooo hacky...
        parent_schema["schema"]["config"] = core_config
        return core_schema.with_info_wrap_validator_function(
            function=struct_cls._validate_gateway_struct, schema=parent_schema, serialization=parent_schema.get("serialization")
        )


GatewayStructMixins = (GatewayLookupMixin, GatewayPydanticMixin, PerspectiveUtilityMixin)


class GatewayStruct(
    *GatewayStructMixins,
    Struct,
):
    """Convenience class composing gateway mixins with csp.Struct.

    Provides id/timestamp fields, lookup/registry utilities, and pydantic
    integration, plus Perspective utilities.
    """

    id: IdType
    timestamp: datetime


def is_gateway_struct_like(cls) -> bool:
    """Strict check: requires all gateway mixins and `csp.Struct`.

    Returns True only if `cls` is a `csp.Struct` subclass AND also
    subclasses `GatewayLookupMixin`, `GatewayPydanticMixin`, and
    `PerspectiveUtilityMixin`.
    """
    if not isinstance(cls, type):
        return False
    # Shortcut for explicit GatewayStruct
    if issubclass(cls, GatewayStruct):
        return True
    try:
        return (
            issubclass(cls, Struct)
            and issubclass(cls, GatewayLookupMixin)
            and issubclass(cls, GatewayPydanticMixin)
            and issubclass(cls, PerspectiveUtilityMixin)
        )
    except TypeError:
        return False
