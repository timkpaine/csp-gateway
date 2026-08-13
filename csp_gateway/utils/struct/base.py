from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar

import csp
from csp import Struct
from pydantic import ValidationInfo
from pydantic_core import CoreConfig, core_schema

from ..id_generator import get_counter
from .psp import PerspectiveUtilityMixin

IdType = str

#: A validator receives a value and returns the (possibly transformed) value, or raises to reject it.
ValidatorFn = Callable[[Any], Any]

#: Accepted ``mode`` arguments; "pre"/"post" are aliases for "before"/"after".
ValidatorMode = Literal["before", "pre", "after", "post"]

__all__ = (
    "GatewayLookupMixin",
    "GatewayPydanticMixin",
    "GatewayStruct",
    "GatewayStructMixins",
    "IdType",
    "ValidatorFn",
    "ValidatorMode",
    "global_lookup",
    "is_gateway_struct_like",
)

T = TypeVar("T")

# Global registry: maps ID -> instance for all GatewayLookupMixin instances
_global_registry: dict[str, Any] = {}

# Class-specific registry: maps (class, ID) -> instance
_class_registry: dict[tuple, Any] = {}

# Bumped on every validator registration/clear so per-class resolved caches invalidate without having
# to know which classes in which MRO were affected.
_validator_registry_version = 0


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
    # Validators registered through add_validator, invoked by the _validate_gateway_struct wrap
    # validator. Deliberately unannotated so csp does not treat them as Struct fields.
    _pre_validators = []
    _post_validators = []

    # Accepted ``mode`` values -> internal registry attribute (both "before"/"pre" and "after"/"post").
    _VALIDATOR_MODE_ATTRS = {
        "before": "_pre_validators",
        "pre": "_pre_validators",
        "after": "_post_validators",
        "post": "_post_validators",
    }

    # Registry attribute -> attribute holding this class's MRO-resolved (version, validators) cache.
    _VALIDATOR_CACHE_ATTRS = {
        "_pre_validators": "_pre_validators_resolved",
        "_post_validators": "_post_validators_resolved",
    }

    @classmethod
    def _validator_mode_attr(cls, mode: ValidatorMode) -> str:
        """Resolve a public ``mode`` ("before"/"pre"/"after"/"post") to its registry attribute name."""
        try:
            return cls._VALIDATOR_MODE_ATTRS[mode]
        except (KeyError, TypeError):
            raise ValueError(f"mode must be 'before'/'pre' or 'after'/'post'; got {mode!r}") from None

    @classmethod
    def add_validator(cls, fn: ValidatorFn | None = None, *, mode: ValidatorMode = "after") -> ValidatorFn | Callable[[ValidatorFn], ValidatorFn]:
        """Register a validator invoked during pydantic validation.

        A validator ``fn(value) -> value`` receives the value and returns the (possibly transformed)
        value; it rejects the input by raising ``ValueError`` (or ``AssertionError``), which pydantic
        converts into a ``ValidationError`` and the REST API reports as a 422. Any OTHER exception type
        propagates uncaught and reaches the client as a 500, so a validator that rejects untrusted input
        must raise ``ValueError`` -- guard lookups rather than letting a ``KeyError`` escape.

        An "after" validator must return an instance of the class being validated; returning another type
        raises ``TypeError``. Returning ``None`` is likewise an error unless the incoming value was itself
        ``None``.

        ``mode="before"`` (alias ``"pre"``) runs on the raw input (usually a dict) before construction --
        useful for accepting legacy/aliased input shapes. It is skipped when the input is already a
        constructed struct, so a "before" validator only ever sees unconstructed input. It must not set
        ``id`` or ``timestamp`` when the caller asked for fresh ones; those are re-scrubbed afterwards.
        ``mode="after"`` (alias ``"post"``, the default) runs on the constructed struct, dispatched on
        that struct's concrete type. ``fn`` may be any callable -- a lambda, or a bound method of some
        other object (e.g. an adapter holding a secmaster). Validators registered on base classes also
        run (aggregated across the MRO); registering on ``GatewayStruct`` or ``GatewayPydanticMixin``
        therefore instruments every struct in the process. Usable directly, as a bare decorator
        (``@Struct.add_validator``), or as a decorator factory (``@Struct.add_validator(mode="before")``).
        Returns ``fn``.
        """
        attr = cls._validator_mode_attr(mode)

        def _register(func: ValidatorFn) -> ValidatorFn:
            if not callable(func):
                raise TypeError(f"validator must be callable; got {func!r}")
            # Give this class its own list rather than mutating an inherited one.
            if attr not in cls.__dict__:
                setattr(cls, attr, [])
            getattr(cls, attr).append(func)
            global _validator_registry_version
            _validator_registry_version += 1
            return func

        if fn is None:
            return _register
        return _register(fn)

    @classmethod
    def _collect_validators(cls, attr: str) -> tuple[ValidatorFn, ...]:
        """Aggregate validators of one kind (``_pre_validators``/``_post_validators``) across the MRO.

        The result is cached on the class, keyed by the global registry version, so any registration or
        clear anywhere invalidates it. The version is sampled BEFORE the walk: a registration racing the
        walk then leaves the cache stamped older than the global and the next read recomputes.
        """
        cache_attr = cls._VALIDATOR_CACHE_ATTRS[attr]
        version = _validator_registry_version
        cached = cls.__dict__.get(cache_attr)
        if cached is not None and cached[0] == version:
            return cached[1]
        collected = []
        for klass in reversed(cls.__mro__):
            collected.extend(klass.__dict__.get(attr, ()))
        resolved = tuple(collected)
        setattr(cls, cache_attr, (version, resolved))
        return resolved

    @classmethod
    def run_validators(cls, val: Any, *, mode: ValidatorMode = "after") -> Any:
        """Run the registered validators of one kind on ``val``; return the (possibly transformed) value.

        Each validator must return the value, raising ``ValueError`` to reject it. Returning ``None`` is
        rejected as a forgotten ``return`` -- unless the incoming value was itself ``None``, in which case
        the value passes through and the type error surfaces normally. An "after" validator must return an
        instance of ``cls``. Useful for manually validating a natively/CSP-constructed struct, which does
        not auto-run validators.
        """
        attr = cls._validator_mode_attr(mode)
        enforce_type = attr == "_post_validators"
        label = "after" if enforce_type else "before"
        for fn in cls._collect_validators(attr):
            result = fn(val)
            name = getattr(fn, "__name__", fn)
            if result is None and val is not None:
                raise ValueError(
                    f"{cls.__name__}: {label!r} validator {name!r} returned None; validators must return the (possibly transformed) value"
                )
            if enforce_type and not isinstance(result, cls):
                raise TypeError(f"{cls.__name__}: 'after' validator {name!r} returned {type(result).__name__}; it must return a {cls.__name__}")
            val = result
        return val

    @classmethod
    def clear_validators(cls, *, mode: ValidatorMode | None = None) -> None:
        """Remove validators registered directly on this class, leaving inherited ones in place.

        ``mode=None`` clears both "before" and "after"; ``"before"``/``"pre"`` or ``"after"``/``"post"``
        clears just that kind. Useful for teardown/idempotency when validators are attached dynamically at
        gateway-build time.
        """
        if mode is None:
            cls._pre_validators = []
            cls._post_validators = []
        else:
            setattr(cls, cls._validator_mode_attr(mode), [])
        global _validator_registry_version
        _validator_registry_version += 1

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
    def _scrub_identity(cls, val, info: ValidationInfo):
        """Drop a caller-supplied id/timestamp when the validation context asks for fresh ones."""
        if not isinstance(info.context, dict):
            return val
        new_id = info.context.get("force_new_id", False)
        new_timestamp = info.context.get("force_new_timestamp", False)
        if not (new_id or new_timestamp):
            return val
        if isinstance(val, dict):
            if new_id:
                val.pop("id", None)
            if new_timestamp:
                val.pop("timestamp", None)
        elif isinstance(val, Struct):
            # Scrubbing in place would mutate the caller's object and strand it in the lookup registry
            # under its old id, so rebuild instead and let __init__ mint and register the new values.
            fields = {name: getattr(val, name) for name in type(val).metadata() if hasattr(val, name)}
            if new_id:
                fields.pop("id", None)
            if new_timestamp:
                fields.pop("timestamp", None)
            return type(val)(**fields)
        return val

    @classmethod
    def _validate_gateway_struct(cls, val, handler, info: ValidationInfo):
        val = cls._scrub_identity(val, info)
        # An already-constructed struct has no raw input to reshape, so "before" validators are skipped.
        if not isinstance(val, cls):
            val = cls.run_validators(val, mode="before")
            # Re-scrub: a before validator that rebuilds the input can otherwise reinstate the old id.
            val = cls._scrub_identity(val, info)
        csp_struct = handler(val)
        # Dispatch on the concrete type: a subclass instance in a base-annotated field is validated
        # against the base's schema, but must still run its own validators and hook.
        concrete = type(csp_struct)
        csp_struct = concrete.run_validators(csp_struct, mode="after")
        final = concrete._validate_gateway_struct_after(csp_struct)
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
