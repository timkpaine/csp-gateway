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
    # Class-level validator registry. Each subclass gets its OWN list via add_validator
    # (see below); validators are aggregated across the MRO when validation runs, so base-class
    # validators and subclass validators both execute. Not a csp.Struct field (no annotation).
    _validators = []

    @classmethod
    def add_validator(cls, fn):
        """Register a validator ``fn(struct) -> Optional[str]`` on this struct class.

        The callable receives a constructed instance and returns an error message string if the
        instance is invalid, or ``None`` if it is valid. Lambdas and bound methods are both fine.
        Validators registered on base classes also run (aggregated across the MRO). Returns ``fn`` so
        it can be used as a decorator.
        """
        # Ensure THIS class has its own list rather than mutating an inherited (shared) one.
        if not callable(fn):
            raise TypeError(f"validator must be callable; got {fn!r}")
        if "_validators" not in cls.__dict__:
            cls._validators = []
        cls._validators.append(fn)
        return fn

    @classmethod
    def _collect_validators(cls):
        """Aggregate validators across the MRO (base-class first)."""
        collected = []
        for klass in reversed(cls.__mro__):
            collected.extend(klass.__dict__.get("_validators", ()))
        return collected

    @classmethod
    def _run_validators(cls, val):
        """Run all registered validators; raise ``ValueError`` on the first failure."""
        for fn in cls._collect_validators():
            error = fn(val)
            if error:
                raise ValueError(error)
        return val

    # Class-level transformer registries. Like validators, these are aggregated across the MRO and hold
    # ARBITRARY callables (lambdas, bound methods) registered dynamically -- pydantic never sees them
    # directly; the wrap validator (_validate_gateway_struct) invokes them. "before" transformers run on
    # the raw input (typically a dict) prior to struct construction; "after" transformers run on the
    # constructed struct. Both may mutate/replace and MUST return the (possibly new) value.
    _pre_transformers = []
    _post_transformers = []

    @classmethod
    def add_transformer(cls, fn, *, mode="after"):
        """Register a transformer ``fn(value) -> value`` invoked during pydantic validation.

        ``mode="before"``: runs on the raw input (usually a dict) before construction.
        ``mode="after"`` (default): runs on the constructed struct.
        ``fn`` may be any callable -- a lambda, a bound method of some other object (e.g. an adapter
        holding a secmaster), etc. It must return the transformed value. Returns ``fn`` for decorator use.
        """
        if mode not in ("before", "after"):
            raise ValueError(f"mode must be 'before' or 'after'; got {mode!r}")
        if not callable(fn):
            raise TypeError(f"transformer must be callable; got {fn!r}")
        attr = "_pre_transformers" if mode == "before" else "_post_transformers"
        if attr not in cls.__dict__:
            setattr(cls, attr, [])
        getattr(cls, attr).append(fn)
        return fn

    @classmethod
    def _collect_transformers(cls, attr):
        """Aggregate transformers of one kind across the MRO (base-class first)."""
        collected = []
        for klass in reversed(cls.__mro__):
            collected.extend(klass.__dict__.get(attr, ()))
        return collected

    @classmethod
    def clear_transformers(cls, *, mode=None):
        """Remove transformers registered directly on THIS class (not inherited ones).

        ``mode=None`` clears both "before" and "after"; ``"before"``/``"after"`` clears one kind. Useful
        for teardown/idempotency when transformers are registered dynamically at gateway-build time.
        """
        if mode in (None, "before"):
            cls._pre_transformers = []
        if mode in (None, "after"):
            cls._post_transformers = []

    @classmethod
    def clear_validators(cls):
        """Remove validators registered directly on THIS class (not inherited ones)."""
        cls._validators = []

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
        # "before" transformers reshape the raw input (e.g. legacy-field coercion) prior to construction.
        for fn in cls._collect_transformers("_pre_transformers"):
            val = fn(val)
            if val is None:
                raise ValueError(
                    f"{cls.__name__}: 'before' transformer {getattr(fn, '__name__', fn)!r} returned None; transformers must return the (possibly reshaped) input"
                )
        csp_struct = handler(val)
        # "after" transformers mutate/enrich the constructed struct before it is validated.
        for fn in cls._collect_transformers("_post_transformers"):
            csp_struct = fn(csp_struct)
            if csp_struct is None:
                raise ValueError(
                    f"{cls.__name__}: 'after' transformer {getattr(fn, '__name__', fn)!r} returned None; transformers must return the (possibly new) struct"
                )
        final = cls._validate_gateway_struct_after(csp_struct)
        # Run the validator registry here (the wrap validator is the robust funnel): subclasses commonly
        # OVERRIDE _validate_gateway_struct_after without calling super(), which would bypass the registry.
        # _validate_gateway_struct, by contrast, is either inherited or overridden-with-super(), so this
        # runs for every struct (registered validators aggregate across the MRO in _run_validators).
        cls._run_validators(final)
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
