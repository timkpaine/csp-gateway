from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic

from ccflow import BaseModel
from pydantic import Field, TypeAdapter, model_validator

from csp_gateway.server.shared import ChannelSelection
from csp_gateway.utils import GatewayStruct

from .channels import ChannelsType

if TYPE_CHECKING:
    from csp_gateway.server import GatewaySettings, GatewayWebApp


class Module(BaseModel, Generic[ChannelsType], ABC):
    model_config = {"arbitrary_types_allowed": True}

    requires: ChannelSelection | None = None
    disable: bool = False
    block_set_channels_until: datetime | None = Field(
        default=None,
        description="""
        This determines the csp time at which this module can start sending data to channels.
        This value overrides any gateway-level blocks imposed.
        """,
    )

    @abstractmethod
    def connect(self, Channels: ChannelsType) -> None: ...

    def rest(self, app: "GatewayWebApp") -> None: ...

    def info(self, settings: "GatewaySettings") -> str | None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    def dynamic_keys(self) -> dict[str, list[Any]] | None: ...

    def dynamic_channels(self) -> dict[str, type[GatewayStruct] | type[list[GatewayStruct]]] | None:
        """
        Channels that this module dynamically adds to the gateway channels when this module is included into the gateway.

        Returns:
            Dictionary keyed by channel name and type of the timeseries of the channel as values.
        """

    def dynamic_state_channels(self) -> set[str] | None:
        """
        The set of dynamic channels that have state.
        """

    # @abc.abstractmethod
    # def subscribe(self):
    #     ...

    def __eq__(self, other):
        # Override equality because occasionally, Modules will contain fields with non-standard equality methods
        # i.e. numpy arrays or csp edges.
        # Without overriding, these types will prevent the modules from being compared with each other
        # which is needed for the dependency resolutions
        return id(self) == id(other)

    # See https://docs.pydantic.dev/latest/concepts/validators/#validation-of-default-values
    @model_validator(mode="before")
    def validate_requires(cls, v):
        requires = v.get("requires", cls.model_fields["requires"].default)
        v["requires"] = TypeAdapter(ChannelSelection).validate_python(requires)
        return v
