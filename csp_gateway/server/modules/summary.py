import json
import logging
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, TypeVar, Tuple, Optional

import csp
from csp import ts
from pydantic import BaseModel, Field

from csp_gateway import ChannelSelection, ChannelsType, GatewayModule, GatewayStruct
from csp_gateway import GatewayChannels, GatewayModule

logger = logging.getLogger(__name__)


T = TypeVar("T")


class Calculation(str, Enum):
    # Categorical/Numerical
    COUNT = "count"
    # UNIQUE = "unique"  # TODO
    # FIRST = "first"  # TODO
    # LAST = "last"  # TODO

    # Categorical only
    # MODE = "mode"  # TODO
    # MEDIAN = "median"  # TODO

    # Numerical only
    # MIN = "min"  # TODO
    # MAX = "max"  # TODO
    # MEAN = "mean"  # TODO
    SUM = "sum"
    # STD = "std"  # TODO
    # VAR = "var"  # TODO


class SummaryStruct(GatewayStruct):
    channel_name: str
    attribute: Optional[str]
    calculation: Calculation
    value: Any
    timestamp: datetime

class Summary(GatewayModule):
    """
    The Summary module computes some basic summary statistics on channels.
    It is good for producing "big number" grids using the `perpsective-summary` plugin in the UI
    """
    selection: ChannelSelection = Field(default_factory=ChannelSelection)
    calculations: Dict[Tuple[str, Optional[str]], List[Calculation]] = Field(
        description=(
            "A dictionary mapping channel names and optional channel struct attributes to a list of calculations to perform. "
            "If the channel struct attribute is None, the calculations are performed on all attributes on the struct",
        ),
        default_factory=dict,
    )

    def connect(self, channels: GatewayChannels):
        channels_to_calculate = []

        for field in self.selection.select_from(channels):
            channels_to_calculate.append(channels.get_channel(field))

    @csp.node
    def compute_summary(
        self,
        channel_data: List[GatewayStruct],
    ) -> ts[SummaryStruct]:
        if csp.tickeD(channel_data):
            return ts.empty(SummaryStruct)