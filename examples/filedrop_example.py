from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import csp

from csp_gateway import (
    AddChannelsToGraphOutput,
    FileDropType,
    Gateway,
    GatewayChannels,
    GatewayModule,
    GatewayStruct,
    ReadFileDrop,
    ReadFileDropConfiguration,
)


class MySubStruct(GatewayStruct):
    foo: str


class MyStruct(GatewayStruct):
    foo: str


class GWC(GatewayChannels):
    data: csp.ts[MyStruct] = None
    list_data: csp.ts[List[MyStruct]] = None
    dict_data: dict[str, csp.ts[MyStruct]] = None

    def dynamic_keys(self) -> Optional[Dict[str, List[Any]]]:
        """Define dynamic dictionary keys by field, driven by data from the channels."""
        keys = ["a", "b", "c"]
        return {
            GWC.dict_data: keys,
        }


class PrintModule(GatewayModule):
    def connect(self, channels):
        data = channels.get_channel(GWC.data)
        csp.print("Data", data)
        list_data = channels.get_channel(GWC.list_data)
        csp.print("ListData", list_data)
        dict_data = channels.get_channel(GWC.dict_data)
        csp.print("DictData", dict_data)


if __name__ == "__main__":
    # In this example, the functionality of ReadFileDrop is shown

    # When ran, this program will wait for:
    #  Any new json files in ./json_fd/
    #  Any new parquet files in ./parquet_fd/
    #
    # The files going into the above directories should represent MyStruct
    # as a json (for json_fd) and as a parquet (for parquet fd).
    # When a new file is found, the program will read that file and convert
    # it into the underlying structs. The PrintModule will print out any structs
    # that are created
    # An example json file might look as:
    #  [
    #       { "foo": "s1" },
    #       { "foo": "s2" },
    #  ]
    #
    # This will result in 2 structs: MyStruct(foo="s1") & MyStruct(foo="s2")
    encoding_with_engine_timestamps = True

    channels = [GWC.data, GWC.list_data]

    print("Waiting for files in ./json_fd/ & ./parquet_fd/")
    fd_module = ReadFileDrop(
        configs=[
            # JSON
            ReadFileDropConfiguration(dir_path="json_fd", channel_name="data", filedrop_type=FileDropType.JSON),
            ReadFileDropConfiguration(dir_path="json_fd", channel_name="list_data", filedrop_type=FileDropType.JSON),
            ReadFileDropConfiguration(dir_path="json_fd", channel_name="dict_data", filedrop_type=FileDropType.JSON),
            # PARQUET
            ReadFileDropConfiguration(dir_path="parquet_fd", channel_name="data", filedrop_type=FileDropType.PARQUET),
            ReadFileDropConfiguration(dir_path="parquet_fd", channel_name="list_data", filedrop_type=FileDropType.PARQUET),
            ReadFileDropConfiguration(dir_path="parquet_fd", channel_name="dict_data", filedrop_type=FileDropType.PARQUET),
        ],
    )

    gateway = Gateway(
        modules=[
            fd_module,
            PrintModule(),
            AddChannelsToGraphOutput(),
        ],
        channels=GWC(),
        channels_model=GWC,
    )

    out = csp.run(
        gateway.graph,
        starttime=datetime.now(timezone.utc),
        realtime=True,
    )
