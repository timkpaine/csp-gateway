from datetime import datetime, timezone

import csp

from csp_gateway import (
    AddChannelsToGraphOutput,
    FiledropConfiguration,
    FiledropType,
    Gateway,
    GatewayChannels,
    GatewayModule,
    GatewayStruct,
    ReadFiledrop,
)


class MyStruct(GatewayStruct):
    foo: str


class GWC(GatewayChannels):
    data: csp.ts[MyStruct] = None
    list_data: csp.ts[[MyStruct]] = None


class PrintModule(GatewayModule):
    def connect(self, channels):
        data = channels.get_channel(GWC.data)
        csp.print("Data", data)
        list_data = channels.get_channel(GWC.list_data)
        csp.print("ListData", list_data)


if __name__ == "__main__":
    # In this example, the functionality of ReadFiledrop is shown

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
    fd_module = ReadFiledrop(
        directory_configs={
            "json_fd": [
                FiledropConfiguration(channel_name="data", fd_type=FiledropType.JSON),
                FiledropConfiguration(channel_name="list_data", fd_type=FiledropType.JSON),
            ],
            "parquet_fd": [
                FiledropConfiguration(channel_name="data", fd_type=FiledropType.PARQUET),
                FiledropConfiguration(channel_name="list_data", fd_type=FiledropType.PARQUET),
            ],
        }
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
