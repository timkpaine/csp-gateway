import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TypeVar, get_args, get_origin

import orjson
import pyarrow.parquet as pq
from csp import ts
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.wiring import py_push_adapter_def
from pydantic import TypeAdapter
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

__all__ = (
    "FileDropType",
    "FileDropAdapterConfiguration",
    "filedrop_adapter_def",
)


log = logging.getLogger(__name__)


class FileDropType(Enum):
    CSV = auto()
    JSON = auto()
    PARQUET = auto()


@dataclass
class FileDropAdapterConfiguration:
    """Configuration for the filedrop push adapter"""

    # Path to the directory to monitor
    dir_path: str
    # Format of files to expect to load properly i.e parquet, json, csv
    filedrop_type: FileDropType
    # Map the data fields from the file to the fields of the structs
    field_map: Dict[str, str]
    # List of extensions to filter, empty list means all extensions are allowed
    extensions: List[str]
    # Extra args to the type adapter deserializer
    type_adapter_args: Dict[str, Any]


class FileReaderBase:
    """The base file reader that reads data from files and generates structs"""

    def __init__(self, config: FileDropAdapterConfiguration, ts_typ: object, deserializer: Optional[object] = None):
        self.field_map = config.field_map
        self.extensions = config.extensions
        if hasattr(config, "type_adapter_args"):
            self.context = config.type_adapter_args
        else:
            self.context = {}
        if not deserializer:
            normalized_type = ContainerTypeNormalizer.normalize_type(ts_typ)
            type_adapter = TypeAdapter(normalized_type)
            if get_origin(normalized_type) is list:

                def deserialize_tick(data, type_adapter=type_adapter, apply_field_map=self.apply_field_map, context=self.context):
                    data = [apply_field_map(d) for d in data]
                    return type_adapter.validate_python(data, context=context)
            elif get_origin(normalized_type) is dict:
                key_type, inner_type = get_args(normalized_type)

                def deserialize_tick(data, type_adapter=type_adapter, apply_field_map=self.apply_field_map, context=self.context):
                    data = {k: apply_field_map(d) for k, d in data.items()}
                    return type_adapter.validate_python(data, context=context)
            else:

                def deserialize_tick(data, type_adapter=type_adapter, apply_field_map=self.apply_field_map, context=self.context):
                    return type_adapter.validate_python(apply_field_map(data), context=context)

            self.deserializer = deserialize_tick
        else:
            self.deserializer = deserializer

    def read(self, src_path: str) -> object:
        """Generator to return stucts from a filepath"""

        should_read = True
        if self.extensions:
            if not any([src_path.endswith(suffix) for suffix in self.extensions]):
                should_read = False
        if should_read:
            dicts = self.read_impl(src_path)
            structs = [self.deserializer(d) for d in dicts]
            for s in structs:
                yield s

    def read_impl(self, src_path: str) -> List[dict]:
        """File type specific implementation"""

        raise Exception(f"read not implemented for {self}")

    def apply_field_map(self, data: dict) -> dict:
        """Convert the keys in the data to the field names of the struct"""

        if self.field_map:
            new_data = {}
            for k, v in data.items():
                new_k = self.field_map.get(k, k)
                new_data[new_k] = v
            return new_data
        else:
            return data


class FileReaderCsv(FileReaderBase):
    """File reader for json file type"""

    def read_impl(self, src_path: str) -> List[dict]:
        data = []
        with open(src_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data


class FileReaderJson(FileReaderBase):
    """File reader for json file type"""

    def read_impl(self, src_path: str) -> List[dict]:
        with open(src_path, "rb") as f:
            data = orjson.loads(f.read())
        if isinstance(data, list):
            res = data
        else:
            res = [data]
        return res


class FileReaderParquet(FileReaderBase):
    """File reader for parquet file type"""

    def read_impl(self, src_path: str) -> List[dict]:
        table = pq.read_table(src_path)
        return table.to_pylist()


class EventHandlerCustom(FileSystemEventHandler):
    def __init__(self, adapter: PushInputAdapter, file_reader: FileReaderBase):
        self.file_reader = file_reader
        self.adapter = adapter
        self._created = set()
        self._opened = set()
        self._modified = set()
        self._closed = set()

    def on_created(self, event: FileSystemEvent):
        self._created.add(event.src_path)

    def on_opened(self, event: FileSystemEvent):
        if event.src_path in self._created:
            self._created.remove(event.src_path)
            self._opened.add(event.src_path)

    def on_modified(self, event: FileSystemEvent):
        if event.src_path in self._opened:
            self._opened.remove(event.src_path)
            self._modified.add(event.src_path)

    def on_closed(self, event: FileSystemEvent):
        if event.src_path in self._modified:
            self._modified.remove(event.src_path)
            file_path = event.src_path
            try:
                for data in self.file_reader.read(file_path):
                    self.adapter.push_tick(data)
            except Exception as e:
                log.error(f"Failed to read data from {file_path} with exception: {e}, skipping")


T = TypeVar("T")


class _FileDropImpl(PushInputAdapter):
    FILEREADER_MAP = {
        FileDropType.CSV: FileReaderCsv,
        FileDropType.JSON: FileReaderJson,
        FileDropType.PARQUET: FileReaderParquet,
    }

    def __init__(self, config: FileDropAdapterConfiguration, ts_typ: T, deserializer: Optional[object] = None):
        # NOTE ts_typ is assumed to be a List["Y"] type where "Y" is the actual type
        self.dir_path = config.dir_path
        self.observer = Observer()
        reader = self.FILEREADER_MAP[config.filedrop_type]
        file_reader = reader(config, ts_typ[0], deserializer)
        self.event_handler = EventHandlerCustom(self, file_reader)
        self.observer.schedule(self.event_handler, self.dir_path, recursive=False)

    def start(self, starttime: datetime, endtime: datetime):
        self.observer.start()
        self.observer_started = True

    def stop(self):
        if self.observer_started:
            self.observer.stop()
            self.observer.join()


# NOTE: Ref: https://github.com/Point72/csp/issues/569
# Instead of passing the actual type T we want as ts_typ, we have to pass in List[T]
# This is due to a type normalization bug in csp, where it converts Dict[K, T] to dict, i.e removing the type annotation hints
# The type annotations are needed to properly deserialize the raw data using the pydantic type adapters.
# Further we can only take the object type as output due to issues caused by using List[T] instead of T and bypassing
# the type normalization process in csp. The user of the adapter, needs to keep track of the type they pass in ts_type,
# and cast the ts[object] to the required ts[T], they want using csp.appy
filedrop_adapter_def = py_push_adapter_def(
    "filedrop_adapter_def",
    _FileDropImpl,
    ts[object],
    config=FileDropAdapterConfiguration,
    ts_typ=List[T],
    deserializer=Optional[object],
)
