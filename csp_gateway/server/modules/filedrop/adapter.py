import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, get_args, get_origin

import orjson
import pyarrow.parquet as pq
from csp import ts
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.wiring import py_push_adapter_def
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

    def __init__(self, config: FileDropAdapterConfiguration, ts_typ: object):
        self.field_map = config.field_map
        self.extensions = config.extensions
        normalized_type = ContainerTypeNormalizer.normalize_type(ts_typ)
        self.is_list = get_origin(normalized_type) is list
        if self.is_list:
            inner_type = get_args(normalized_type)[0]
            type_adapter = inner_type.type_adapter()
        else:
            type_adapter = ts_typ.type_adapter()
        self.type_adapter = type_adapter
        if hasattr(config, "type_adapter_args"):
            self.context = config.type_adapter_args
        else:
            self.context = {}

    def read(self, src_path: str) -> object:
        """Generator to return stucts from a filepath"""

        should_read = True
        if self.extensions:
            if not any([src_path.endswith(suffix) for suffix in self.extensions]):
                should_read = False
        if should_read:
            dicts = self.read_impl(src_path)
            structs = [self.deserialize_dict(self.apply_field_map(v)) for v in dicts]
            if self.is_list:
                yield structs
            else:
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

    def deserialize_dict(self, dict: dict) -> object:
        """Convert a dict to struct"""

        return self.type_adapter.validate_python(dict, context=self.context)


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


class _FileDropImpl(PushInputAdapter):
    FILEREADER_MAP = {
        FileDropType.CSV: FileReaderCsv,
        FileDropType.JSON: FileReaderJson,
        FileDropType.PARQUET: FileReaderParquet,
    }

    def __init__(self, config: FileDropAdapterConfiguration, ts_typ: "T"):  # noqa
        self.dir_path = config.dir_path
        self.observer = Observer()
        reader = self.FILEREADER_MAP[config.filedrop_type]
        file_reader = reader(config, ts_typ)
        self.event_handler = EventHandlerCustom(self, file_reader)
        self.observer.schedule(self.event_handler, self.dir_path, recursive=False)

    def start(self, starttime: datetime, endtime: datetime):
        self.observer.start()
        self.observer_started = True

    def stop(self):
        if self.observer_started:
            self.observer.stop()
            self.observer.join()


filedrop_adapter_def = py_push_adapter_def(
    "filedrop_adapter_def",
    _FileDropImpl,
    ts["T"],
    config=FileDropAdapterConfiguration,
    ts_typ="T",
)
