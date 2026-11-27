"""Inspect GGUF documents and read encoded tensor payloads."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping, Sequence
from os import PathLike
from sys import maxsize

from . import _core
from ._core import ErrorCode, ParseError, ParseOptions, TensorInfo, TensorType, ValueType

__all__ = [
    "open", "mmap", "parse", "Document", "File", "MappedFile", "MetadataArray",
    "TensorInfo", "TensorType", "ValueType", "ParseOptions", "ParseError", "ErrorCode",
]

_DEFAULT_OPTIONS = ParseOptions()


def _value(native):
    if isinstance(native, _core._Array):
        return MetadataArray(native)
    return native


class MetadataArray(Sequence):
    """Read-only sequence. Each access converts only the requested elements."""

    def __init__(self, native):
        self._native = native

    @property
    def element_type(self) -> ValueType:
        return self._native.element_type

    def __len__(self) -> int:
        return len(self._native)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(_value(self._native.item(i)) for i in range(*index.indices(len(self))))
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index > maxsize:
            raise IndexError("metadata array index out of range")
        return _value(self._native.item(index))


class _Metadata(Mapping):
    def __init__(self, directory):
        self._directory = directory

    def __len__(self) -> int:
        return self._directory.metadata_count()

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self._directory.metadata_key(i)

    def __getitem__(self, key: str):
        if not isinstance(key, str):
            raise KeyError(key)
        return _value(self._directory.metadata_value(key))

    def value_type(self, key: str) -> ValueType:
        """Return the exact GGUF type, including for Python integer and float values."""
        if not isinstance(key, str):
            raise KeyError(key)
        return self._directory.metadata_type(key)


class _Tensors(Mapping):
    def __init__(self, directory):
        self._directory = directory

    def __len__(self) -> int:
        return self._directory.tensor_count()

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self._directory.tensor_name(i)

    def __getitem__(self, name: str) -> TensorInfo:
        if not isinstance(name, str):
            raise KeyError(name)
        return self._directory.tensor(name)


class Document:
    """GGUF header, metadata and tensor directory in file order."""

    def __init__(self, directory):
        self._directory = directory

    @property
    def metadata(self) -> _Metadata:
        return _Metadata(self._directory)

    @property
    def tensors(self) -> Mapping[str, TensorInfo]:
        return _Tensors(self._directory)

    @property
    def version(self) -> int:
        return self._directory.version

    @property
    def byte_order(self) -> str:
        return self._directory.byte_order

    @property
    def alignment(self) -> int:
        return self._directory.alignment

    @property
    def file_size(self) -> int:
        return self._directory.file_size

    @property
    def data_offset(self) -> int:
        return self._directory.data_offset


class _FileHandle(Document):
    def __init__(self, source):
        super().__init__(source.directory())
        self._source = source

    @property
    def closed(self) -> bool:
        return self._directory.closed

    def close(self) -> None:
        """Close this handle and invalidate its metadata views."""
        self._directory.close()
        self._source = None

    def _open_source(self):
        source = self._source
        if source is None:
            raise ValueError("I/O operation on a closed GGUF file")
        return source

    def __enter__(self):
        self._open_source()
        return self

    def __copy__(self):
        raise TypeError("GGUF file handles cannot be copied")

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class File(_FileHandle):
    """Buffered GGUF file returned by open()."""

    def readinto(self, name: str, buffer, *, offset: int = 0) -> int:
        """Copy encoded bytes, starting at a tensor-relative offset. Return bytes read."""
        offset = operator.index(offset)
        if not 0 <= offset <= (1 << 64) - 1:
            raise ValueError("offset outside tensor bounds")
        return self._open_source().readinto(name, buffer, offset)


class MappedFile(_FileHandle):
    """Read-only mapped GGUF file returned by mmap()."""

    def tensor_view(self, name: str) -> memoryview:
        """Borrow encoded bytes. The view and its slices keep the mapping alive."""
        return memoryview(self._open_source().payload(name))


def open(path: str | PathLike, *, options: ParseOptions | None = None) -> File:
    """Open a GGUF file for inspection and buffered tensor reads."""
    return File(_core.open(path, _DEFAULT_OPTIONS if options is None else options))


def mmap(path: str | PathLike, *, options: ParseOptions | None = None) -> MappedFile:
    """Map a GGUF file for inspection and zero-copy tensor views."""
    return MappedFile(_core.mmap(path, _DEFAULT_OPTIONS if options is None else options))


def parse(buffer, *, options: ParseOptions | None = None) -> Document:
    """Parse a complete contiguous buffer. The document owns its decoded metadata."""
    return Document(_core.parse(buffer, _DEFAULT_OPTIONS if options is None else options))
