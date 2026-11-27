import gc
import copy
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import ggufpp


def fixture(order="<"):
    """Independent GGUF v3 fixture with nested metadata and two aligned tensors."""
    def pack(fmt, *args):
        return struct.pack(order + fmt, *args)

    def string(value):
        data = value.encode("utf-8")
        return pack("Q", len(data)) + data

    def array(tag, items):
        return pack("IQ", tag, len(items)) + b"".join(items)

    entries = [
        ("general.name", 8, string("tiny模型")),
        ("general.alignment", 4, pack("I", 32)),
        ("big.count", 10, pack("Q", 2**63 + 7)),
        ("tokens", 9, array(8, [string("x"), string("世界"), string("z")])),
        ("nested", 9, array(9, [array(7, [b"\x00", b"\x01"]), array(6, [])])),
    ]
    body = b"GGUF" + pack("IQQ", 3, 2, len(entries))
    for name, tag, data in entries:
        body += string(name) + pack("I", tag) + data
    # Q4_0 has 18 encoded bytes per 32 elements. File order is deliberate.
    body += string("z.weight") + pack("IQIQ", 1, 32, 2, 0)
    body += string("a.weight") + pack("IQQIQ", 2, 2, 3, 0, 32)
    return (body + bytes(-len(body) % 32) + bytes(range(18)) + bytes(14)
            + pack("6f", 1, 2, 3, 4, 5, 6))


class PythonAPI(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name, "模型.gguf")
        self.path.write_bytes(fixture())

    def test_all_entry_points_and_byte_orders(self):
        for order, label in [("<", "little"), (">", "big")]:
            data = fixture(order)
            self.path.write_bytes(data)
            options = ggufpp.ParseOptions(byte_order=label)
            with ggufpp.open(self.path, options=options) as stream, ggufpp.mmap(
                str(self.path), options=options
            ) as mapped:
                for doc in [ggufpp.parse(data, options=options), stream, mapped]:
                    self.assertEqual((doc.version, doc.byte_order, doc.alignment), (3, label, 32))
                    self.assertEqual(doc.file_size, len(data))
                    self.assertIsInstance(doc.metadata, Mapping)
                    self.assertEqual(list(doc.metadata),
                                     ["general.name", "general.alignment", "big.count", "tokens", "nested"])
                    self.assertEqual(doc.metadata["general.name"], "tiny模型")
                    self.assertIsNone(doc.metadata.get("missing"))
                    self.assertIs(doc.metadata.value_type("general.alignment"), ggufpp.ValueType.UINT32)
                    self.assertIs(doc.metadata.value_type("big.count"), ggufpp.ValueType.UINT64)
                    self.assertEqual(doc.metadata["big.count"], 2**63 + 7)
                    self.assertEqual(list(doc.tensors), ["z.weight", "a.weight"])
                    first, second = doc.tensors.values()
                    self.assertEqual((first.encoding, first.nbytes, first.file_offset),
                                     (ggufpp.TensorType.Q4_0, 18, doc.data_offset))
                    self.assertEqual((second.dimensions, second.element_count, second.nbytes),
                                     ((2, 3), 6, 24))
                    self.assertIs(second.encoding, ggufpp.TensorType.F32)
                with mapped.tensor_view("a.weight") as raw:
                    destination = bytearray(24)
                    self.assertEqual(stream.readinto("a.weight", destination), 24)
                    self.assertEqual(bytes(raw), struct.pack(order + "6f", 1, 2, 3, 4, 5, 6))
                    self.assertEqual(destination, bytes(raw))

    def test_lazy_arrays_and_owned_input(self):
        data = bytearray(fixture())
        doc = ggufpp.parse(memoryview(data))
        data.clear()
        tokens = doc.metadata["tokens"]
        self.assertIsInstance(tokens, Sequence)
        self.assertIs(tokens.element_type, ggufpp.ValueType.STRING)
        self.assertEqual(tokens[-2], "世界")
        self.assertEqual(tokens[::-2], ("z", "x"))
        self.assertEqual(list(tokens), ["x", "世界", "z"])
        for index in [-4, 3, 2**100]:
            with self.assertRaises(IndexError):
                _ = tokens[index]
        bools, empty = doc.metadata["nested"]
        del doc
        gc.collect()
        self.assertIs(bools[1], True)
        self.assertIs(bools[0], False)
        self.assertIs(empty.element_type, ggufpp.ValueType.FLOAT32)
        self.assertEqual(len(empty), 0)

    def test_numeric_wire_types(self):
        # Use values that expose signedness and truncation at the binding boundary.
        for tag, fmt, value in [
            (0, "B", 255), (1, "b", -128), (2, "H", 65535), (3, "h", -32768),
            (4, "I", 2**32 - 1), (5, "i", -(2**31)), (6, "f", 1.25),
            (7, "?", True), (10, "Q", 2**64 - 1), (11, "q", -(2**63)), (12, "d", -1.5),
        ]:
            with self.subTest(tag=tag):
                header = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
                scalar = struct.pack("<Q", 1) + b"v" + struct.pack("<I" + fmt, tag, value)
                array = struct.pack("<Q", 1) + b"a" + struct.pack("<IIQ" + fmt, 9, tag, 1, value)
                metadata = ggufpp.parse(header + scalar + array).metadata
                self.assertIs(metadata.value_type("v"), ggufpp.ValueType(tag))
                self.assertIs(metadata["a"].element_type, ggufpp.ValueType(tag))
                for result in (metadata["v"], metadata["a"][0]):
                    self.assertIs(type(result), type(value))
                    self.assertEqual(result, value)

    def test_stream_boundaries_and_buffers(self):
        with ggufpp.open(self.path) as model:
            buf = bytearray(b"xxxxxx")
            self.assertEqual(model.readinto("z.weight", memoryview(buf)[1:5], offset=16), 2)
            self.assertEqual(buf, b"x\x10\x11xxx")
            self.assertEqual(model.readinto("z.weight", buf, offset=18), 0)
            self.assertEqual(model.readinto("z.weight", bytearray()), 0)
            for offset in [-1, 19, 2**100]:
                with self.assertRaises(ValueError):
                    model.readinto("z.weight", buf, offset=offset)
            with self.assertRaises(BufferError):
                model.readinto("z.weight", bytes(20))
            with self.assertRaises(BufferError):
                model.readinto("z.weight", memoryview(buf)[::2])
            with self.assertRaises(KeyError):
                model.readinto("missing", buf)
        with self.assertRaises(BufferError):
            ggufpp.parse(memoryview(self.path.read_bytes())[::2])

    def test_close_invalidates_directory_and_nested_arrays(self):
        for opener in [ggufpp.open, ggufpp.mmap]:
            with opener(self.path) as model:
                with self.assertRaises(TypeError):
                    copy.copy(model)
                metadata, tensors = model.metadata, model.tensors
                it = iter(metadata)
                next(it)
                array = metadata["nested"][0]
                descriptor = tensors["z.weight"]
            model.close()
            self.assertTrue(model.closed)
            self.assertEqual(descriptor.nbytes, 18)
            for access in [lambda: metadata["general.name"], lambda: len(tensors),
                           lambda: next(it), lambda: array[0], lambda: array.element_type,
                           lambda: model.version, lambda: model.__enter__()]:
                with self.assertRaises(ValueError):
                    access()
            with self.assertRaises(ValueError):
                if isinstance(model, ggufpp.File):
                    model.readinto("z.weight", bytearray(1))
                else:
                    model.tensor_view("z.weight")

    def test_view_and_slice_keep_mapping_alive(self):
        with ggufpp.mmap(self.path) as model:
            raw = model.tensor_view("z.weight")
            self.assertIsInstance(raw, memoryview)
            self.assertEqual((raw.readonly, raw.ndim, raw.format, raw.nbytes), (True, 1, "B", 18))
            with self.assertRaises(TypeError):
                raw[0] = 42
            tail = raw[7:]
        del model
        gc.collect()
        self.assertEqual(raw[0], 0)
        raw.release()
        self.assertEqual(tail.tobytes(), bytes(range(7, 18)))
        tail.release()
        # Windows permits truncation once the last exported slice is released.
        self.path.write_bytes(b"")

    def test_errors_and_options(self):
        with self.assertRaises(ggufpp.ParseError) as caught:
            ggufpp.parse(b"NOPE")
        self.assertIs(caught.exception.code, ggufpp.ErrorCode.INVALID_MAGIC)
        self.assertEqual(caught.exception.offset, 4)
        options = ggufpp.ParseOptions(max_metadata_entries=1)
        with self.assertRaises(ggufpp.ParseError) as limited:
            ggufpp.parse(fixture(), options=options)
        self.assertIs(limited.exception.code, ggufpp.ErrorCode.RESOURCE_LIMIT)
        with self.assertRaises(ValueError):
            ggufpp.ParseOptions(byte_order="native")
        with self.assertRaises(TypeError):
            ggufpp.ParseOptions("big")
        for opener in [ggufpp.mmap, ggufpp.open]:
            with self.assertRaises(OSError):
                opener(self.path.with_name("missing.gguf"))
        with ggufpp.mmap(self.path) as model:
            for lookup in [model.metadata.__getitem__, model.tensors.__getitem__,
                           model.metadata.value_type, model.tensor_view]:
                with self.assertRaises(KeyError):
                    lookup("missing")

    def test_concurrent_reads_from_one_stream(self):
        with ggufpp.open(self.path) as model, ThreadPoolExecutor(4) as pool:
            def read(offset):
                buf = bytearray(4)
                model.readinto("z.weight", buf, offset=offset)
                return bytes(buf)
            offsets = [0, 14, 5, 2] * 100
            self.assertEqual(list(pool.map(read, offsets)),
                             [bytes(range(n, n + 4)) for n in offsets])


if __name__ == "__main__":
    unittest.main()
