import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v41"))
import gguf_bytes  # noqa: E402


def w_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_u32(key, value):
    return w_str(key) + struct.pack("<II", 4, value)


def kv_str(key, value):
    return w_str(key) + struct.pack("<I", 8) + w_str(value)


def kv_str_array(key, items):
    return (w_str(key) + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(items))
            + b"".join(w_str(i) for i in items))


def tinfo(name, dims, ttype, offset):
    return (w_str(name) + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ttype, offset))


# name, dims, type, offset: sizes follow from consecutive offsets; 1888 = end of data
TENSORS = [
    ("token_embd.weight", [8, 10], 1, 0),                 # 160
    ("blk.0.attn_q_a.weight", [8, 8], 8, 160),             # 96
    ("blk.0.ffn_gate_exps.weight", [8, 4, 4], 16, 256),    # 128
    ("blk.0.ffn_up_exps.weight", [8, 4, 4], 16, 384),      # 128
    ("blk.0.ffn_down_exps.weight", [4, 8, 4], 10, 512),    # 256
    ("blk.1.engram_embd.weight", [264, 100], 24, 768),     # 1024
    ("output.weight", [8, 10], 8, 1792),                   # 96
]
DATA_BYTES = 1888


def write_gguf(path):
    kvs = [kv_str("general.architecture", "deepseek41"), kv_u32("general.alignment", 32),
           kv_u32("deepseek41.num_hidden_layers", 2), kv_u32("deepseek41.n_routed_experts", 4),
           kv_u32("deepseek41.num_experts_per_tok", 2),
           kv_str_array("tokenizer.ggml.tokens", ["a", "b", "c"])]
    head = (b"GGUF" + struct.pack("<IQQ", 3, len(TENSORS), len(kvs)) + b"".join(kvs)
            + b"".join(tinfo(*t) for t in TENSORS))
    pad = (-len(head)) % 32
    with open(path, "wb") as fp:
        fp.write(head + b"\0" * pad + b"\0" * DATA_BYTES)
    return len(head) + pad


class HeaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "m.gguf")
        self.data_start = write_gguf(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_header(self):
        meta, tensors, data_start = gguf_bytes.read_header(self.path)
        self.assertEqual(meta["deepseek41.n_routed_experts"], 4)
        self.assertEqual(meta["tokenizer.ggml.tokens"], ["a", "b", "c"])
        self.assertEqual(len(tensors), 7)
        self.assertEqual(data_start, self.data_start)

    def test_account(self):
        meta, tensors, data_start = gguf_bytes.read_header(self.path)
        acct = gguf_bytes.account(meta, tensors, data_start, os.path.getsize(self.path))
        self.assertEqual(acct["roles"], {"embedding": 160, "attention": 96, "routed": 512,
                                         "engram_disk": 1024, "output": 96})
        self.assertEqual(acct["per_token"], {"resident": 192, "embedding_row": 16, "routed": 256,
                                             "engram_rows": 24 * 264, "ram_total": 464})
        self.assertEqual(acct["resident_floor"], 352)
        self.assertEqual(acct["engram_tables"], 1)
        self.assertEqual(acct["expert_unit_bytes"], {"0": 128})
        self.assertEqual((acct["n_layer"], acct["n_expert"], acct["k"], acct["n_tensors"]),
                         (2, 4, 2, 7))

    def test_rejects_non_gguf(self):
        bad = os.path.join(self.tmp.name, "bad.bin")
        with open(bad, "wb") as fp:
            fp.write(b"NOPE" + b"\0" * 64)
        with self.assertRaises(ValueError):
            gguf_bytes.read_header(bad)


class RoleTest(unittest.TestCase):
    def test_roles(self):
        cases = {
            "blk.3.indexer.attn_q_b.weight": "indexer",
            "blk.3.attn_compressor_norm.weight": "compressor",
            "blk.3.hc_attn_fn.weight": "hc",
            "blk.3.ffn_gate_shexp.weight": "shared",
            "blk.3.ffn_exp_probs_b.bias": "router",
            "blk.3.ffn_gate_inp.weight": "router",
            "blk.3.engram_kv.weight": "engram",
            "blk.3.attn_q_a_norm.weight": "attention",
            "blk.3.ffn_norm.weight": "norm",
            "output_norm.weight": "output",
        }
        for name, want in cases.items():
            self.assertEqual(gguf_bytes.role(name), want, name)


class RooflineTest(unittest.TestCase):
    def test_one_ms_at_290(self):
        acct = {"per_token": {"resident": 290_000_000, "embedding_row": 0, "routed": 0}}
        r = gguf_bytes.roofline(acct, 290.0)
        self.assertAlmostEqual(r["total_ms"], 1.0)
        self.assertAlmostEqual(r["tps_ceiling"], 1000.0)


if __name__ == "__main__":
    unittest.main()
