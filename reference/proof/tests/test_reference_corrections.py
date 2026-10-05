"""Reproduce native defects before checking the shared corrected reference."""
import ast
import json
import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT
from reference_corrections import corrected_source


def definitions(source, names):
    nodes = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = [] if node.name != "LayerSpec" else node.decorator_list
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    scope = dict(globals(), nn=torch.nn)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app-functions>", "exec"), scope)
    return scope


class ReferenceCorrectionsTests(unittest.TestCase):
    def test_duplicate_vocabulary_preserves_order_without_false_unknown(self):
        original = (ROOT / "assignments/2/u2_utils.py").read_text()
        old = definitions(original, ["convert_to_onehot"])["convert_to_onehot"]
        new = definitions(corrected_source(2, "u2_utils.py"), ["convert_to_onehot"])["convert_to_onehot"]
        vocab = ["zebra", "zebra", "cat"]
        self.assertTrue(old(vocab, "cat")[1])
        frame, unknown = new(vocab, "cat zebra unknown")
        np.testing.assert_array_equal(frame.values, [[0, 1], [1, 0], [0, 0]])
        self.assertEqual(unknown, ["Word 'unknown' not found in vocabulary."])
        for vocab in (["zebra", "cat"], ["cat"], []):
            pd.testing.assert_frame_equal(old(vocab, "cat zebra")[0], new(vocab, "cat zebra")[0])

    def test_multiword_one_hot_uses_actual_token_rows(self):
        helper = definitions(corrected_source(2, "u2_utils.py"), ["convert_to_onehot"])["convert_to_onehot"]
        frame, _ = helper(["New York", "cat"], "New York cat")
        with self.assertRaises(ValueError):
            pd.DataFrame(frame.values, index=["New York", "cat"])
        from types import SimpleNamespace
        scope = definitions(corrected_source(2), ["one_hot_table"])
        scope.update(get_words=lambda: ["New York", "cat"],
                     u2=SimpleNamespace(convert_to_onehot=helper),
                     render=SimpleNamespace(DataGrid=lambda view, **kwargs: view))
        result = scope["one_hot_table"]()
        self.assertEqual(result["word"].tolist(), ["New", "York", "cat"])
        self.assertEqual(result.iloc[2, 1:].tolist(), ["0", "1"])

    def test_both_architecture_importers_accept_lists_and_objects(self):
        for unit in (6, 7):
            names = ["parse_architecture", "ACTIVATIONS", "LayerSpec", "_yamlish_to_json"]
            original = definitions((ROOT / f"assignments/{unit}/app.py").read_text(), names)
            with self.assertRaises(AttributeError):
                original["parse_architecture"]('[{"type":"linear","out_features":2}]')
            parse = definitions(corrected_source(unit), names)["parse_architecture"]
            layers = [{"type": "linear", "out_features": 2}]
            self.assertEqual(parse(json.dumps(layers)), parse(json.dumps({"layers": layers})))
            for invalid in ("null", "42", '"bad"', '{"layers":1}', '[1]'):
                with self.assertRaises(ValueError):
                    parse(invalid)

    def test_rectangular_convolution_and_pools_match_real_tensor_shapes(self):
        names = ["parse_architecture", "ACTIVATIONS", "LayerSpec", "_yamlish_to_json", "_tuple_or_int", "build_model"]
        old = definitions((ROOT / "assignments/7/app.py").read_text(), names)
        new = definitions(corrected_source(7), names)
        for pool in ("maxpool2d", "avgpool2d"):
            layers = [{"type": "conv2d", "out_channels": 2, "kernel_size": [3, 5], "stride": [1, 2], "padding": [1, 0]},
                      {"type": pool, "kernel_size": [2, 3], "stride": [2, 1]},
                      {"type": "linear", "out_features": 3}]
            x = torch.zeros(4, 1, 12, 24)
            with self.assertRaises(RuntimeError):
                old["build_model"](old["parse_architecture"](json.dumps({"layers": layers})), (1, 12, 24))(x)
            model = new["build_model"](new["parse_architecture"](json.dumps(layers)), (1, 12, 24))
            self.assertEqual(tuple(model(x).shape), (4, 3))
            self.assertEqual(model[-1].in_features, 2 * 6 * 8)


if __name__ == "__main__":
    unittest.main()
