import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from common import PROOF
from build import replace_once
from stage_image_loading import unit6_loading, unit7_loading
from reference_corrections import corrected_source


class StageImageLoadingTests(unittest.TestCase):
    def test_original_calculation_body_and_rng_calls_are_preserved(self):
        source = corrected_source(6).replace(
            "import json, math, numpy as np, pandas as pd, torch",
            "import json, math, numpy as np, pandas as pd\nfrom browser_torch import torch",
        )
        original = ast.parse(source)
        transformed = ast.parse(unit6_loading(source, replace_once))
        before = next(
            node
            for node in ast.walk(original)
            if isinstance(node, ast.FunctionDef) and node.name == "data_bundle"
        )
        after = next(
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.FunctionDef) and node.name == "_course_build_data"
        )
        self.assertEqual(
            [ast.dump(node) for node in before.body], [ast.dump(node) for node in after.body[2:]]
        )
        task = next(
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_course_load_data"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Name) and node.id in ("input", "reactive")
                for statement in task.body
                for node in ast.walk(statement)
            )
        )
        data = next(
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.FunctionDef) and node.name == "data_bundle"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "result"
                for node in ast.walk(data)
            )
        )

    def test_unit7_original_loader_body_and_generator_are_preserved(self):
        source = corrected_source(7).replace(
            "import torch\n", "from browser_torch import torch\n", 1
        )
        original = ast.parse(source)
        transformed = ast.parse(unit7_loading(source, replace_once))
        before = next(
            node
            for node in ast.walk(original)
            if isinstance(node, ast.FunctionDef) and node.name == "loaders"
        )
        after = next(
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.FunctionDef) and node.name == "_course_build_loaders"
        )
        self.assertEqual(
            [ast.dump(node) for node in before.body], [ast.dump(node) for node in after.body[2:]]
        )
        task = next(
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_course_load_data"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Name) and node.id in ("input", "reactive")
                for statement in task.body
                for node in ast.walk(statement)
            )
        )
        starter = next(
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.FunctionDef) and node.name == "_start_train_task"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "loaders"
                for node in ast.walk(starter)
            )
        )
        dispatch = next(
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.FunctionDef) and node.name == "_course_dispatch_pending_train"
        )
        condition = next(
            node
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Compare) and len(node.ops) == 2
        )
        self.assertEqual(
            ast.unparse(condition),
            "receipt['ticket'] == pending['ticket'] == _course_data_ticket.get()",
        )


if __name__ == "__main__":
    unittest.main()
