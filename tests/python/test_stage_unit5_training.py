import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from common import ROOT
from unit5_training_source import training_source
from reference_corrections import corrected_source


class StageUnit5TrainingTests(unittest.TestCase):
    def test_worker_keeps_supplied_body_verbatim(self):
        original = ast.parse((ROOT / "assignments/5/app.py").read_text())
        worker = ast.parse(training_source())
        before = next(
            n
            for n in ast.walk(original)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_mn_train"
        )
        after = next(
            n
            for n in ast.walk(worker)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_mn_train"
        )
        self.assertEqual([ast.dump(n) for n in before.body], [ast.dump(n) for n in after.body])

    def test_extended_task_does_not_access_live_reactive_inputs(self):
        tree = ast.parse(corrected_source(5))
        task = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_mn_train_task"
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Name) and n.id in ("input", "reactive")
                for statement in task.body
                for n in ast.walk(statement)
            )
        )
        consumer = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_course_finalize_train"
        )
        self.assertIn("receipt['ticket'] != _course_train_ticket.get()", ast.unparse(consumer))
        reset = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_mn_reset_on_dataset_change"
        )
        self.assertIn("_course_cancel_train()", ast.unparse(reset))


if __name__ == "__main__":
    unittest.main()
