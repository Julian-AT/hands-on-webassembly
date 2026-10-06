"""Full model loading is deferred and failed reads cannot poison a retry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/proof"))
from repository import ASSETS, RUNTIME
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proof"
sys.path.insert(0, str(RUNTIME))
from embedding_runtime import English


class EmbeddingLazyTests(unittest.TestCase):
    def test_initialization_and_tokenization_do_not_load_vectors(self):
        model = English(ASSETS / "v1/embedding-runtime")
        with patch.object(model, "_read", wraps=model._read) as read:
            self.assertIsNone(model._vectors)
            self.assertIsNone(model._rows)
            doc = model("Don't stop!")
            self.assertEqual([t.text for t in doc], ["Do", "n't", "stop", "!"])
            self.assertEqual([c.args[0] for c in read.call_args_list], ["tokenizer.json"])
            self.assertEqual(doc.vector.shape, (300,))
            self.assertEqual(len(model.rows), 514157)
            self.assertEqual(model.vectors.shape, (20000, 300))
            calls = read.call_count
            model("Another sentence").vector
            self.assertEqual(read.call_count, calls)

    def test_failed_download_can_retry_the_complete_model(self):
        model = English(ASSETS / "v1/embedding-runtime")
        original = model._read

        def interrupted(name):
            if name == "vectors.npy":
                raise OSError("Interrupted download")
            return original(name)

        with patch.object(model, "_read", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "Interrupted"):
                model("king").vector
        self.assertIsNone(model._vectors)
        self.assertIsNone(model._rows)
        self.assertEqual(model("king").vector.shape, (300,))
        self.assertEqual(len(model.rows), 514157)


if __name__ == "__main__":
    unittest.main()
