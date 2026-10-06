"""Match the supplied IDX arrays to the complete browser dataset assets."""

from repository import ASSETS

from repository import original_path, EVIDENCE
import gzip
import json
import struct
from pathlib import Path
import numpy as np
from common import ROOT, PROOF, sha, write_json


def main():
    source = Path("assignments/Material-20261003/Fashion-MNIST/raw")
    cases = {}
    for split, prefix in [("train", "train"), ("test", "t10k")]:
        image_key = source / (prefix + "-images-idx3-ubyte")
        label_key = source / (prefix + "-labels-idx1-ubyte")
        image_path = original_path(image_key)
        label_path = original_path(label_key)
        images = image_path.read_bytes()
        labels = label_path.read_bytes()
        magic, n, h, w = struct.unpack(">IIII", images[:16])
        label_magic, n_labels = struct.unpack(">II", labels[:8])
        assert magic == 2051 and label_magic == 2049 and n == n_labels
        for key in (image_key, label_key):
            assert (
                gzip.decompress(original_path(key.with_suffix(key.suffix + ".gz")).read_bytes())
                == original_path(key).read_bytes()
            )
        packaged = ASSETS / f"v1/images/FashionMNIST-{split}.npz"
        with np.load(packaged) as archive:
            np.testing.assert_array_equal(
                np.frombuffer(images, np.uint8, offset=16).reshape(n, h, w), archive["images"]
            )
            np.testing.assert_array_equal(
                np.frombuffer(labels, np.uint8, offset=8), archive["labels"]
            )
        cases[split] = {
            "status": "pass",
            "count": n,
            "supplied_images_sha256": sha(image_path),
            "supplied_labels_sha256": sha(label_path),
            "packaged_sha256": sha(packaged),
        }
    write_json(EVIDENCE / "assignment5-data-identity.json", {"status": "pass", "cases": cases})
    print("Both complete supplied Fashion-MNIST splits match the packaged arrays exactly")


if __name__ == "__main__":
    main()
