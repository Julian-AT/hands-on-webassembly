"""Generate test inputs from canonical source, without a prior scientific build."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/proof"))
from repository import REFERENCE, RUNTIME, REQUIREMENTS
from reference_corrections import training_source
from common import sha


def main():
    target = REFERENCE / "unit7"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUNTIME / "isolated_training.py", target / "isolated_training.py")
    body = training_source(7)
    identity = hashlib.sha256(
        (body + sha(target / "isolated_training.py") + sha(REQUIREMENTS / "native.lock")).encode()
    ).hexdigest()
    (target / "course-training.json").write_text(
        json.dumps({"source": body, "build_id": identity}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
