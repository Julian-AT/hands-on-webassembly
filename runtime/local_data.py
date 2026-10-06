"""Local, full OpenML snapshots for the exact two datasets used by Unit 3."""
import json
from pathlib import Path
import pandas as pd
from sklearn.utils import Bunch


def fetch_openml(name, *, version=1, as_frame=False):
    if name not in ("seeds", "ionosphere") or version != 1:
        raise ValueError(f"Unpackaged OpenML dataset: {name} v{version}")
    payload = json.loads((Path(__file__).parent / "resources" / f"{name}.json").read_text())
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    target = payload["target"]
    frame[target] = pd.Categorical(frame[target], categories=payload["categories"])
    X, y = frame.drop(columns=[target]), frame[target]
    return Bunch(data=X if as_frame else X.to_numpy(), target=y if as_frame else y.to_numpy(),
                 frame=frame if as_frame else None, feature_names=list(X.columns),
                 target_names=[target], details=payload["details"])
