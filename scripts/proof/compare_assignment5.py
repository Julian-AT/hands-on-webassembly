"""Compare paired original/browser downloads and same-viewport screenshots."""

from repository import EVIDENCE
import json
import numpy as np
from PIL import Image
from common import PROOF, sha, write_json
from provenance import stamp


def main():
    cases = {}
    visual = {}
    for name in (
        "u5_polynomial_coeffs.json",
        "u5_logistic_coeffs.json",
        "u5_polynomial_regression_dataset.csv",
        "u5_logistic_dataset.csv",
    ):
        a = EVIDENCE / "downloads/native-chrome-unit5" / name
        b = EVIDENCE / "downloads/browser-chrome-unit5" / name
        if name.endswith(".json"):
            left, right = json.loads(a.read_text()), json.loads(b.read_text())
            equal = left.keys() == right.keys() and np.allclose(
                left["coefficients"], right["coefficients"], rtol=1e-4, atol=1e-5
            )
        else:
            equal = a.read_bytes() == b.read_bytes()
        cases[name] = {
            "status": "pass" if equal else "fail",
            "native_sha256": sha(a),
            "browser_sha256": sha(b),
            "byte_identical": a.read_bytes() == b.read_bytes(),
        }
    screenshots = EVIDENCE / "screenshots"
    for a in sorted(screenshots.glob("native-chrome-unit5-*.png")):
        b = a.with_name(a.name.replace("native-", "browser-", 1))
        if not b.exists():
            continue
        x, y = np.asarray(Image.open(a).convert("RGB")), np.asarray(Image.open(b).convert("RGB"))
        record = {
            "native_size": list(x.shape),
            "browser_size": list(y.shape),
            "same_size": x.shape == y.shape,
            "native_sha256": sha(a),
            "browser_sha256": sha(b),
        }
        if x.shape == y.shape:
            delta = np.abs(x.astype(float) - y.astype(float))
            record.update(
                mean_channel_abs_error=float(delta.mean()),
                fraction_pixels_differing=float(np.any(delta, axis=-1).mean()),
            )
        visual[a.stem.replace("native-chrome-unit5-", "")] = record
    current = stamp()
    reports = [
        json.loads((EVIDENCE / f"{prefix}-chrome-unit5.json").read_text())
        for prefix in ("native", "browser")
    ]
    current_inputs = all(
        report.get("status") == "pass"
        and report.get("provenance", {}).get("fingerprint") == current["fingerprint"]
        for report in reports
    )
    status = "pass" if all(v["status"] == "pass" for v in cases.values()) else "fail"
    write_json(
        EVIDENCE / "unit5-comparison.json",
        {
            "status": status if current_inputs else "stale",
            "observed_status": status,
            "provenance": current,
            "cases": cases,
            "screenshots": visual,
            "scope": "Download content comparison and measured screenshots only. Pixel differences require review; no exhaustive visual-parity claim.",
        },
    )


if __name__ == "__main__":
    main()
