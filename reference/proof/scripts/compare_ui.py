"""Compare captured native/browser text and one landing-page screenshot."""
import json
import numpy as np
from PIL import Image
from common import PROOF, write_json


def main():
    for unit in (1,3):
        native = json.loads((PROOF / f"evidence/native-chrome-unit{unit}.json").read_text())
        browser = json.loads((PROOF / f"evidence/browser-chrome-unit{unit}.json").read_text())
        comparison = {}
        for key, value in browser["cases"].items():
            old = native["cases"].get(key,{})
            if unit == 1:
                value, old = value.get("detail") or {}, old.get("detail") or {}
            comparison[key] = {field:value[field] == old.get(field) for field in ("info","summary","variance","text") if field in value}
        write_json(PROOF / f"evidence/unit{unit}-text-comparison.json",comparison)
    screenshots = PROOF / "evidence/screenshots"
    native = Image.open(screenshots / "native-chrome-unit1-intro.png").convert("RGB")
    browser = Image.open(screenshots / "browser-chrome-unit1-intro.png").convert("RGB")
    size = (min(native.width,browser.width),min(native.height,browser.height))
    difference = np.abs(np.asarray(native.crop((0,0,*size))).astype(float) - np.asarray(browser.crop((0,0,*size))).astype(float))
    write_json(PROOF / "evidence/unit1-visual-comparison.json", {
        "common_viewport":size,"mean_channel_abs_error":float(difference.mean()),
        "fraction_pixels_differing":float(np.mean(np.any(difference>0,axis=-1))),
        "qualification":"One landing-page screenshot comparison only; not exhaustive visual parity.",
    })


if __name__ == "__main__":
    main()
