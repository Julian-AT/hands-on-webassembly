"""Release requirements derived from the supplied interfaces and task contract.

A passing diagnostic suite earns only its own row. Exhaustive suites must cover
all inventoried controls, outputs, tabs, uploads and downloads individually.
"""

from repository import EVIDENCE
from common import UNITS
from case_inventory import identities
from dynamic_choices import choice_cases
from common import PROOF
from hardware_contract import contract
import json

BROWSERS = ("chrome", "edge", "firefox", "safari")
FEATURES = {
    1: [
        "all-four-datasets",
        "summaries",
        "five-visualizations",
        "correlations",
        "pca-2d-3d",
        "tsne-2d-3d-supported-settings",
    ],
    2: [
        "all-image-operations",
        "generated-audio",
        "uploaded-audio",
        "audio-transforms-resampling",
        "playback-seeking",
        "one-hot",
        "embedding-projections",
        "similarity",
        "nearest-distant",
        "vector-arithmetic",
        "tokenizer-regression",
        "full-vocabulary",
    ],
    3: [
        "all-datasets",
        "csv-uploads-errors",
        "four-clusterers-settings",
        "overlays",
        "diagnostics",
        "tuning",
        "sampling",
        "image-segmentation",
    ],
    4: [
        "function-fitting",
        "all-six-datasets",
        "preprocessing-splits",
        "pca",
        "pairplots",
        "three-classifiers",
        "tuning",
        "decision-boundaries",
        "classification-report",
        "confusion-matrix",
        "roc-pr",
        "model-insights",
        "error-analysis",
    ],
    5: [
        "linear-regression",
        "logistic-1d",
        "logistic-2d",
        "csv-header-handling",
        "split-order",
        "coefficient-downloads",
        "dataset-downloads",
        "dataset-exploration",
        "full-mnist-training",
        "full-fashionmnist-training",
        "evaluation",
        "misclassification-analysis",
        "inversion-normalization",
        "rng-advancement",
    ],
    6: [
        "linear-regression",
        "logistic-regression",
        "seven-presets",
        "custom-layers",
        "toy-datasets",
        "image-datasets",
        "training-validation",
        "early-stopping",
        "prediction-analysis",
        "loss-averaging",
        "rng-advancement",
        "best-model-restoration",
    ],
    7: [
        "filters-building-blocks",
        "three-presets",
        "custom-layers",
        "five-image-datasets",
        "augmentation",
        "training-validation",
        "architecture-import-export",
        "filter-history",
        "intermediate-activations",
        "prediction-analysis",
        "loss-averaging",
        "epoch-reseeding-preview",
        "best-model-restoration",
    ],
}

# Confirmed defects remain explicit acceptance cases after the shared repair.
FEATURES[1] += ["tsne-zero-distance-nonleaf-summary", "tsne-native-neighbor-tie"]
FEATURES[2] += [
    "one-hot-multiword-rows",
    "one-hot-duplicate-vocabulary",
    "plotly-loading-readiness",
    "hidden-plot-resize",
]
FEATURES[6] += ["architecture-top-level-list", "architecture-invalid-root"]
FEATURES[7] += [
    "architecture-top-level-list",
    "architecture-invalid-root",
    "rectangular-convolution-pooling",
]
FEATURES[5] += [
    "disposable-training-worker",
    "dataset-replacement-cancellation",
    "owned-training-error-retry",
    "obsolete-training-error-rejected",
]


def requirements(inventory):
    rows = []
    hardware, hardware_sha256 = contract()
    choice_path = PROOF / "contracts/native-choice-discovery.json"
    discovered = (
        json.loads(choice_path.read_text()).get("controls", {}) if choice_path.exists() else {}
    )

    def add(id, report, section, cases, **metadata):
        scope = metadata.pop("acceptance_scope", None) or (
            "release"
            if metadata.get("hardware") or metadata.get("browser", "chrome") != "chrome"
            else "application"
        )
        rows.append(
            dict(
                id=id,
                report=report,
                section=section,
                required_cases=cases,
                acceptance_scope=scope,
                **metadata,
            )
        )

    for unit in UNITS:
        controls = [f"{node['call']}@{node['line']}" for node in inventory[str(unit)]["interface"]]
        behaviors = identities(unit, inventory[str(unit)]["interface"])
        stable = [case["id"] for case in behaviors]
        dynamic = sorted(
            {case["identifier"] for case in behaviors if case["requires_runtime_expansion"]}
        )
        added = [
            case["id"]
            for record in discovered.values()
            if record.get("unit") == unit
            for observation in record.get("observations", [])
            if observation.get("status") == "pass"
            for case in choice_cases(
                unit, record["identifier"], observation["parent_state"], observation["choices"]
            )
        ]
        # UI workflows and appearance are distinct evidence; pixel similarity
        # cannot satisfy a numerical or training requirement.
        for browser in BROWSERS:
            for kind, cases in [
                ("workflows", FEATURES[unit] + controls + stable + added),
                ("appearance", controls + added),
                (
                    "lifecycle",
                    [
                        "cold-start",
                        "warm-start",
                        "reload",
                        "service-worker-update",
                        "reset",
                        "failed-load-retry",
                        "navigate-during-task",
                        "stale-result-rejected",
                        "worker-terminated",
                    ],
                ),
            ]:
                metadata = (
                    {"minimum_consecutive_cycles": 30}
                    if kind == "lifecycle"
                    else {
                        "behavior_assertions": stable + added,
                        "comprehensive": True,
                        "unit": unit,
                        "dynamic_controls": dynamic,
                    }
                    if kind == "workflows"
                    else {
                        "comprehensive": True,
                        "appearance": True,
                        "unit": unit,
                        "dynamic_controls": dynamic,
                    }
                )
                add(
                    f"unit{unit}/{kind}/{browser}",
                    f"coverage/unit{unit}-{kind}-{browser}.json",
                    "cases",
                    cases,
                    browser=browser,
                    **metadata,
                )
            add(
                f"unit{unit}/8gb/{browser}",
                f"coverage/unit{unit}-8gb-{browser}.json",
                "cases",
                [
                    "cold-warm-startup",
                    "transfer-size",
                    "total-browser-peak-memory",
                    "full-presets-training-time",
                    "responsive-progress-reset",
                    "resource-exhaustion-recovery",
                ],
                browser=browser,
                hardware=True,
                hardware_contract_id=hardware["id"],
                hardware_contract_sha256=hardware_sha256,
                legacy_hardware_id=hardware["supersession"]["previous"],
            )
        add(
            f"unit{unit}/numerical",
            f"coverage/unit{unit}-numerical.json",
            "comparisons",
            FEATURES[unit],
            behavior_assertions=FEATURES[unit],
        )
    add(
        "full-image-assets",
        "coverage/full-image-assets.json",
        "cases",
        [
            f"{dataset}/{split}/{property}"
            for dataset in ("MNIST", "FashionMNIST", "CIFAR10", "SVHN", "USPS")
            for split in ("train", "test")
            for property in (
                "hash",
                "order-labels",
                "preprocessing",
                "complete-training-evaluation-inspection",
            )
        ],
    )
    add(
        "neural-seeded-behavior",
        "coverage/neural-seeded.json",
        "comparisons",
        [
            "initialization",
            "independent-generators",
            "splits",
            "loader-seed-consumption",
            "shuffling",
            "dropout",
            "augmentation",
            "batches",
            "epochs",
            "repeated-training",
            "gradients",
            "optimizer-updates",
            "restoration",
        ],
    )
    add(
        "reproducible-static-build",
        "coverage/build.json",
        "cases",
        ["clean-staging", "identical-builds", "local-asset-checksums", "lazy-assignment-assets"],
    )
    # Preserve useful measured progress as separately named, bounded checks.
    unit4 = ["fit/Noisy sine", "fit/Mystery function"] + [
        f"{ds}/{case}"
        for ds in ("Wine", "Breast Cancer", "Digits", "Pima Diabetes", "Iris", "Banknotes")
        for case in ("load-pca", "k-NN", "Decision Tree", "Random Forest")
    ]
    add(
        "unit4/default-workflows/chrome",
        "browser-chrome-unit4.json",
        "cases",
        unit4,
        browser="chrome",
    )
    add(
        "unit4/default-workflows/native",
        "native-chrome-unit4.json",
        "cases",
        unit4,
        browser="chrome",
    )
    for unit, report in [(1, "tabular"), (4, "supervised"), (6, "neural")]:
        add(
            f"unit{unit}/diagnostic-numerical",
            f"browser-chrome-unit{report}.json",
            "comparisons",
            [],
            browser="chrome",
        )
    unit5 = (
        ["linear/" + choice for choice in ("Custom", "Random", "Classes")]
        + [
            "download/polynomial-coefficients",
            "download/polynomial-dataset",
            "logistic-1d/train",
            "download/logistic-coefficients",
            "download/logistic-dataset",
            "logistic-1d/sigmoid",
            "logistic-2d/header-False",
            "logistic-2d/header-True",
            "dataset-exploration",
        ]
        + [ds + "/full-training-evaluation-inspection" for ds in ("mnist", "fashion")]
    )
    for platform in ("browser", "native"):
        add(
            f"unit5/default-workflows/{platform}",
            f"{platform}-chrome-unit5.json",
            "cases",
            unit5,
            browser="chrome",
        )
    for kind in ("unit5probe", "unit5full"):
        add(
            f"unit5/{kind}/chrome",
            f"browser-chrome-unit{kind}.json",
            "comparisons",
            [],
            browser="chrome",
        )
    add(
        "scientific-runtime/chrome",
        "browser-chrome-unitruntime.json",
        "checks",
        ["scientific-versions", "freetype-version", "matplotlib-agg", "native-fused-gini-score"],
        browser="chrome",
    )
    add(
        "unit5/cancellation/chrome",
        "browser-chrome-unitunit5life.json",
        "cases",
        [
            "reset-during-training",
            "stale-result-rejected",
            "retry-after-cancellation",
            "navigate-during-task",
            "worker-terminated",
        ],
        browser="chrome",
    )
    # Delivery is part of final certification. These release-only rows cannot
    # authorize the shell early or replace any of the original 133 suites.
    for browser in BROWSERS:
        add(
            f"nextjs/navigation/{browser}",
            f"coverage/nextjs-navigation-{browser}.json",
            "cases",
            [
                "default-unit",
                "all-seven-selections",
                "invalid-unit-normalization",
                "back-forward",
                "reload",
                "one-same-origin-iframe",
                "ready-inputs",
                "navigation-disposal",
                "standalone-direct-links",
            ],
            browser=browser,
            acceptance_scope="release",
            behavior_assertions=[
                "default-unit",
                "all-seven-selections",
                "invalid-unit-normalization",
                "back-forward",
                "reload",
                "one-same-origin-iframe",
                "ready-inputs",
                "navigation-disposal",
                "standalone-direct-links",
            ],
        )
    add(
        "nextjs/reproducible-export",
        "coverage/nextjs-build.json",
        "cases",
        [
            "pinned-dependencies",
            "two-clean-identical-builds",
            "static-export",
            "asset-archive-sha256",
            "extracted-file-manifest",
            "small-hobby-source-upload",
            "licenses",
        ],
        acceptance_scope="release",
        behavior_assertions=[
            "pinned-dependencies",
            "two-clean-identical-builds",
            "static-export",
            "asset-archive-sha256",
            "extracted-file-manifest",
            "small-hobby-source-upload",
            "licenses",
        ],
    )
    hosting = [
        "https",
        "deployed-file-manifest",
        "zero-runtime-functions",
        "javascript-wasm-mime",
        "get-head",
        "byte-ranges",
        "redirects-direct-links",
        "cache-policy",
        "stable-origin-real-upgrades",
    ]
    add(
        "vercel/https-static-delivery",
        "coverage/vercel-https.json",
        "cases",
        hosting,
        acceptance_scope="release",
        behavior_assertions=hosting,
    )
    delivery = [
        "all-default-training-combinations",
        "three-independent-seed42-repetitions",
        "no-concurrent-verification",
        "raw-phase-timings",
        "memory-definition",
        "hardware-browser-cache-power",
        "physical-8gb-integrated-graphics",
        "versioned-release-asset",
        "release-rollback-instructions",
    ]
    add(
        "release/benchmarks-and-materials",
        "coverage/release-materials.json",
        "cases",
        delivery,
        acceptance_scope="release",
        behavior_assertions=delivery,
        hardware=True,
        hardware_contract_id=hardware["id"],
        hardware_contract_sha256=hardware_sha256,
        superseded_cases={"physical-8gb-integrated-graphics": hardware["id"]},
    )
    return rows
