"""Independent native builder/loss controls for every preset and dataset shape.

These synthetic two-sample controls discover executable loss compatibility and
retain unrounded trajectories. They do not certify full-data browser training.
"""

from repository import EVIDENCE

from repository import ASSETS, FIXTURES, REQUIREMENTS, SCRIPTS
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import torch
from common import ROOT, PROOF, sha, write_json
from reference_corrections import corrected_source
from hardware_contract import collect
from storage_budget import require_space

STATES = {
    6: {
        "toy_reg": ((1,), 1, "regression"),
        "toy_sine": ((1,), 1, "regression"),
        "toy_bin": ((1,), 2, "classification"),
        "toy_bin2": ((1,), 2, "classification"),
        "blob2d": ((2,), 2, "classification"),
        "moons2d": ((2,), 2, "classification"),
        "MNIST": ((1, 28, 28), 10, "classification"),
        "FashionMNIST": ((1, 28, 28), 10, "classification"),
    },
    7: {
        "MNIST": ((1, 28, 28), 10, "classification"),
        "FashionMNIST": ((1, 28, 28), 10, "classification"),
        "CIFAR10": ((3, 32, 32), 10, "classification"),
        "SVHN": ((3, 32, 32), 10, "classification"),
        "USPS": ((1, 16, 16), 10, "classification"),
    },
}


def builders(source):
    names = {
        "ACTIVATIONS",
        "PRESETS",
        "LayerSpec",
        "parse_architecture",
        "_yamlish_to_json",
        "_tuple_or_int",
        "build_model",
    }
    nodes = []
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign):
            name = next(
                (t.id for t in node.targets if isinstance(t, ast.Name) and t.id in names), None
            )
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in names:
            nodes.append(node)
    scope = dict(globals(), nn=torch.nn)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<native-builders>", "exec"), scope)
    return scope


def trajectory(scope, architecture, state, seed):
    shape, classes, kind = state
    torch.manual_seed(seed)
    model = scope["build_model"](scope["parse_architecture"](json.dumps(architecture)), shape)
    x = torch.linspace(-1, 1, 2 * math.prod(shape), dtype=torch.float32).reshape(2, *shape)
    arrays = {f"initial/{k}": v.detach().numpy().copy() for k, v in model.state_dict().items()}
    arrays["initial/rng"] = torch.get_rng_state().numpy().copy()
    y = torch.tensor([[0.25], [-0.5]]) if kind == "regression" else torch.tensor([0, classes - 1])
    output = model(x)
    arrays.update(inputs=x.numpy(), targets=y.numpy(), logits=output.detach().numpy())
    error = None
    try:
        with warnings.catch_warnings(record=True) as messages:
            loss = (torch.nn.MSELoss() if kind == "regression" else torch.nn.CrossEntropyLoss())(
                output, y
            )
        arrays["loss"] = loss.detach().numpy()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite loss")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss.backward()
        for k, v in model.named_parameters():
            arrays["gradient/" + k] = v.grad.detach().numpy().copy()
        optimizer.step()
        for k, v in model.state_dict().items():
            arrays["updated/" + k] = v.detach().numpy().copy()
        for k, v in model.named_parameters():
            arrays["momentum/" + k] = optimizer.state[v]["momentum_buffer"].numpy().copy()
        warnings_observed = [str(w.message) for w in messages]
    except (IndexError, ValueError, RuntimeError) as exception:
        error = type(exception).__name__ + ": " + str(exception)
        warnings_observed = []
    arrays["final/rng"] = torch.get_rng_state().numpy().copy()
    return dict(
        loss_executable=error is None,
        task_output_shape_matches=tuple(output.shape) == (2, classes),
        output_shape=list(output.shape),
        error=error,
        warnings=warnings_observed,
    ), arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    target = FIXTURES / f"preset-compatibility-{args.label}"
    if target.exists():
        raise FileExistsError("Preserve previous controls; choose a fresh label")
    require_space(256 * 1024**2)
    target.mkdir(parents=True)
    torch.set_num_threads(1)
    output = EVIDENCE / f"preset-compatibility-{args.label}.json"
    report = dict(
        status="running",
        cases={},
        hardware=collect(),
        torch_version=torch.__version__,
        executor=str(Path(__file__).relative_to(ROOT)),
        executor_sha256=sha(Path(__file__)),
        dependencies={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                REQUIREMENTS / "native.lock",
                SCRIPTS / "reference_corrections.py",
                ASSETS / "v1/images/manifest.json",
            )
        },
        scope="Every supplied preset/dataset shape and seeds 0,42,123: original and corrected native builders, loss compatibility, initialization, forward, first gradients, SGD momentum/update and RNG. Synthetic two-sample controls; full-data/browser trajectories remain mandatory.",
    )
    try:
        for unit, states in STATES.items():
            source = ROOT / f"assignments/{unit}/app.py"
            original = builders(source.read_text())
            corrected = builders(corrected_source(unit))
            report["dependencies"][str(source.relative_to(ROOT))] = sha(source)
            for preset, architecture in original["PRESETS"].items():
                for dataset, state in states.items():
                    for seed in (0, 42, 123):
                        name = f"unit{unit}/{preset}/{dataset}/seed-{seed}"
                        expected, control = trajectory(original, architecture, state, seed)
                        observed, actual = trajectory(corrected, architecture, state, seed)
                        if expected != observed:
                            raise AssertionError((name, expected, observed))
                        if control.keys() != actual.keys():
                            raise AssertionError("Trajectory fields differ")
                        for key in control:
                            np.testing.assert_array_equal(control[key], actual[key])
                        path = target / (hashlib.sha256(name.encode()).hexdigest() + ".npz")
                        np.savez(path, **control)
                        report["cases"][name] = dict(
                            status="pass",
                            parent_state=dict(unit=unit, preset=preset, dataset=dataset, seed=seed),
                            assertion=dict(
                                kind="computation",
                                expected=expected,
                                observed=observed,
                                matched=True,
                            ),
                            trajectory=dict(
                                path=str(path.relative_to(ROOT)),
                                sha256=sha(path),
                                arrays=list(control),
                            ),
                        )
                print(unit, preset, "observed", flush=True)
        report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=str(error))
    write_json(output, report)
    print(report["status"], len(report["cases"]), "native combinations", flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
