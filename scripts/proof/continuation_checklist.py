"""One requirement-linked checklist; missing execution remains explicitly blocked."""

from repository import EVIDENCE
import json
from common import PROOF, UNITS, write_json, sha
from coverage_contract import requirements
from case_inventory import identities
from dynamic_choices import choice_cases
from hardware_contract import contract


def main():
    inventory = json.loads((EVIDENCE / "source-inventory.json").read_text())
    rows = requirements(inventory)
    stable = [
        case for unit in UNITS for case in identities(unit, inventory[str(unit)]["interface"])
    ]
    discovery_path = EVIDENCE / "native-choice-discovery.json"
    if discovery_path.exists():
        discovery = json.loads(discovery_path.read_text())
        for record in discovery.get("controls", {}).values():
            for observation in record.get("observations", []):
                if observation.get("status") != "pass":
                    continue
                for case in choice_cases(
                    record["unit"],
                    record["identifier"],
                    observation["parent_state"],
                    observation["choices"],
                ):
                    stable.append(
                        dict(
                            case,
                            kind="control",
                            requires_runtime_expansion=False,
                            discovery_report="evidence/native-choice-discovery.json",
                        )
                    )
    gate_path = EVIDENCE / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    coverage = gate.get("coverage", {})
    suites = [
        dict(
            row,
            status=coverage.get(row["id"], {}).get("status", "missing"),
            cases=[
                dict(id=name, report=row["report"], section=row["section"])
                for name in row["required_cases"]
            ],
        )
        for row in rows
    ]
    behaviors = []
    for case in stable:
        bindings = [row["id"] for row in rows if case["id"] in row["required_cases"]]
        behaviors.append(
            dict(
                case,
                requirements=bindings,
                executor=None,
                dependencies={},
                expected=None,
                observed=None,
                status="blocked",
                reason="Dynamic discovery and executor pending"
                if case["requires_runtime_expansion"]
                else "Behavior executor pending",
            )
        )
    record = dict(
        status="blocked",
        release_suites=suites,
        behavior_cases=behaviors,
        hardware_contract=contract()[0],
        hardware_contract_sha256=contract()[1],
        source_inventory_sha256=sha(EVIDENCE / "source-inventory.json"),
        gate_sha256=sha(gate_path) if gate_path.exists() else None,
        scope="Every retained release requirement and stable behavior is linked. Entries describe obligations, never inferred test passes.",
    )
    write_json(EVIDENCE / "continuation-checklist.json", record)
    lines = [
        "# Continuation checklist",
        "",
        "Order: shared blockers, Units 1–7 completion, then certification. "
        "The [machine-readable checklist](evidence/continuation-checklist.json) binds each requirement to its report and assertion case.",
        "",
        "The preserved [original manifest](continuation-baseline/manifest.json) is retained. "
        "The [corrected manifest](continuation-baseline/manifest.corrected.json) binds all preserved copies and unchanged inputs. "
        "The [verification report](evidence/continuation-baseline-verification.json) records the four discovered binding inconsistencies.",
        "",
        "Accepted t-SNE and Small CNN arithmetic remain verified within their documented scopes. "
        "Further presets, configurations, UI behavior and release certification remain required.",
        "",
        "Certification uses the authorized 24 GB M4 Pro MacBook. Legacy 8 GB IDs remain stable with explicit supersession in [the hardware contract](hardware-contract.json). Safari Remote Automation is authorized for testing and must be restored afterward.",
        "",
        "## Release suites",
        "",
    ]
    lines.extend(
        f"- [ ] `{row['id']}` — {row['status']}; [required evidence](evidence/{row['report']}); {len(row['required_cases'])} cases"
        for row in suites
    )
    lines.extend(
        [
            "",
            "## Stable and discovered behaviors",
            "",
            "Each entry still needs a concrete executor, dependency hashes, expected behavior and observed assertion. "
            "Dynamic controls additionally need discovery under all relevant parent states. Legacy IDs remain required.",
            "",
        ]
    )
    lines.extend(
        f"- [ ] `{case['id']}` — {case['reason']}; release suites: "
        + ", ".join(f"`{name}`" for name in case["requirements"])
        for case in behaviors
    )
    (EVIDENCE / "continuation-checklist.md").write_text("\n".join(lines) + "\n")
    print(len(rows), "release suites;", len(stable), "stable behaviors linked")


if __name__ == "__main__":
    main()
