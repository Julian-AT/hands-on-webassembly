"""Stable runtime choice identities and complete parent-state matrix validation."""

import hashlib
import json
from urllib.parse import quote
from assertions import assertion_matches


def parent_key(state):
    return json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)


def choice_cases(unit, identifier, parent_state, choices):
    parent = hashlib.sha256(parent_key(parent_state).encode()).hexdigest()[:20]
    return [
        dict(
            id=f"unit{unit}/control/{quote(identifier, safe='-_:')}/parent-{parent}/choice-{quote(str(choice), safe='-_')}",
            unit=unit,
            identifier=identifier,
            parent_state=parent_state,
            scenario="choice-" + str(choice),
            choice=str(choice),
        )
        for choice in choices
    ]


def validate_matrix(matrix, unit, controls, checks, assertion_kinds=None):
    assertion_kinds = assertion_kinds or {
        "value",
        "content",
        "computation",
        "workflow",
        "recoverable-error",
        "download-roundtrip",
    }
    errors = []
    if (
        not isinstance(matrix, dict)
        or matrix.get("status") != "pass"
        or matrix.get("unresolved") != []
    ):
        return ["Dynamic parent-state choice matrix is unresolved"]
    records = matrix.get("controls", {})
    if not isinstance(records, dict):
        return ["Dynamic controls missing"]
    for identifier in controls:
        record = records.get(identifier, {})
        if not isinstance(record, dict):
            errors.append(f"{identifier}: dynamic control record invalid")
            continue
        expected = record.get("expected_parent_states")
        observations = record.get("observations")
        if (
            not isinstance(expected, list)
            or not expected
            or any(not isinstance(state, dict) for state in expected)
            or not isinstance(observations, list)
        ):
            errors.append(f"{identifier}: relevant parent states not specified")
            continue
        observed_states = []
        for observation in observations:
            if (
                not isinstance(observation, dict)
                or observation.get("status") != "pass"
                or not isinstance(observation.get("parent_state"), dict)
                or not isinstance(observation.get("choices"), list)
                or observation.get("reference_scope") != "corrected-native"
                or not observation.get("reference_fingerprint")
                or not observation.get("executor")
                or not observation.get("dependencies")
            ):
                errors.append(f"{identifier}: corrected native discovery evidence missing")
                continue
            observed_states.append(parent_key(observation["parent_state"]))
            if len(observation["choices"]) != len(
                set(str(choice) for choice in observation["choices"])
            ):
                errors.append(f"{identifier}: duplicate choice identities")
            for case in choice_cases(
                unit, identifier, observation["parent_state"], observation["choices"]
            ):
                assertion = checks.get(case["id"], {}).get("assertion", {})
                if (
                    not isinstance(assertion, dict)
                    or not assertion_matches(assertion)
                    or assertion.get("kind") not in assertion_kinds
                    or not {"expected", "observed"} <= assertion.keys()
                ):
                    errors.append(f"{case['id']}: discovered behavior has not been executed")
        wanted = [parent_key(state) for state in expected]
        if sorted(wanted) != sorted(observed_states) or len(wanted) != len(set(wanted)):
            errors.append(f"{identifier}: incomplete or duplicate parent-state matrix")
    return errors
