"""Native power observations outside application/browser instrumentation."""

import math
import plistlib
import subprocess
import sys
import time
from audit_host_sleep import power_events


def capability_state(capabilities):
    # Apple's IOPM root-domain capabilities: CPU=1, Graphics=2, Audio=4.
    if not isinstance(capabilities, int) or isinstance(capabilities, bool) or capabilities < 0:
        return "Unknown"
    if capabilities & (2 | 4):
        return "FullWake"
    return "DarkWake" if capabilities & 1 else "Sleep"


def snapshot():
    started = time.time()
    if sys.platform != "darwin":
        raise RuntimeError("A native power-state collector is required on this host platform")
    events = power_events(
        subprocess.run(
            ["pmset", "-g", "log"], capture_output=True, text=True, check=True, timeout=15
        ).stdout
    )
    if not events:
        raise RuntimeError("Native power transitions were not available")
    roots = plistlib.loads(
        subprocess.run(
            ["ioreg", "-a", "-r", "-c", "IOPMrootDomain", "-d1"],
            capture_output=True,
            check=True,
            timeout=15,
        ).stdout
    )
    if len(roots) != 1:
        raise RuntimeError("A unique native power root was not available")
    capabilities = roots[0].get("System Capabilities")
    state = capability_state(capabilities)
    if state == "Unknown":
        raise RuntimeError("Native power capabilities were not available")
    # Retain only power observations; IORegistry contains unrelated device identifiers.
    collected = time.time()
    return dict(
        started_at=started,
        collected_at=collected,
        events=events,
        native_power=dict(
            system_capabilities=capabilities,
            state=state,
            clamshell_closed=roots[0].get("AppleClamshellState"),
            recorded_at=collected,
        ),
    )


def interruptions(before, after):
    if after["collected_at"] < before["started_at"]:
        raise ValueError("Host clock moved backwards during verification")
    return [
        event
        for event in after["events"]
        if event["type"] == "Sleep"
        and before["started_at"] <= event["timestamp"] <= after["collected_at"]
    ]


def preceding(events, timestamp):
    return next((event for event in reversed(events) if event["timestamp"] <= timestamp), None)


def condition_errors(conditions):
    """Recheck the retained observations, rather than trusting eligibility booleans."""
    if not isinstance(conditions, dict):
        return ["Native host observations missing"]
    errors = []
    start, finish = conditions.get("started_at"), conditions.get("finished_at")
    if (
        any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in (start, finish)
        )
        or finish < start
    ):
        return ["Native host observation interval invalid"]
    if (
        conditions.get("uninterrupted") is not True
        or conditions.get("sleep_interruptions") != []
        or conditions.get("nonawake_transitions") != []
    ):
        errors.append("Native host interval includes a nonawake transition")
    transitions = conditions.get("power_transitions")
    if not isinstance(transitions, list) or any(
        not isinstance(event, dict)
        or event.get("type") != "Wake"
        or not isinstance(event.get("timestamp"), (int, float))
        or not start <= event["timestamp"] <= finish
        for event in transitions
    ):
        errors.append("Native host interval transition evidence missing or nonawake")
    for boundary, timestamp in (("start", start), ("finish", finish)):
        event = conditions.get(f"{boundary}_power_transition")
        native = conditions.get(f"{boundary}_native_power", {})
        collected = conditions.get(f"{boundary}_collection", {})
        valid_times = (
            isinstance(collected, dict)
            and all(
                isinstance(collected.get(key), (int, float))
                and not isinstance(collected[key], bool)
                and math.isfinite(collected[key])
                for key in ("started_at", "collected_at")
            )
            and collected["started_at"] <= collected["collected_at"]
            and collected["started_at"] >= start
            and collected["collected_at"] <= finish
        )
        if (
            not isinstance(event, dict)
            or event.get("type") != "Wake"
            or not isinstance(event.get("timestamp"), (int, float))
            or event["timestamp"] > timestamp
            or not isinstance(native, dict)
            or capability_state(native.get("system_capabilities")) != "FullWake"
            or native.get("state") != "FullWake"
            or not valid_times
            or native.get("recorded_at") != collected.get("collected_at")
        ):
            errors.append(f"Native FullWake observations at {boundary} are not established")
    first, last = conditions.get("start_collection", {}), conditions.get("finish_collection", {})
    if (
        not isinstance(first, dict)
        or not isinstance(last, dict)
        or first.get("started_at") != start
        or last.get("collected_at") != finish
        or not isinstance(first.get("collected_at"), (int, float))
        or not isinstance(last.get("started_at"), (int, float))
        or first["collected_at"] > last["started_at"]
    ):
        errors.append("Native boundary collection order is not established")
    if any(
        conditions.get(key) is not True
        for key in (
            "awake_at_start",
            "awake_at_finish",
            "awake_throughout",
            "certification_eligible",
        )
    ):
        errors.append("Awake host certification is not established")
    if not conditions.get("measurement"):
        errors.append("Native host measurement description missing")
    return errors


def apply(report, before, after):
    sleeps = interruptions(before, after)
    transitions = [
        event
        for event in after["events"]
        if before["started_at"] <= event["timestamp"] <= after["collected_at"]
    ]
    nonawake = [event for event in transitions if event["type"] in ("Sleep", "DarkWake")]
    first = preceding(before["events"], before["started_at"])
    last = preceding(after["events"], after["collected_at"])
    first_native, last_native = before.get("native_power", {}), after.get("native_power", {})
    awake_start = bool(
        first
        and first["type"] == "Wake"
        and capability_state(first_native.get("system_capabilities")) == "FullWake"
    )
    awake_finish = bool(
        last
        and last["type"] == "Wake"
        and capability_state(last_native.get("system_capabilities")) == "FullWake"
    )
    eligible = awake_start and awake_finish and not nonawake
    conditions = dict(
        started_at=before["started_at"],
        finished_at=after["collected_at"],
        preceding_power_transition=first,
        start_power_transition=first,
        finish_power_transition=last,
        start_native_power=first_native,
        finish_native_power=last_native,
        start_collection={key: before.get(key) for key in ("started_at", "collected_at")},
        finish_collection={key: after.get(key) for key in ("started_at", "collected_at")},
        power_transitions=transitions,
        sleep_interruptions=sleeps,
        nonawake_transitions=nonawake,
        uninterrupted=not sleeps,
        awake_at_start=awake_start,
        awake_at_finish=awake_finish,
        awake_throughout=eligible,
        certification_eligible=eligible,
        measurement="Native macOS pmset sleep/wake transitions and filtered IORegistry root-domain System Capabilities at both boundaries, collected outside the browser sequence; no application or lifecycle hooks.",
    )
    conditions["eligibility_errors"] = condition_errors(conditions)
    if conditions["eligibility_errors"]:
        conditions["certification_eligible"] = False
    report["host_conditions"] = conditions
    if report.get("status") == "pass" and not conditions["certification_eligible"]:
        report.update(observed_status="pass", status="interrupted" if nonawake else "inactive-host")
    return report
