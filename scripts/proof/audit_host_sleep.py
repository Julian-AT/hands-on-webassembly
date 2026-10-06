"""Bind failed diagnostics to retained native sleep/wake observations."""

from repository import EVIDENCE
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from common import PROOF, sha, write_json


def power_events(text):
    events = []
    for line in text.splitlines():
        match = re.match(
            r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4}) (Sleep|Wake|DarkWake)\s+(.+)", line
        )
        if match:
            stamp = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S %z")
            events.append(
                dict(
                    utc=stamp.astimezone(timezone.utc).isoformat(),
                    timestamp=stamp.timestamp(),
                    type=match[2],
                    detail=match[3].strip(),
                )
            )
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = EVIDENCE / f"host-sleep-audit-{args.label}.json"
    if output.exists():
        raise FileExistsError("Retain prior evidence; choose another label")
    log = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, check=True).stdout
    events = power_events(log)
    if not events:
        raise ValueError("No native sleep/wake observations found")
    # Retain only power transitions, avoiding unrelated application activity.
    names = [p.name for p in sorted((EVIDENCE).glob("generation-*-all-loading-oct04.json"))]
    names += [
        "unit6-loading-ui-chrome-owned-context-oct04.json",
        "generation-diagnostic-chrome-unit1-actual-generations-oct04.json",
    ]
    observations = []
    for name in names:
        path = EVIDENCE / name
        report = json.loads(path.read_text())
        end = path.stat().st_mtime
        preceding = [event for event in events if event["timestamp"] <= end]
        nearest = preceding[-1] if preceding else None
        item = dict(
            report="evidence/" + name,
            report_sha256=sha(path),
            observed_status=report["status"],
            report_file_modified_utc=datetime.fromtimestamp(end, timezone.utc).isoformat(),
            preceding_power_transition=nearest,
            seconds_after_transition=None if nearest is None else end - nearest["timestamp"],
        )
        if "cycle_evidence" in report:
            failed = report["cycle_evidence"][-1]
            lifecycle = failed.get("worker_lifecycle", [])
            times = [
                v["scriptResponseTime"]
                for event in lifecycle
                for v in event.get("versions", [])
                if v.get("scriptResponseTime")
            ]
            if times:
                latest = max(times)
                sleep = [
                    e for e in events if e["type"] == "Sleep" and latest < e["timestamp"] <= end
                ]
                item.update(
                    latest_worker_script_response_utc=datetime.fromtimestamp(
                        latest, timezone.utc
                    ).isoformat(),
                    sleep_between_worker_start_and_failure=sleep,
                )
        observations.append(item)
    write_json(
        output,
        dict(
            status="pass",
            executor="scripts/proof/audit_host_sleep.py",
            executor_sha256=sha(Path(__file__)),
            power_events=events,
            observations=observations,
            interpretation="The detailed startup trace overlaps native Clamshell Sleep after activation. Earlier failed generation and download reports were written immediately after DarkWake during an extended sleep interval. This establishes interrupted host conditions; it does not convert failures to passing evidence or establish post-wake recovery.",
            limits="Earlier uninstrumented reports did not record operation start times. File modification time is an observation, not a claim that every earlier waiting-worker stall has the same cause.",
            required_next="Preserve failures; record sleep transitions in new acceptance runs; reject interrupted sequences; verify recovery and complete fresh affected consecutive sequences on an awake host.",
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
