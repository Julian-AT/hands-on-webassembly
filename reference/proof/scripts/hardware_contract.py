"""Explicit certification device and native observations, without device identifiers."""
import json
import subprocess
from common import PROOF, sha


def contract():
    path = PROOF / 'hardware-contract.json'
    value = json.loads(path.read_text())
    return value, sha(path)


def collect():
    value, digest = contract()
    def sysctl(name):
        return subprocess.check_output(['sysctl', '-n', name], text=True).strip()
    displays = json.loads(subprocess.check_output(
        ['system_profiler', 'SPDisplaysDataType', '-json'], text=True))['SPDisplaysDataType']
    return dict(contract_id=value['id'], contract_sha256=digest, machine='MacBook Pro',
        model=sysctl('hw.model'), cpu=sysctl('machdep.cpu.brand_string'),
        physical_ram_bytes=int(sysctl('hw.memsize')),
        integrated_graphics=any(d.get('sppci_bus') == 'spdisplays_builtin' and
                                d.get('sppci_model') == value['active']['cpu'] for d in displays),
        measurement='Native sysctl hw.model/hw.memsize/machdep.cpu.brand_string and system_profiler SPDisplaysDataType')


def errors(observed):
    value, digest = contract()
    if not isinstance(observed, dict):
        return ['Physical hardware observations missing']
    failures = [f'Certification hardware {key} differs from active contract'
                for key, expected in value['active'].items() if observed.get(key) != expected]
    if observed.get('contract_id') != value['id'] or observed.get('contract_sha256') != digest:
        failures.append('Hardware contract identity or checksum missing/stale')
    if not observed.get('measurement'):
        failures.append('Native hardware measurement missing')
    return failures
