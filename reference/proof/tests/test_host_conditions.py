import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_host_sleep import power_events
from host_conditions import apply,interruptions


class HostConditionsTests(unittest.TestCase):
    def test_native_transition_parser_retains_timezone_not_related_activity(self):
        events=power_events('2026-10-04 18:25:45 +0200 Sleep               Entering Sleep state due to Clamshell Sleep\n2026-10-04 18:25:45 +0200 WakeDetails DriverReason\n2026-10-04 18:40:45 +0200 DarkWake            DarkWake from Deep Idle\n')
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]['utc'],'2026-10-04T16:25:45+00:00')
        self.assertEqual(events[1]['type'],'DarkWake')

    def test_sleep_interruption_cannot_certify_a_passing_sequence(self):
        before=dict(started_at=10,events=[])
        after=dict(collected_at=30,events=[dict(type='Sleep',timestamp=9),dict(type='Sleep',timestamp=20),dict(type='Sleep',timestamp=31)])
        report=apply(dict(status='pass'),before,after)
        self.assertEqual(report['status'],'interrupted')
        self.assertEqual(report['observed_status'],'pass')
        self.assertEqual(report['host_conditions']['sleep_interruptions'],[dict(type='Sleep',timestamp=20)])
        failed=apply(dict(status='fail'),before,after)
        self.assertEqual(failed['status'],'fail')
        self.assertFalse(failed['host_conditions']['uninterrupted'])

    def test_prior_sleep_and_darkwake_do_not_invent_a_new_interruption(self):
        before=dict(started_at=10,events=[dict(type='Sleep',timestamp=9)])
        after=dict(collected_at=30,events=[dict(type='Sleep',timestamp=9),dict(type='DarkWake',timestamp=20)])
        report=apply(dict(status='pass'),before,after)
        self.assertEqual(report['status'],'interrupted')
        self.assertEqual(report['host_conditions']['sleep_interruptions'],[])
        self.assertFalse(report['host_conditions']['certification_eligible'])

    def test_backwards_clock_blocks_condition_evidence(self):
        with self.assertRaises(ValueError):interruptions(dict(started_at=30),dict(collected_at=20,events=[]))


if __name__=='__main__':unittest.main()
