import copy
import math
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from host_conditions import apply, condition_errors, capability_state


def boundary(start, finish, events, capabilities=15):
    return dict(started_at=start, collected_at=finish, events=events,
        native_power=dict(system_capabilities=capabilities, state=capability_state(capabilities),
            recorded_at=finish, clamshell_closed=False))


class AwakeHostTests(unittest.TestCase):
    def awake(self):
        wake = dict(type='Wake', timestamp=9)
        return apply(dict(status='pass'), boundary(10, 11, [wake]), boundary(29, 30, [wake]))

    def test_complete_fullwake_boundaries_certify_the_interval(self):
        report = self.awake()
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(condition_errors(report['host_conditions']), [])

    def test_existing_sleep_needs_no_new_sleep_to_block_certification(self):
        sleep = dict(type='Sleep', timestamp=9)
        report = apply(dict(status='pass'), boundary(10, 11, [sleep], 9), boundary(29, 30, [sleep], 9))
        self.assertEqual(report['status'], 'inactive-host')
        self.assertEqual(report['observed_status'], 'pass')
        self.assertEqual(report['host_conditions']['sleep_interruptions'], [])
        self.assertTrue(report['host_conditions']['uninterrupted'])
        self.assertFalse(report['host_conditions']['certification_eligible'])

    def test_darkwake_inside_interval_blocks_even_without_sleep(self):
        wake = dict(type='Wake', timestamp=9)
        dark = dict(type='DarkWake', timestamp=20)
        awake = dict(type='Wake', timestamp=25)
        report = apply(dict(status='pass'), boundary(10, 11, [wake]), boundary(29, 30, [wake, dark, awake]))
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['host_conditions']['sleep_interruptions'], [])
        self.assertEqual(report['host_conditions']['nonawake_transitions'], [dark])

    def test_fullwake_during_run_cannot_repair_nonawake_start(self):
        sleep = dict(type='Sleep', timestamp=9)
        wake = dict(type='Wake', timestamp=20)
        report = apply(dict(status='pass'), boundary(10, 11, [sleep], 9), boundary(29, 30, [sleep, wake]))
        self.assertEqual(report['status'], 'inactive-host')

    def test_direct_capability_observation_can_contradict_wake_log(self):
        wake = dict(type='Wake', timestamp=9)
        report = apply(dict(status='pass'), boundary(10, 11, [wake], 9), boundary(29, 30, [wake]))
        self.assertEqual(report['status'], 'inactive-host')

    def test_missing_boundaries_cannot_certify_legacy_boolean_claims(self):
        condition = self.awake()['host_conditions']
        del condition['start_native_power']
        self.assertTrue(condition_errors(condition))

    def test_forged_booleans_do_not_hide_nonawake_native_observations(self):
        condition = self.awake()['host_conditions']
        condition['start_power_transition']['type'] = 'Sleep'
        self.assertTrue(condition_errors(condition))
        condition = self.awake()['host_conditions']
        condition['finish_native_power']['system_capabilities'] = 9
        self.assertTrue(condition_errors(condition))

    def test_absent_transition_and_nonfinite_clocks_fail_closed(self):
        condition = self.awake()['host_conditions']
        condition['power_transitions'] = None
        self.assertTrue(condition_errors(condition))
        for invalid in (math.nan, math.inf, True):
            condition = self.awake()['host_conditions']
            condition['started_at'] = invalid
            self.assertTrue(condition_errors(condition))

    def test_native_collection_times_cannot_be_outside_or_inverted(self):
        condition = self.awake()['host_conditions']
        condition['finish_collection']['started_at'] = 8
        self.assertTrue(condition_errors(condition))

    def test_failed_application_observation_stays_failed(self):
        sleep = dict(type='Sleep', timestamp=9)
        report = apply(dict(status='fail'), boundary(10, 11, [sleep], 9), boundary(29, 30, [sleep], 9))
        self.assertEqual(report['status'], 'fail')
        self.assertFalse(report['host_conditions']['certification_eligible'])


if __name__ == '__main__':
    unittest.main()
