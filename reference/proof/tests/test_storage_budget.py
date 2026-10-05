import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from collections import namedtuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from storage_budget import require_space, RESERVE_BYTES

Usage = namedtuple('Usage', 'total used free')


class StorageBudgetTests(unittest.TestCase):
    def test_next_run_footprint_is_reserved(self):
        with patch('storage_budget.shutil.disk_usage', return_value=Usage(100,0,RESERVE_BYTES+99)):
            with self.assertRaises(OSError): require_space(100)
        with patch('storage_budget.shutil.disk_usage', return_value=Usage(100,0,RESERVE_BYTES+100)):
            self.assertEqual(require_space(100)['required_bytes'],RESERVE_BYTES+100)

    def test_unknown_or_negative_footprint_rejected(self):
        for value in (-1,None,'100'):
            with self.assertRaises(ValueError): require_space(value)


if __name__ == '__main__': unittest.main()
