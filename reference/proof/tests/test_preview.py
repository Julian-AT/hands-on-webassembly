"""Real HTTP regression: assignment URLs without a slash must load their assets."""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from urllib.parse import urljoin
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = tempfile.TemporaryDirectory()
        root = Path(cls.files.name)
        (root / 'index.html').write_text('Preview')
        for unit in range(1, 8):
            directory = root / f'unit{unit}'
            directory.mkdir()
            (directory / 'index.html').write_text('<script>fetch("./app.json")</script>')
            (directory / 'app.json').write_text(json.dumps({'unit': unit}))
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            cls.port = reservation.getsockname()[1]
        cls.server = subprocess.Popen(
            ['node', str(ROOT / 'tools/preview.mjs'), str(root)],
            env={**os.environ, 'PORT': str(cls.port)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        message = cls.server.stdout.readline()
        if not message.startswith('Static preview:'):
            cls.server.terminate()
            cls.server.wait(timeout=5)
            cls.files.cleanup()
            raise RuntimeError(f'Preview did not start: {message}')
        cls.base = f'http://127.0.0.1:{cls.port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=5)
        cls.server.stdout.close()
        cls.files.cleanup()

    def test_all_units_redirect_get_and_head_preserving_query(self):
        for unit in range(1, 8):
            for method in ('GET', 'HEAD'):
                with self.subTest(unit=unit, method=method):
                    connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
                    try:
                        connection.request(method, f'/unit{unit}?example=1&mode=demo')
                        response = connection.getresponse()
                        self.assertEqual(response.status, 308)
                        self.assertEqual(response.getheader('Location'), f'/unit{unit}/?example=1&mode=demo')
                        self.assertEqual(response.read(), b'')
                    finally:
                        connection.close()

    def test_relative_app_data_resolves_after_redirect(self):
        for unit in range(1, 8):
            with self.subTest(unit=unit):
                with urlopen(f'{self.base}/unit{unit}', timeout=5) as response:
                    self.assertEqual(response.geturl(), f'{self.base}/unit{unit}/')
                    data_url = urljoin(response.geturl(), './app.json')
                with urlopen(data_url, timeout=5) as response:
                    self.assertEqual(json.load(response), {'unit': unit})

    def test_canonical_directory_does_not_redirect(self):
        with urlopen(f'{self.base}/unit5/', timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn('text/html', response.headers['Content-Type'])


if __name__ == '__main__':
    unittest.main()
