import socket
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from owned_preview import start


class OwnedPreviewTests(unittest.TestCase):
    def test_occupied_origin_is_rejected_before_a_second_process_launches(self):
        with tempfile.TemporaryDirectory() as folder,socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            with patch('owned_preview.subprocess.Popen') as launch:
                with self.assertRaises(OSError):start(Path(folder),listener.getsockname()[1])
                launch.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_readiness_verifies_the_new_process_static_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'index.html').write_text('isolated owned static artifact')
            server,identity=start(root,0)
            try:
                self.assertIsNone(server.poll())
                with urllib.request.urlopen(f"http://127.0.0.1:{identity['port']}/{identity['marker']}") as response:
                    self.assertEqual(response.read().decode(),identity['identity'])
                with urllib.request.urlopen(f"http://127.0.0.1:{identity['port']}/") as response:
                    self.assertEqual(response.read().decode(),'isolated owned static artifact')
            finally:server.terminate();server.wait(timeout=10)


if __name__=='__main__':unittest.main()
