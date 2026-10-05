"""Refuse occupied origins and verify a private static root before browser work."""
import os
import socket
import subprocess
import time
import urllib.request
import uuid
from common import ROOT


def start(site,port, *, dynamic_root=False):
    site=site.absolute() if dynamic_root else site.resolve()
    with socket.socket() as reservation:
        # This check catches existing listeners before an HTTP response from
        # another process could be mistaken for our newly launched preview.
        reservation.bind(('127.0.0.1',port))
        port=reservation.getsockname()[1]
    identity=uuid.uuid4().hex
    marker=site/('.course-preview-'+identity+'.txt')
    marker.write_text(identity)
    server=subprocess.Popen(['node',str(ROOT/'tools/preview.mjs'),str(site)],
        env=dict(os.environ,PORT=str(port)),stdout=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if server.poll() is not None:raise RuntimeError('Owned preview process exited before readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/{marker.name}',timeout=1) as response:
                    received=response.read().decode()
                if received!=identity:raise RuntimeError('Origin is serving a different static root')
                if server.poll() is not None:raise RuntimeError('Owned preview process exited during readiness')
                return server,dict(port=port,pid=server.pid,identity=identity,marker=marker.name)
            except urllib.error.HTTPError as error:
                if error.code==404:raise RuntimeError('Origin is serving a different static root') from error
                raise
            except (ConnectionError,urllib.error.URLError):time.sleep(.05)
        raise TimeoutError('Owned preview failed to bind its isolated origin')
    except BaseException:
        server.terminate();server.wait(timeout=10)
        marker.unlink(missing_ok=True)
        raise
