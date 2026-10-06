"""Checked patch for the pinned Shinylive startup race; no app UI changes."""
import hashlib
import json
from common import PROOF, UNITS, sha, write_json

OLD = '''if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register(serviceWorkerPath, { type: "module" }).then((registration) => registration.update()).then(() => console.log("Service Worker registered")).catch(() => console.log("Service Worker registration failed"));
  navigator.serviceWorker.ready.then(() => {
    if (!navigator.serviceWorker.controller) {
      window.location.reload();
    }
  });
}
'''


def patch_runtime():
    loader = PROOF/'site/shinylive/load-shinylive-sw.js'
    source = loader.read_text()
    before = sha(loader)
    patch = (PROOF/'runtime/service-worker-ready.js').read_text()
    if source.count(OLD) != 1:
        raise ValueError('Pinned Shinylive service-worker loader changed')
    loader.write_text(source.replace(OLD,patch))
    client = PROOF/'site/shinylive/shinylive.js'
    client_source = client.read_text()
    old = '''  navigator.serviceWorker.addEventListener("message", (event) => {
    if (event.data.type === "serviceworkerStart") {
      createHttpRequestChannel(proxy, appName, urlPath);
    }
  });'''
    new = '''  navigator.serviceWorker.addEventListener("message", (event) => {
    if (event.data.type === "serviceworkerStart" && navigator.serviceWorker.controller) {
      createHttpRequestChannel(proxy, appName, urlPath);
    }
  });
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (navigator.serviceWorker.controller) {
      createHttpRequestChannel(proxy, appName, urlPath);
    }
  });'''
    if client_source.count(old)!=1:
        raise ValueError('Pinned Shinylive proxy channel setup changed')
    client_source = client_source.replace(old,new)
    start = client_source.index('function setupAppProxyPath(')
    end = client_source.index('function createHttpRequestChannel(', start)
    if hashlib.sha256(client_source[start:end].encode()).hexdigest() != 'cf5aa2378f15f71f0b2ab347916cc5e8fe96c23bc3416bd08495b0840d3fa77b':
        raise ValueError('Pinned proxy lifecycle changed')
    client_source = client_source[:start] + (PROOF/'runtime/proxy-lifecycle.js').read_text() + client_source[end:]
    # Worker import/startup errors otherwise leave postMessageAsync pending.
    old = '    this.pyWorker.onmessage = (e) => {'
    new = '    this.pyWorker.addEventListener("error", (event) => {\n      window.dispatchEvent(new CustomEvent("course:startup-error", {\n        detail: event.message || "Python worker failed to start."\n      }));\n      this.pyWorker.terminate();\n    });\n    window.addEventListener("course:startup-abort", () => this.pyWorker.terminate(), { once: true });\n    window.addEventListener("pagehide", () => this.pyWorker.terminate(), { once: true });\n    this.pyWorker.onmessage = (e) => {'
    if client_source.count(old) != 1:
        raise ValueError('Pinned Python worker constructor changed')
    client_source = client_source.replace(old, new)
    old = '    this.pyWorker.onmessage = (e) => {'
    new = '    window.courseSessionBridge?.registerDisposer(() => this.pyWorker.terminate());\n' + old
    if client_source.count(old) != 1:
        raise ValueError('Pinned Python worker disposal contract changed')
    client_source = client_source.replace(old, new)
    old = '    })().catch((e) => {\n      console.error(e);\n    });\n  }, [pyodideProxyHandlePromise2]);'
    new = old.replace('      console.error(e);', '      console.error(e);\n      window.dispatchEvent(new CustomEvent("course:startup-error", { detail: e }));')
    if client_source.count(old) != 1:
        raise ValueError('Pinned Python startup rejection handler changed')
    client_source = client_source.replace(old, new)
    old = '      status = { stage, error: nextError };'
    new = old + '\n      if (stage === "failed") window.dispatchEvent(new CustomEvent("course:startup-error", { detail: nextError }));'
    if client_source.count(old) != 1:
        raise ValueError('Pinned runtime load status store changed')
    client_source = client_source.replace(old, new)
    old = '    response = await fetch(url);'
    new = '    response = await fetch(url, { headers: { Range: "bytes=0-3" }, cache: "no-store" });'
    if client_source.count(old) != 1:
        raise ValueError('Pinned WASM signature guard changed')
    client_source = client_source.replace(old, new)
    client.write_text(client_source)
    worker = PROOF/'site/shinylive-sw.js'
    worker_source = worker.read_text()
    if not worker_source.startswith('// Shinylive 0.10.15\n') or 'course:claim-client' in worker_source:
        raise ValueError('Pinned Shinylive worker changed')
    start = worker_source.index('async function fetchASGI(')
    end = worker_source.index('function headersToASGI(', start)
    if hashlib.sha256(worker_source[start:end].encode()).hexdigest() != '734bf3e849218ee3ae193bb292e90430f720c6e032877f07f6b0fa62703670a8':
        raise ValueError('Pinned service worker HTTP implementation changed')
    worker_source = worker_source[:start] + (PROOF/'runtime/service-worker-http.js').read_text() + worker_source[end:]
    start = worker_source.index('function injectSocketFilter(')
    # This is the last function in the pinned upstream worker.
    original_filter = worker_source[start:]
    expected_filter = '''function injectSocketFilter(bodyChunk, response) {
  const contentType = response.headers.get("content-type");
  if (contentType && /^text\\/html(;|$)/.test(contentType)) {
    const bodyChunkStr = uint8ArrayToString(bodyChunk);
    const base_path = dirname(self.location.pathname);
    const newStr = bodyChunkStr.replace(
      /<\\/head>/,
      `<script src="${base_path}/shinylive-inject-socket.js" type="module"><\\/script>
</head>`
    );
    const newChunk = Uint8Array.from(
      newStr.split("").map((s) => s.charCodeAt(0))
    );
    return newChunk;
  }
  return bodyChunk;
}
'''
    if original_filter.rstrip() != expected_filter.rstrip():
        raise ValueError('Pinned HTML socket injection changed')
    worker_source = worker_source[:start] + (PROOF/'runtime/service-worker-html.js').read_text()
    replacements = {
        'var version = "v10";': 'var version = "course-__COURSE_RUNTIME_VERSION__";',
        'return key.indexOf(version + cacheName) !== 0;': 'return key.endsWith(cacheName) && key !== version + cacheName;',
        'const cachedResponse = await caches.match(request);': 'const cachedResponse = await (await caches.open(version + cacheName)).match(request);',
        'var apps = {};': 'var apps = {};\nvar courseAppOwners = {};\nvar courseRequests = new Map();\n' + (PROOF/'runtime/service-worker-proxy.js').read_text(),
        '    apps[path] = port;': '    courseConfigureProxy(event);',
        'const filter = isAppRoot ? injectSocketFilter : identityFilter;': 'const filter = isAppRoot ? createInjectSocketFilter() : identityFilter;',
        '''            referrer: request.referrer
          }),
          void 0,
          filter''': '''            referrer: request.referrer,
            signal: request.signal
          }),
          void 0,
          filter,
          m_appPath[1]''',
        '''  if (coiRequested) {
    event.respondWith(
      (async () => {
        const resp = await fetch(request);
        return addCoiHeaders(resp);
      })()
    );
  }
});''': '''  // Ordinary static downloads must not depend on the lifetime of this worker.
  // Entry/bootstrap HTTP responses revalidate; bootstrap URLs bind the build.
  // Intercept only requests that actually need cross-origin isolation headers.
  if (coiRequested) {
    event.respondWith((async () => {
      const resp = await fetch(request, { cache: "no-cache" });
      return addCoiHeaders(resp);
    })());
  }
});''',
    }
    for old, new in replacements.items():
        if worker_source.count(old) != 1:
            raise ValueError('Pinned Shinylive cache implementation changed')
        worker_source = worker_source.replace(old, new)
    worker.write_text(worker_source + '''
// Reclaim an uncontrolled document after reload without navigation loops.
self.addEventListener("message", event => {
  if (event.data?.type === "course:activate-runtime" && event.data.version === version) {
    event.waitUntil(self.skipWaiting());
  }
  if (event.data?.type === "course:claim-client") {
    event.waitUntil(self.clients.claim());
  }
  if (event.data?.type === "course:dispose-proxy" && courseAppOwners[event.data.path] === event.source?.id) {
    const path = event.data.path;
    for (const cancel of [...(courseRequests.get(path) || [])]) cancel();
    apps[path]?.close();
    delete apps[path];
    delete courseAppOwners[path];
  }
  if (event.data?.type === "course:diagnostics" && event.ports[0]) {
    event.waitUntil((async () => {
      const controlled = await self.clients.matchAll({type: "window", includeUncontrolled: false});
      event.ports[0].postMessage({version, requests: [...courseRequests].map(([path, pending]) => ({path, pending: pending.size})),
        applications: Object.keys(apps), owners: {...courseAppOwners},
        controlled: controlled.map(client => ({id: client.id, url: client.url}))});
      event.ports[0].close();
    })());
  }
});
''')
    for unit in UNITS:
        index = PROOF/f'site/unit{unit}/index.html'
        html = index.read_text()
        old = '      import { runExportedApp } from "./../shinylive/shinylive.js";'
        new = '''      const { serviceWorkerReady } = await import("./../shinylive/load-shinylive-sw.js");
      await serviceWorkerReady;
      const { runExportedApp } = await import("./../shinylive/shinylive.js");'''
        if html.count(old) != 1:
            raise ValueError('Pinned Shinylive exported entry changed')
        index.write_text(html.replace(old,new))
    write_json(PROOF/'evidence/runtime-patch.json', {'shinylive_assets':'0.10.15',
        'loader_original_sha256':before,'loader_patched_sha256':sha(loader),
        'patch_sha256':sha(PROOF/'runtime/service-worker-ready.js'),
        'reason':'Wait for the expected build controller and retry guarded activation of that waiting build; uninstrumented Edge regression reproduced and isolated candidate passed 30 consecutive affected sequences.'})


def session_startup_script(unit, version, source_id):
    """Identity and bridge precede all worker construction and startup observers."""
    identity = json.dumps(dict(unit=unit, build_id=version, source_id=source_id),
                          sort_keys=True, separators=(',', ':'))
    return ('window.courseAssignmentIdentity = Object.freeze(' + identity + ');\n' +
            (PROOF/'runtime/session-bridge.js').read_text() + '\n' +
            (PROOF/'runtime/startup-status.js').read_text())


def version_runtime():
    """Bind bootstrap/worker URLs to deployed content, including pinned packages."""
    site = PROOF/'site'
    paths = [site/'shinylive/load-shinylive-sw.js', site/'shinylive/shinylive.js',
             site/'shinylive-sw.js', site/'shinylive/pyodide/pyodide-lock.json',
             PROOF/'assets.lock.json', PROOF/'runtime/startup-status.js',
             PROOF/'runtime/session-bridge.js',
             site/'shinylive/course-training-worker.js',
             site/'assets/v1/neural-runtime/manifest.json',
             *[site/f'unit{unit}/app.json' for unit in UNITS]]
    version = hashlib.sha256(''.join(sha(path) for path in paths).encode()).hexdigest()[:20]
    for path in (paths[0], paths[2]):
        source = path.read_text()
        if source.count('__COURSE_RUNTIME_VERSION__') != 1:
            raise ValueError('Runtime version placeholder missing or duplicated')
        path.write_text(source.replace('__COURSE_RUNTIME_VERSION__', version))
    for unit in UNITS:
        path = site/f'unit{unit}/index.html'
        html = path.read_text()
        for asset in ('load-shinylive-sw.js', 'shinylive.js'):
            html = html.replace(f'/shinylive/{asset}"', f'/shinylive/{asset}?v={version}"')
        # Dynamic imports allow a visible recovery message instead of a blank page.
        status_script = session_startup_script(unit, version, sha(site/f'unit{unit}/app.json'))
        html = html.replace('      const { serviceWorkerReady }', status_script + '\n      try {\n      const { serviceWorkerReady }', 1)
        html = html.replace('      runExportedApp({', '      await runExportedApp({', 1)
        html = html.replace('      });\n    </script>', '''      });
      } catch (error) {
        console.error(error);
        startupFailed(error);
      }
    </script>''', 1)
        html = html.replace('id="root"></div>', 'id="root"></div>' +
            f'<section id="course-startup" hidden role="status" style="position:fixed;inset:0;z-index:10000;background:white;padding:2rem;font:16px system-ui">'
            f'<p>Loading Assignment {unit}… The first load may take a moment.</p>'
            '<button type="button" hidden>Reload to retry</button></section>')
        path.write_text(html)
    write_json(PROOF/'evidence/runtime-version.json', {'version':version, 'inputs':{str(path.relative_to(PROOF)):sha(path) for path in paths}})
    write_json(site/'assignment-sessions.json', dict(protocol=1, build_id=version,
        sources={str(unit):sha(site/f'unit{unit}/app.json') for unit in UNITS}))


if __name__ == '__main__':
    patch_runtime()
