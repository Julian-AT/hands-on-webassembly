"""Recovery UI must outlive React mounting and await session initialization."""
import subprocess
import unittest
from pathlib import Path


class StartupStatusTests(unittest.TestCase):
    def test_connect_failure_timeout_and_retry(self):
        script = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const source = fs.readFileSync('proof/runtime/startup-status.js', 'utf8');
function setup() {
  const events = new Map(); let poll, timeout, initialized, reloads = 0, aborts = 0;
  const message = {textContent: 'Loading'};
  const retry = {hidden: true};
  const panel = {hidden: false, querySelector: tag => tag === 'p' ? message : retry, setAttribute: () => {}};
  const app = {isConnected: () => true, $socket: {readyState: 1}, $initialInput: {dataset: 'iris'}};
  const frame = {contentWindow: {Shiny: {initializedPromise: {then: fn => initialized = fn}, shinyapp: app}}};
  const context = {document: {getElementById: () => panel, querySelector: () => frame},
    window: {addEventListener: (name, fn) => events.set(name, fn), dispatchEvent: event => {if(event.type==='course:startup-abort')aborts++}},
    location: {reload: () => reloads++}, Event,
    setInterval: fn => {poll=fn;return 1}, clearInterval: () => {},
    setTimeout: fn => {timeout=fn;return 2}, clearTimeout: () => {}};
  vm.runInNewContext(source, context);
  return {panel, message, retry, events, frame, app, poll: () => poll(), connect: () => initialized(),
    timeout: () => timeout(), reloads: () => reloads, aborts: () => aborts};
}
const loaded = setup();
loaded.poll(); assert.equal(loaded.panel.hidden, false);
loaded.connect(); assert.equal(loaded.panel.hidden, true);
loaded.events.get('error')({message: 'Later unrelated plot error'});
assert.equal(loaded.panel.hidden, true); assert.equal(loaded.aborts(), 0);
const opening = setup(); opening.app.$socket.readyState = 0;
opening.poll(); opening.connect(); assert.equal(opening.panel.hidden, false);
opening.app.$socket.readyState = 1; opening.app.$initialInput = {};
opening.poll(); assert.equal(opening.panel.hidden, false);
opening.app.$initialInput = {dataset: 'iris'}; opening.poll(); assert.equal(opening.panel.hidden, true);
const disconnected = setup(); disconnected.app.isConnected = () => false;
disconnected.poll(); disconnected.connect(); assert.equal(disconnected.panel.hidden, false);
const abandoned = setup(); abandoned.poll();
abandoned.events.get('course:session-dispose')(); abandoned.connect();
assert.equal(abandoned.panel.hidden, false, 'Disposed sessions cannot publish readiness');
const replaced = setup(); replaced.poll(); const obsolete = replaced.connect;
replaced.frame.contentWindow.Shiny = {initializedPromise: {then: () => {}}, shinyapp: replaced.app};
obsolete(); assert.equal(replaced.panel.hidden, false, 'Old initialized promises cannot certify new documents');
for (const type of ['error', 'course:startup-error', 'unhandledrejection', 'timeout']) {
  const app = setup(); app.poll();
  if(type === 'timeout') app.timeout();
  else app.events.get(type)({message: 'WASM unavailable', detail: 'WASM unavailable', reason: 'WASM unavailable'});
  assert.equal(app.panel.hidden, false); assert.equal(app.retry.hidden, false);
  assert.match(app.message.textContent, /could not load/); assert.equal(app.aborts(), 1);
  app.connect(); assert.equal(app.panel.hidden, false, 'Late work must not clear a failure');
  app.retry.onclick(); assert.equal(app.reloads(), 1);
}
'''
        subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[2],
                       check=True, capture_output=True, text=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
