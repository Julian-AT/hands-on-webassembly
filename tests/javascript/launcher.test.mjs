import test from 'node:test';
import assert from 'node:assert/strict';
import { AssignmentLauncher } from '../../runtime/launcher-session.mjs';
function fixture() {
  const handlers = new Map(),
    frames = [],
    statuses = [];
  let count = 0;
  const window = {
    location: { origin: 'https://course.test' },
    crypto: { randomUUID: () => String(++count) },
    setTimeout,
    clearTimeout,
    addEventListener: (name, fn) => handlers.set(name, fn),
    removeEventListener: (name) => handlers.delete(name),
  };
  const createFrame = () => {
    const frame = {
      events: new Map(),
      messages: [],
      addEventListener: (n, f) => frame.events.set(n, f),
      removeEventListener: (n) => frame.events.delete(n),
      remove: () => {
        frame.removed = true;
      },
      contentWindow: {
        postMessage: (data, origin) => frame.messages.push({ data, origin }),
        courseSessionBridge: {
          dispose: () => {
            frame.terminated = true;
          },
        },
      },
    };
    return frame;
  };
  const manifest = {
    build_id: 'build-one',
    sources: Object.fromEntries(
      Array.from({ length: 7 }, (_, i) => [i + 1, `source-${i + 1}`]),
    ),
  };
  const launcher = new AssignmentLauncher({
    window,
    manifest,
    container: { appendChild: (frame) => frames.push(frame) },
    createFrame,
    onStatus: (status) => statuses.push(status),
    disposalMs: 20,
    readinessMs: 1000,
  });
  const message = (
    data,
    origin = window.location.origin,
    source = launcher.active.frame.contentWindow,
  ) => handlers.get('message')?.({ data, origin, source });
  return { launcher, frames, statuses, message, handlers };
}
test('requires initialized inputs and rejects every obsolete or wrong message identity', async () => {
  const f = fixture();
  await f.launcher.select(1);
  const active = f.launcher.active;
  f.frames[0].events.get('load')();
  const ready = {
    type: 'course:ready',
    identity: active.identity,
    connected: true,
    inputs_initialized: true,
  };
  f.message({ ...ready, inputs_initialized: false });
  f.message(ready, 'https://other.test');
  f.message(ready, undefined, {});
  for (const key of ['protocol', 'unit', 'build_id', 'source_id', 'session_id'])
    f.message({ ...ready, identity: { ...ready.identity, [key]: 'wrong' } });
  assert.equal(active.state, 'loading');
  f.message(ready);
  assert.equal(active.state, 'ready');
  await f.launcher.close();
  assert.equal(f.handlers.has('message'), false);
});
test('rapid selection keeps latest, disposes old session before mounting, ignores stale errors', async () => {
  const f = fixture();
  await Promise.all([
    f.launcher.select(1),
    f.launcher.select(2),
    f.launcher.select(7),
  ]);
  assert.equal(f.frames.length, 1);
  assert.equal(f.launcher.active.identity.unit, 7);
  const old = f.launcher.active;
  const change = f.launcher.select(3);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(f.frames.length, 1);
  f.message({ type: 'course:disposed', identity: old.identity, clean: true });
  await change;
  assert.equal(f.frames.length, 2);
  assert.equal(f.frames[0].removed, true);
  assert.equal(f.frames[0].terminated, true);
  assert.equal(f.launcher.active.identity.unit, 3);
  f.message(
    { type: 'course:error', identity: old.identity, error: 'stale' },
    undefined,
    old.frame.contentWindow,
  );
  assert.equal(f.launcher.active.state, 'loading');
  await f.launcher.close();
});
test('deadline forces removal and Retry creates a new session', async () => {
  const f = fixture();
  await f.launcher.select(4);
  const old = f.launcher.active;
  await f.launcher.retry();
  assert.notEqual(
    f.launcher.active.identity.session_id,
    old.identity.session_id,
  );
  assert.equal(old.frame.removed, true);
  assert.equal(old.frame.terminated, true);
  await f.launcher.close();
});
