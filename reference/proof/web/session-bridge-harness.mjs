import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {AssignmentLauncher} from "../runtime/launcher-session.mjs";

const bridge = fs.readFileSync(new URL("../runtime/session-bridge.js", import.meta.url), "utf8");
const origin = "https://owned.example";
const pinned = {unit: 1, build_id: "build", source_id: "source-1"};
const owner = {protocol: 1, ...pinned, session_id: "session"};
const cases = {};
async function check(name, fn) {
  await fn();
  cases[name] = {status: "pass", assertion: {kind: "workflow", expected: true, observed: true, matched: true}};
}
function child({standalone = false} = {}) {
  const messages = [], events = new Map(), dispatches = [];
  const window = {courseAssignmentIdentity: pinned,
    addEventListener: (name, fn) => events.set(name, fn),
    dispatchEvent: event => {dispatches.push(event.type); events.get(event.type)?.(event);}};
  const parent = standalone ? window : {postMessage: (message, target) => messages.push({message, target})};
  window.parent = parent;
  vm.runInNewContext(bridge, {window, location: {origin}, Event, Promise});
  const send = (type, identity = owner, extras = {}) => events.get("message")({origin, source: parent,
    data: {type, identity}, ...extras});
  return {window, events, messages, dispatches, send, api: window.courseSessionBridge};
}
await check("child-pinned-handshake-and-buffered-readiness", () => {
  const c = child(); c.api.ready(); assert.equal(c.messages.length, 0);
  for (const identity of [{...owner, protocol: 2}, {...owner, unit: 2}, {...owner, build_id: "stale"},
    {...owner, source_id: "stale"}, {...owner, session_id: ""}, {...owner, session_id: 10}]) {
    c.send("course:connect", identity); assert.equal(c.messages.length, 0);
  }
  c.send("course:connect", owner, {origin: "https://foreign.example"});
  c.send("course:connect", owner, {source: {}}); assert.equal(c.messages.length, 0);
  c.send("course:connect"); assert.equal(c.messages[0].message.type, "course:ready");
  assert.equal(c.messages[0].target, origin);
  const count = c.messages.length;
  c.send("course:connect", {...owner, session_id: "replacement"}); assert.equal(c.messages.length, count);
  const standalone = child({standalone: true}); standalone.send("course:connect"); standalone.api.ready();
  assert.equal(standalone.messages.length, 0);
});
await check("child-error-recovery-buffer-and-late-readiness-rejection", () => {
  const c = child(); c.api.failed(new Error("checksum rejected")); c.api.ready(); c.send("course:connect");
  assert.equal(c.messages.at(-1).message.type, "course:error");
  assert.equal(c.messages.at(-1).message.error, "checksum rejected");
});
await check("child-complete-owner-disposal-idempotence-and-port-worker-hooks", async () => {
  const c = child(); c.send("course:connect"); let closed = 0, terminated = 0;
  c.api.registerDisposer(() => closed++); c.api.registerDisposer(() => terminated++);
  for (const key of Object.keys(owner)) {
    c.send("course:dispose", {...owner, [key]: "obsolete"}); assert.equal(closed, 0);
  }
  c.send("course:dispose"); await c.api.dispose();
  assert.equal(closed, 1); assert.equal(terminated, 1);
  assert.equal(c.messages.at(-1).message.type, "course:disposed");
  assert.equal(c.messages.at(-1).message.clean, true);
  assert.deepEqual(c.dispatches, ["course:session-dispose", "course:startup-abort"]);
  const count = c.messages.length; c.api.ready(); c.api.failed("stale"); assert.equal(c.messages.length, count);
  c.send("course:dispose"); await Promise.resolve(); assert.equal(closed, 1);
  assert.equal(c.messages.at(-1).message.repeated, true);
});
await check("child-failed-disposal-never-acknowledges-clean", async () => {
  const c = child(); c.send("course:connect");
  c.api.registerDisposer(() => {throw new Error("port close failed");});
  await c.api.dispose(); assert.equal(c.messages.at(-1).message.clean, false);
  c.send("course:dispose"); await Promise.resolve(); assert.equal(c.messages.at(-1).message.clean, false);
});
function launcher() {
  const frames = [], statuses = [], listeners = new Map(), timers = new Map(); let timer = 0, session = 0;
  const window = {location: {origin},
    addEventListener: (name, fn) => listeners.set(name, fn), removeEventListener: name => listeners.delete(name),
    setTimeout: (fn, ms) => {timers.set(++timer, {fn, ms}); return timer;}, clearTimeout: id => timers.delete(id)};
  const container = {children: [], appendChild(frame) {this.children.push(frame);}};
  const makeFrame = () => {
    const events = new Map(), sent = [];
    const frame = {sent, events, contentWindow: {postMessage: (message, target) => sent.push({message, target})},
      addEventListener: (type, fn) => events.set(type, fn), removeEventListener: type => events.delete(type),
      remove: () => {container.children = container.children.filter(child => child !== frame);}};
    frames.push(frame); return frame;
  };
  const launch = new AssignmentLauncher({window, container, manifest: {build_id: "build",
    sources: Object.fromEntries(Array.from({length: 7}, (_, i) => [i + 1, `source-${i + 1}`]))},
    onStatus: state => statuses.push(state), createFrame: makeFrame, sessionId: () => `session-${++session}`});
  const receive = (type, extras = {}, event = {}) => {
    const active = launch.active;
    listeners.get("message")?.({origin, source: active.frame.contentWindow,
      data: {type, identity: active.identity, ...extras}, ...event});
  };
  const expire = ms => [...timers].filter(([, timer]) => timer.ms === ms).forEach(([id, {fn}]) => {timers.delete(id); fn();});
  return {launch, frames, statuses, container, receive, expire, timers};
}
await check("launcher-origin-source-owner-and-ready-inputs", async () => {
  const l = launcher(); await l.launch.select(1); const active = l.launch.active;
  active.frame.events.get("load")(); assert.equal(active.frame.sent.at(-1).target, origin);
  l.receive("course:ready", {connected: true, inputs_initialized: true}, {origin: "https://foreign.example"});
  l.receive("course:ready", {connected: true, inputs_initialized: true}, {source: {}});
  for (const key of Object.keys(active.identity)) {
    l.receive("course:ready", {connected: true, inputs_initialized: true, identity: {...active.identity, [key]: "old"}});
  }
  l.receive("course:ready", {connected: false, inputs_initialized: true});
  l.receive("course:ready", {connected: true, inputs_initialized: false});
  assert.equal(active.state, "loading");
  l.receive("course:ready", {connected: true, inputs_initialized: true}); assert.equal(active.state, "ready");
  const previousIdentity = active.identity;
  active.frame.events.get("load")(); assert.equal(active.state, "loading", "Reload must repeat handshake");
  assert.notEqual(active.identity.session_id, previousIdentity.session_id);
  l.receive("course:ready", {connected: true, inputs_initialized: true, identity: previousIdentity});
  assert.equal(active.state, "loading", "Queued readiness from the previous document is obsolete");
});
await check("launcher-rapid-selection-waits-and-retains-only-latest", async () => {
  const l = launcher(); await l.launch.select(1); const old = l.launch.active;
  const selection = l.launch.select(2); await Promise.resolve();
  l.launch.select(3); l.launch.select(7);
  assert.equal(l.container.children.length, 1); assert.equal(old.state, "disposing");
  l.receive("course:ready", {connected: true, inputs_initialized: true}); assert.equal(old.state, "disposing");
  l.receive("course:disposed", {clean: true}); await selection;
  assert.equal(l.frames.length, 2); assert.equal(l.launch.active.identity.unit, 7);
  assert.equal(l.container.children.length, 1); assert.equal(old.frame.events.size, 0);
  assert.equal(l.frames[1].src, `${origin}/unit7/`);
  l.receive("course:ready", {connected: true, inputs_initialized: true}, {source: old.frame.contentWindow});
  assert.equal(l.launch.active.state, "loading");
});
await check("launcher-deadline-forces-resource-disposal-and-removal", async () => {
  const l = launcher(); await l.launch.select(1); let terminated = 0;
  l.launch.active.frame.contentWindow.courseSessionBridge = {dispose: () => terminated++};
  const selection = l.launch.select(2); await Promise.resolve();
  l.receive("course:disposed", {clean: false}); assert.equal(l.frames.length, 1);
  l.expire(1000); await selection;
  assert.equal(terminated, 1); assert.equal(l.container.children.length, 1);
  assert.equal(l.statuses.find(status => status.state === "disposed").outcome, "deadline");
});
await check("launcher-error-timeout-retry-and-close", async () => {
  const l = launcher(); await l.launch.select(1); const initial = l.launch.active;
  l.expire(120000); assert.equal(initial.state, "error");
  l.receive("course:ready", {connected: true, inputs_initialized: true}); assert.equal(initial.state, "error");
  const retry = l.launch.retry(); await Promise.resolve(); l.receive("course:disposed", {clean: true}); await retry;
  assert.notEqual(l.launch.active.identity.session_id, initial.identity.session_id);
  l.receive("course:error", {error: "corrupt asset"}); assert.equal(l.launch.active.state, "error");
  const close = l.launch.close(); await Promise.resolve(); l.receive("course:disposed", {clean: true}); await close;
  assert.equal(l.container.children.length, 0); assert.equal(l.timers.size, 0);
  assert.equal(l.statuses.at(-1).outcome, "cooperative");
  await assert.rejects(l.launch.select(1), /closed/);
});
await check("launcher-same-turn-selection-starts-only-latest", async () => {
  const l = launcher(); const first = l.launch.select(1); l.launch.select(2); l.launch.select(6); await first;
  assert.equal(l.frames.length, 1); assert.equal(l.launch.active.identity.unit, 6);
  await assert.rejects(l.launch.select(8), /Invalid assignment/);
});
process.stdout.write(JSON.stringify({status: "pass", cases}) + "\n");
