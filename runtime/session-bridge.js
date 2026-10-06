// Installed before Shinylive starts. Standalone pages need no launcher handshake.
(() => {
  const pinned = window.courseAssignmentIdentity;
  if (!pinned || !Number.isInteger(pinned.unit) || pinned.unit < 1 || pinned.unit > 7 ||
      !pinned.build_id || !pinned.source_id) throw new Error("Missing assignment identity");
  const origin = location.origin;
  const keys = ["protocol", "unit", "build_id", "source_id", "session_id"];
  const disposers = new Set();
  let owner, ready = false, error, disposing = false, disposal, clean = false;
  const matches = tags => owner && keys.every(key => tags?.[key] === owner[key]);
  const send = (type, detail = {}) => {
    if (owner) window.parent.postMessage({type, identity: owner, ...detail}, origin);
  };
  const publish = () => {
    if (disposing) return;
    if (error) send("course:error", {error});
    else if (ready) send("course:ready", {connected: true, inputs_initialized: true});
  };
  const dispose = () => {
    if (disposal) return disposal;
    disposing = true;
    window.dispatchEvent(new Event("course:session-dispose"));
    // Stop ports and the main Python worker even if startup never completed.
    window.dispatchEvent(new Event("course:startup-abort"));
    const jobs = [...disposers].map(fn => {
      try { return Promise.resolve(fn()); } catch (e) { return Promise.reject(e); }
    });
    disposers.clear();
    disposal = Promise.allSettled(jobs).then(results => {
      clean = results.every(result => result.status === "fulfilled");
      send("course:disposed", {clean});
    });
    return disposal;
  };
  window.courseSessionBridge = Object.freeze({
    ready() { if (!disposing && !error) { ready = true; publish(); } },
    failed(reason) {
      if (disposing || ready) return;
      error = String(reason?.message || reason || "The application did not connect.");
      publish();
    },
    registerDisposer(fn) {
      if (typeof fn !== "function") throw new TypeError("A disposer must be a function");
      if (disposing) { Promise.resolve().then(fn).catch(() => {}); return () => {}; }
      disposers.add(fn);
      return () => disposers.delete(fn);
    },
    dispose,
  });
  window.addEventListener("message", event => {
    if (window.parent === window || event.origin !== origin || event.source !== window.parent) return;
    const data = event.data, tags = data?.identity;
    if (data?.type === "course:connect" && !disposing) {
      if (tags?.protocol !== 1 || tags.unit !== pinned.unit || tags.build_id !== pinned.build_id ||
          tags.source_id !== pinned.source_id || typeof tags.session_id !== "string" ||
          !tags.session_id || tags.session_id.length > 128 || (owner && !matches(tags))) return;
      owner = Object.freeze(Object.fromEntries(keys.map(key => [key, tags[key]])));
      publish();
    } else if (data?.type === "course:dispose" && matches(tags)) {
      if (disposal) disposal.then(() => send("course:disposed", {clean, repeated: true}));
      else dispose();
    }
  });
  window.addEventListener("pagehide", dispose, {once: true});
})();
