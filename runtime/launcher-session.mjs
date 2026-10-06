// Framework-independent launcher side of the bridge. No shell or history UI here.
const identityKeys = ["protocol", "unit", "build_id", "source_id", "session_id"];
const sameIdentity = (a, b) => identityKeys.every(key => a?.[key] === b?.[key]);

export class AssignmentLauncher {
  constructor({window, container, manifest, onStatus = () => {},
    createFrame = () => window.document.createElement("iframe"),
    sessionId = () => window.crypto.randomUUID(), readinessMs = 120000, disposalMs = 1000}) {
    if (!manifest?.build_id || !Array.from({length: 7}, (_, i) => manifest.sources?.[i + 1])
      .every(source => typeof source === "string" && source)) throw new Error("Incomplete assignment manifest");
    if (!(disposalMs > 0 && disposalMs <= 1000) || !(readinessMs > 0)) throw new Error("Invalid lifecycle deadline");
    Object.assign(this, {window, container, manifest, onStatus, createFrame, sessionId, readinessMs, disposalMs});
    this.origin = window.location.origin;
    this.closed = false;
    this.revision = 0;
    this.receive = event => this._receive(event);
    window.addEventListener("message", this.receive);
  }

  select(unit, {retry = false} = {}) {
    if (this.closed) return Promise.reject(new Error("Launcher has been closed"));
    if (!Number.isInteger(unit) || unit < 1 || unit > 7) return Promise.reject(new Error("Invalid assignment"));
    if (this.desired?.unit !== unit || retry) this.desired = {unit, revision: ++this.revision};
    return this._run();
  }

  retry() {
    return this.select(this.desired?.unit || 1, {retry: true});
  }

  _run() {
    if (!this.running) {
      // Defer so rapid selections in the same turn only start their latest unit.
      this.running = Promise.resolve().then(async () => {
        while (this.closed ? this.active : this.active?.revision !== this.desired?.revision) {
          if (this.active) await this._dispose(this.active);
          if (!this.closed) this._mount(this.desired);
        }
      }).finally(() => { this.running = null; });
    }
    return this.running;
  }

  _mount(selection) {
    const identity = Object.freeze({protocol: 1, unit: selection.unit, build_id: this.manifest.build_id,
      source_id: this.manifest.sources[selection.unit], session_id: this.sessionId()});
    if (typeof identity.session_id !== "string" || !identity.session_id || identity.session_id.length > 128)
      throw new Error("Invalid launcher session identity");
    const frame = this.createFrame();
    frame.src = new URL(`/unit${selection.unit}/`, this.origin).href;
    frame.title = `Assignment ${selection.unit}`;
    const active = {identity, frame, revision: selection.revision, state: "loading"};
    this.active = active;
    active.loaded = () => {
      if (this.active === active && active.state !== "disposing") {
        // Each document load must establish its own connection and initialized inputs.
        active.identity = Object.freeze({...identity, session_id: this.sessionId()});
        active.state = "loading";
        this._deadline(active);
        frame.contentWindow.postMessage({type: "course:connect", identity: active.identity}, this.origin);
      }
    };
    frame.addEventListener("load", active.loaded);
    this._deadline(active);
    this.container.appendChild(frame);
    this.onStatus({state: "loading", unit: identity.unit});
  }

  _deadline(active) {
    this.window.clearTimeout(active.readinessTimer);
    active.readinessTimer = this.window.setTimeout(() => {
      if (this.active !== active || active.state !== "loading") return;
      active.state = "error";
      this.onStatus({state: "error", unit: active.identity.unit, error: "Assignment startup timed out. Retry to restart."});
    }, this.readinessMs);
  }

  _receive(event) {
    const active = this.active, data = event.data;
    if (!active || event.origin !== this.origin || event.source !== active.frame.contentWindow ||
      !sameIdentity(data?.identity, active.identity)) return;
    if (active.state === "disposing") {
      if (data?.type === "course:disposed" && data.clean === true) active.disposed?.("cooperative");
      return;
    }
    if (active.state !== "loading") return;
    if (data.type === "course:ready" && data.connected === true && data.inputs_initialized === true) {
      this.window.clearTimeout(active.readinessTimer);
      active.state = "ready";
      this.onStatus({state: "ready", unit: active.identity.unit});
    } else if (data.type === "course:error" && typeof data.error === "string" && data.error) {
      this.window.clearTimeout(active.readinessTimer);
      active.state = "error";
      this.onStatus({state: "error", unit: active.identity.unit, error: data.error});
    }
  }

  async _dispose(active) {
    active.state = "disposing";
    this.window.clearTimeout(active.readinessTimer);
    const outcome = await new Promise(resolve => {
      active.disposed = resolve;
      active.disposalTimer = this.window.setTimeout(() => resolve("deadline"), this.disposalMs);
      try { active.frame.contentWindow.postMessage({type: "course:dispose", identity: active.identity}, this.origin); }
      catch (_) { resolve("inaccessible"); }
    });
    this.window.clearTimeout(active.disposalTimer);
    active.frame.removeEventListener("load", active.loaded);
    // Directly terminate registered resources before detaching the document.
    // This also works if the child's message listener is stalled or gone.
    try { active.frame.contentWindow.courseSessionBridge?.dispose(); } catch (_) {}
    active.frame.remove();
    if (this.active === active) this.active = null;
    this.onStatus({state: "disposed", unit: active.identity.unit, outcome});
  }

  close() {
    this.closed = true;
    return this._run().finally(() => this.window.removeEventListener("message", this.receive));
  }
}
