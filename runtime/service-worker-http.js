// A proxy request belongs to one application document. Never leave an orphaned
// respondWith promise keeping the old service worker active during an upgrade.
async function fetchASGI(client, resource, init, filter = (chunk) => chunk, appPath = "") {
  if (typeof resource === "string" || typeof init !== "undefined") resource = new Request(resource, init);
  const channel = new MessageChannel();
  const port = channel.port1;
  let finished = false, response, streamController, resolveResponse, rejectResponse;
  const pending = courseRequests.get(appPath) || new Set();
  courseRequests.set(appPath, pending);
  const responsePromise = new Promise((resolve, reject) => { resolveResponse = resolve; rejectResponse = reject; });
  const stream = new ReadableStream({
    start(controller) { streamController = controller; },
    cancel(reason) { cancel(reason); }
  });
  function cleanup() {
    clearTimeout(deadline);
    clearInterval(ownerCheck);
    resource.signal.removeEventListener("abort", aborted);
    port.close();
    pending.delete(cancel);
    if (!pending.size) courseRequests.delete(appPath);
  }
  function cancel(reason = "The application document closed.") {
    if (finished) return;
    finished = true;
    const error = reason instanceof Error ? reason : new DOMException(String(reason), "AbortError");
    try { port.postMessage({type: "http.disconnect"}); } catch (_) {}
    try { streamController.error(error); } catch (_) {}
    if (!response) rejectResponse(error);
    cleanup();
  }
  const aborted = () => cancel(resource.signal.reason);
  const deadline = setTimeout(() => cancel(new Error("The application HTTP request timed out. Reload to retry.")), 120000);
  // pagehide is best-effort. Also detect a document that disappeared before it
  // could send disposal, without depending on fetch AbortSignal propagation.
  const ownerCheck = setInterval(async () => {
    const owner = courseAppOwners[appPath];
    // A replacement worker can take over a still-open document. clients.get()
    // still finds that window, but it no longer owns this worker's proxy.
    const controlled = owner && await self.clients.matchAll({type: "window", includeUncontrolled: false});
    if (owner && !controlled.some(client => client.id === owner)) {
      for (const abort of [...(courseRequests.get(appPath) || [])]) abort();
      apps[appPath]?.close();
      delete apps[appPath];
      delete courseAppOwners[appPath];
    }
  }, 500);
  pending.add(cancel);
  resource.signal.addEventListener("abort", aborted, {once: true});
  port.onmessageerror = () => cancel(new Error("The application sent an invalid HTTP response."));
  port.onmessage = (event) => {
    if (finished) return;
    try {
      const msg = event.data;
      if (msg.type === "http.response.start") {
        if (response) throw new Error("The application started its HTTP response twice.");
        response = asgiToRes(filter.transformResponseStart ? filter.transformResponseStart(msg) : msg, stream);
        resolveResponse(response);
      } else if (msg.type === "http.response.body" && response) {
        const chunk = filter(msg.body || new Uint8Array(0), response, !msg.more_body);
        if (chunk.length) streamController.enqueue(chunk);
        if (!msg.more_body) {
          finished = true;
          streamController.close();
          cleanup();
        }
      } else {
        cancel(new Error("Unexpected application HTTP response: " + msg.type));
      }
    } catch (error) { cancel(error); }
  };
  port.start();
  if (resource.signal.aborted) {
    cancel(resource.signal.reason);
    return responsePromise;
  }
  try {
    client.postMessage({type: "makeRequest", scope: reqToASGI(resource)}, [channel.port2]);
    (async () => {
      const blob = await resource.blob();
      if (finished) return;
      if (!blob.size) {
        port.postMessage({type: "http.request", more_body: false});
        return;
      }
      const reader = blob.stream().getReader();
      try {
        while (!finished) {
          const {value, done} = await reader.read();
          if (finished) break;
          port.postMessage({type: "http.request", body: value, more_body: !done});
          if (done) break;
        }
      } finally { reader.releaseLock(); }
    })().catch(cancel);
  } catch (error) { cancel(error); }
  return responsePromise;
}
