function setupAppProxyPath(proxy) {
  const appName = `app_${makeRandomKey(20)}`;
  const urlPath = appName + "/";
  let channel, controller, disposed = false;
  const owners = new Set();
  const release = owner => {
    // An old worker may already be redundant. Its independent owner check
    // settles remaining requests; it must not prevent disposal of newer ports.
    try { owner?.postMessage({type: "course:dispose-proxy", path: urlPath}); } catch (_) {}
  };
  const connect = () => {
    if (disposed || !navigator.serviceWorker.controller) return;
    // The proxy belongs to the worker that received its port. After takeover,
    // disposing only the current controller strands the previous worker's
    // respondWith promises and can prevent the following upgrade activating.
    release(controller);
    channel?.port1.close();
    controller = navigator.serviceWorker.controller;
    owners.add(controller);
    channel = createHttpRequestChannel(proxy, appName, urlPath);
  };
  const restarted = event => {
    if (event.data?.type === "serviceworkerStart" && event.source === navigator.serviceWorker.controller) connect();
  };
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    navigator.serviceWorker.removeEventListener("message", restarted);
    navigator.serviceWorker.removeEventListener("controllerchange", connect);
    for (const owner of owners) release(owner);
    channel?.port1.close();
  };
  if (!navigator.serviceWorker.controller) throw new Error("ServiceWorker controller was not found!");
  connect();
  navigator.serviceWorker.addEventListener("message", restarted);
  navigator.serviceWorker.addEventListener("controllerchange", connect);
  window.addEventListener("pagehide", dispose, {once: true});
  window.addEventListener("course:startup-abort", dispose, {once: true});
  window.courseSessionBridge?.registerDisposer(dispose);
  return {appName, urlPath};
}
