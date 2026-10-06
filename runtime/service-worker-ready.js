// Wait for the current build to control the page before loading the app.
export const serviceWorkerReady = (async () => {
  if (!("serviceWorker" in navigator)) {
    throw new Error("This browser does not support service workers.");
  }
  serviceWorkerPath += "?v=__COURSE_RUNTIME_VERSION__";
  const expectedWorker = new URL(serviceWorkerPath, location.href).href;
  const registration = await navigator.serviceWorker.register(serviceWorkerPath, { type: "module", updateViaCache: "none" });
  await registration.update();
  await navigator.serviceWorker.ready;
  if (navigator.serviceWorker.controller?.scriptURL !== expectedWorker) {
    await new Promise((resolve, reject) => {
      const finish = () => {
        if (registration.waiting?.scriptURL === expectedWorker) {
          registration.waiting.postMessage({type: "course:activate-runtime",
            version: "course-" + new URL(expectedWorker).searchParams.get("v")});
        }
        if (navigator.serviceWorker.controller?.scriptURL !== expectedWorker) return;
        clearTimeout(timer);
        clearInterval(poll);
        navigator.serviceWorker.removeEventListener("controllerchange", finish);
        resolve();
      };
      const timer = setTimeout(() => {
        clearInterval(poll);
        navigator.serviceWorker.removeEventListener("controllerchange", finish);
        reject(new Error("Service worker did not take control within 30 seconds. Reload to retry."));
      }, 30000);
      const poll = setInterval(finish, 200);
      navigator.serviceWorker.addEventListener("controllerchange", finish);
      // Reloads may deliberately bypass control. An already-active worker does
      // not rerun its activate handler, so ask it to claim this document too.
      registration.active?.postMessage({ type: "course:claim-client" });
      finish();
    });
  }
  return registration;
})();
