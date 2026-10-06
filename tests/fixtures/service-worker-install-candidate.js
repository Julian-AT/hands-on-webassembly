// Installation depends on cache preparation, not on completing activation.
// Keeping skipWaiting's activation promise in install.waitUntil can create a
// circular lifecycle dependency in browsers that defer that promise.
let courseInstallCacheReady = false;
let courseActivationRequests = 0;
let courseActivationError = null;
function courseRequestActivation() {
  courseActivationRequests++;
  self.skipWaiting().catch(error => { courseActivationError = String(error); });
}
self.addEventListener("install", event => {
  event.waitUntil(caches.open(version + cacheName).then(() => { courseInstallCacheReady = true; }));
  courseRequestActivation();
});
