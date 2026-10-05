// Replacing a proxy port also replaces its request owner. A queued makeRequest
// on the closed port can never complete, even when its document is still alive.
function courseConfigureProxy(event) {
  const path = event.data.path, port = event.ports[0], owner = event.source?.id;
  if (!port || !owner || (courseAppOwners[path] && courseAppOwners[path] !== owner)) {
    port?.close();
    return false;
  }
  for (const cancel of [...(courseRequests.get(path) || [])]) cancel();
  apps[path]?.close();
  apps[path] = port;
  courseAppOwners[path] = owner;
  return true;
}
