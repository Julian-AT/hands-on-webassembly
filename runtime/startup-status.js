// Outside React's mount point: stays visible until the Shiny session connects.
const startupPanel = document.getElementById("course-startup");
const startupMessage = startupPanel.querySelector("p");
const startupRetry = startupPanel.querySelector("button");
let startupFinished = false;
let watchedWindow;
let watchedShiny;
let initializedShiny;
function startupFailed(error) {
  if (startupFinished) return;
  startupFinished = true;
  clearInterval(startupPoll);
  clearTimeout(startupDeadline);
  startupPanel.hidden = false;
  startupPanel.setAttribute("role", "alert");
  startupMessage.textContent = "This assignment could not load. Check your connection and retry. " +
    (error?.message || String(error || "The application did not connect."));
  startupRetry.hidden = false;
  window.courseSessionBridge?.failed(error);
  window.dispatchEvent(new Event("course:startup-abort"));
}
function startupConnected(frameWindow, shiny) {
  if (startupFinished || frameWindow !== watchedWindow || shiny !== watchedShiny ||
      shiny !== initializedShiny || frameWindow.Shiny !== shiny ||
      document.querySelector("#root iframe")?.contentWindow !== frameWindow) return;
  const app = shiny.shinyapp;
  // isConnected() alone only checks socket existence in the pinned Shiny client.
  if (!app?.isConnected() || app.$socket?.readyState !== 1 ||
      !app.$initialInput || !Object.keys(app.$initialInput).length) return;
  startupFinished = true;
  clearInterval(startupPoll);
  clearTimeout(startupDeadline);
  startupPanel.hidden = true;
  window.courseSessionBridge?.ready();
}
startupRetry.onclick = () => location.reload();
window.addEventListener("course:startup-error", event => startupFailed(event.detail));
window.addEventListener("error", event => startupFailed(event.error || event.message));
window.addEventListener("unhandledrejection", event => startupFailed(event.reason));
const startupPoll = setInterval(() => {
  const frame = document.querySelector("#root iframe");
  try {
    const frameWindow = frame?.contentWindow;
    const shiny = frameWindow?.Shiny;
    const initialized = shiny?.initializedPromise;
    if (initialized && (frameWindow !== watchedWindow || shiny !== watchedShiny)) {
      watchedWindow = frameWindow;
      watchedShiny = shiny;
      initialized.then(() => {
        if (watchedWindow !== frameWindow || watchedShiny !== shiny) return;
        initializedShiny = shiny;
        startupConnected(frameWindow, shiny);
      }, error => {
        if (watchedWindow === frameWindow && watchedShiny === shiny) startupFailed(error);
      });
      window.courseSessionBridge?.registerDisposer(() => shiny.shinyapp?.$socket?.close());
    }
    if (initializedShiny === shiny) startupConnected(frameWindow, shiny);
  } catch (_) { /* A frame navigating to its same-origin application is transient. */ }
}, 100);
const startupDeadline = setTimeout(() => startupFailed(new Error(
  "Startup did not finish within two minutes. Reload to restart the browser worker."
)), 120000);
const stopStartup = () => {
  startupFinished = true;
  clearInterval(startupPoll);
  clearTimeout(startupDeadline);
};
window.addEventListener("pagehide", stopStartup, { once: true });
window.addEventListener("course:session-dispose", stopStartup, { once: true });
