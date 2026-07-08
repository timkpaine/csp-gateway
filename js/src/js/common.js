// eslint-disable-next-line import/no-mutable-exports, prefer-const
export let CUSTOM_LAYOUT_CONFIG_NAME = "csp_gateway_demo_config";

export const changeLayoutConfigName = (newName) => {
  CUSTOM_LAYOUT_CONFIG_NAME = newName;
};

// URL prefix the app is served under behind a reverse proxy (e.g. "/watchtower").
// Supplied by the server in the injected UI config so every API/websocket call
// targets the proxied path rather than the origin root.
export const getBasePath = () => {
  const cfg = window.__CSP_GATEWAY_UI_CONFIG__;
  let base = (cfg && cfg.basePath) || "";
  if (!base && typeof document !== "undefined") {
    // Fall back to the server-rendered <base> element so API calls and the
    // /ui-config fetch still target the proxied sub-path when the config global
    // is not injected (e.g. a downstream shell reusing main.js under ROOT_PATH).
    const baseEl = document.querySelector("base");
    const href = baseEl && baseEl.getAttribute("href");
    if (href) base = href;
  }
  if (base.endsWith("/")) base = base.slice(0, -1);
  return base;
};

export const httpUrl = (path) =>
  `${window.location.protocol}//${window.location.host}${getBasePath()}${path}`;

export const wsUrl = (path) => {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${getBasePath()}${path}`;
};

export const hideLoader = () => {
  setTimeout(() => {
    const progress = document.getElementById("loader");
    progress.setAttribute("style", "display:none;");
  }, 3000);
};

export const getOpenApi = async () => {
  const openapi = await fetch(httpUrl("/openapi.json"));
  const json = await openapi.json();
  return json;
};

export const shutdownDefault = async () => {
  // TODO check if can shutdown by checking openapi
  await fetch(httpUrl("/api/v1/controls/shutdown"), { method: "POST" });
};
