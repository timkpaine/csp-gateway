import React from "react";
import { createRoot } from "react-dom/client";
import App, * as CspGateway from "./index";
import { changeLayoutConfigName, httpUrl } from "./common";

// Expose the gateway frontend library and React on the window so that custom
// Javascript injected by the server (via the `CUSTOM_JS` / `CUSTOM_STATIC_DIR`
// settings) can use helpers like `getDefaultViewerConfig` without bundling.
window.React = React;
window.CSPGateway = CspGateway;
// Backwards-compatible alias for custom JS written against the prior name.
window.cspGateway = CspGateway;

function isPerspectiveViewNotFoundError(error) {
  const message = error?.message || String(error || "");
  return message.includes("View not found");
}

// Perspective can reject stale view operations after a viewer is replaced
// during rapid layout changes. The library treats this as a benign cancelled
// render, but some paths still surface it as an unhandled browser error.
window.addEventListener("unhandledrejection", (event) => {
  if (isPerspectiveViewNotFoundError(event.reason)) {
    event.preventDefault();
  }
});

window.addEventListener("error", (event) => {
  if (isPerspectiveViewNotFoundError(event.error || event.message)) {
    event.preventDefault();
  }
});

// Read server-provided UI customization (title, logos, custom assets). Prefer
// the config injected into the page, falling back to the `/ui-config` endpoint.
async function getUiConfig() {
  if (window.__CSP_GATEWAY_UI_CONFIG__) {
    return window.__CSP_GATEWAY_UI_CONFIG__;
  }
  try {
    const response = await fetch(httpUrl("/ui-config"));
    if (response.ok) {
      return await response.json();
    }
  } catch (err) {
    console.warn("Failed to load /ui-config:", err);
  }
  return {};
}

// Wrap a raw HTML/SVG string into a React component, used for custom logos and
// loaders supplied as HTML strings (so no JSX/build step is needed downstream).
function htmlComponent(html, className) {
  return function HtmlComponent() {
    return React.createElement("div", {
      className,
      dangerouslySetInnerHTML: { __html: html },
    });
  };
}

// Build the props passed to <App> from server config and any custom globals set
// by injected Javascript on `window.__CSP_GATEWAY_CUSTOM__`.
function buildAppProps(config, custom) {
  const props = {};

  if (custom.layoutConfigName) {
    changeLayoutConfigName(custom.layoutConfigName);
  }

  if (custom.headerLogo) {
    props.headerLogo = custom.headerLogo;
  } else if (custom.headerLogoHtml) {
    props.headerLogo = React.createElement(
      htmlComponent(custom.headerLogoHtml, "header-logo"),
    );
  } else if (config.headerLogo) {
    props.headerLogo = React.createElement("img", {
      className: "header-logo",
      src: config.headerLogo,
      alt: config.title || "logo",
    });
  }

  if (custom.footerLogo) {
    props.footerLogo = custom.footerLogo;
  } else if (custom.footerLogoHtml) {
    props.footerLogo = React.createElement(
      htmlComponent(custom.footerLogoHtml, "footer-logo"),
    );
  } else if (config.footerLogo) {
    props.footerLogo = React.createElement("img", {
      className: "footer-logo",
      src: config.footerLogo,
      alt: config.title || "logo",
    });
  }

  if (custom.loader) {
    props.loader = custom.loader;
  } else if (custom.loaderHtml) {
    props.loader = htmlComponent(custom.loaderHtml, "custom-loader");
  }

  if (custom.processTables) props.processTables = custom.processTables;
  if (custom.shutdown) props.shutdown = custom.shutdown;
  if (custom.overrideSettingsButtons)
    props.overrideSettingsButtons = custom.overrideSettingsButtons;
  if (custom.extraSettingsButtons)
    props.extraSettingsButtons = custom.extraSettingsButtons;

  return props;
}

window.addEventListener("load", async () => {
  const container = document.getElementById("gateway-root");

  // Clear selections on other viewers when clicking into a new one.
  document.addEventListener("mousedown", (e) => {
    const clicked = e.target.closest("perspective-viewer");
    if (!clicked) return;
    for (const v of document.querySelectorAll("perspective-viewer")) {
      try {
        if (v !== clicked && v.getSelection()) v.setSelection();
      } catch (err) {
        console.warn("Failed to clear Perspective selection:", err);
      }
    }
  });

  // Ctrl+C / Cmd+C: copy selected cells from the focused viewer.
  document.addEventListener("keydown", async (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "c") {
      const viewer = document.activeElement?.closest("perspective-viewer");
      if (viewer?.getSelection()) {
        e.preventDefault();
        try {
          await viewer.copy("plugin");
        } catch (err) {
          console.warn("Copy failed:", err);
        }
      }
    }
  });

  if (container) {
    const config = await getUiConfig();
    const custom = window.__CSP_GATEWAY_CUSTOM__ || {};
    const props = buildAppProps(config, custom);
    const root = createRoot(container);
    root.render(<App {...props} />);
  }
});
