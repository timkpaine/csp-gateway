import React from "react";
import { createRoot } from "react-dom/client";
import App from "./index";

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

window.addEventListener("load", () => {
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
    const root = createRoot(container);
    root.render(<App />);
  }
});
