import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./index";

window.addEventListener("load", () => {
  const container = document.getElementById("gateway-root");

  // handle regular-table in light dom
  customElements.whenDefined("perspective-viewer-datagrid").then((datagrid) => {
    datagrid.renderTarget = "light";
  });

  if (container) {
    const root = createRoot(container);
    root.render(<App />);
  }
});
