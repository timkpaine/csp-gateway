import { registerHandler } from "../../js/cdn/index.js";

const CUSTOM_LAYOUT = "Custom Layout";
const CUSTOM_LAYOUT_STORAGE_KEY = "csp_gateway_demo_config";
const LAYOUT_SELECTOR_ID = "gateway-layout-selector";
const WORKSPACE_ID = "gateway-workspace";

function stripTransientFields(layout) {
  const cloned = structuredClone(layout);
  for (const panel of Object.values(cloned.panels || {})) {
    delete panel.theme;
    for (const column of Object.values(panel.plugin_config?.columns || {})) {
      delete column.column_size_override;
    }
  }
  return cloned;
}

function workspace(currentTarget) {
  return currentTarget.ownerDocument.getElementById(WORKSPACE_ID);
}

registerHandler("csp-gateway:save-layout", (_event, currentTarget) => {
  void (async () => {
    const layout = stripTransientFields(await workspace(currentTarget).save());
    localStorage.setItem(CUSTOM_LAYOUT_STORAGE_KEY, JSON.stringify(layout));
    const customLayout = globalThis.cspGatewayCustomLayout;
    for (const key of Object.keys(customLayout)) {
      delete customLayout[key];
    }
    Object.assign(customLayout, layout);

    const selector =
      currentTarget.ownerDocument.getElementById(LAYOUT_SELECTOR_ID);
    selector.value = CUSTOM_LAYOUT;
    selector.dispatchEvent(new Event("input", { bubbles: true }));
  })().catch((error) =>
    console.error("Failed to save Perspective layout:", error),
  );
});

registerHandler("csp-gateway:download-layout", (_event, currentTarget) => {
  void (async () => {
    const layout = stripTransientFields(await workspace(currentTarget).save());
    const json = JSON.stringify(layout).replace(
      /PERSPECTIVE_GENERATED_/g,
      "CSP_GATEWAY_GENERATED_",
    );
    const form = currentTarget.ownerDocument.createElement("form");
    form.method = "POST";
    form.action = currentTarget.dataset.downloadUrl;
    form.target = "_blank";
    form.hidden = true;
    const input = currentTarget.ownerDocument.createElement("input");
    input.type = "hidden";
    input.name = "layout";
    input.value = json;
    form.appendChild(input);
    currentTarget.ownerDocument.body.appendChild(form);
    form.submit();
    form.remove();
  })().catch((error) =>
    console.error("Failed to download Perspective layout:", error),
  );
});
