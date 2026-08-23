import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { getDefaultViewerConfig } from "./gateway";
import { fetchTables } from "./tables";
import {
  buildEmptyLayout,
  getServerLayouts,
  getCustomLayout,
  saveCustomLayout,
  getUrlLayout,
  setUrlLayout,
  stripTransientFields,
} from "./layout";
import { getCurrentTheme, getViewerTheme } from "./theme";

/**
 * Self-contained Perspective workspace component.
 *
 * Uses the raw <perspective-viewer> custom element directly, which in Perspective 5 hosts the
 * multi-panel layout that <perspective-workspace> used to provide, so we control the exact order
 * of load() -> restoreWorkspace().
 *
 * Exposes imperative methods via ref:
 *   - getLayouts()       → current layouts map
 *   - getActiveLayout()  → name of active layout
 *   - setActiveLayout(n) → switch to a named layout
 *   - saveLayout()       → persist current workspace state as "Custom Layout"
 *   - exportLayout()     → returns current workspace state as JSON string
 */
export const Workspace = forwardRef(function Workspace(
  { processTables, onReady },
  ref,
) {
  const wsRef = useRef(null);
  const [layouts, setLayouts] = useState({});
  const [activeLayoutName, setActiveLayoutName] = useState(null);
  const initializedRef = useRef(false);
  const restoreGenerationRef = useRef(0);
  const restoreQueueRef = useRef(Promise.resolve());
  const restoringRef = useRef(false);
  const pendingLayoutRef = useRef(null);
  const pendingSyncUrlRef = useRef(false);
  const restoreTimerRef = useRef(null);

  const restoreLayoutNow = (layout, { syncUrl = false } = {}) => {
    const generation = ++restoreGenerationRef.current;
    const themedLayout = applyThemeToLayout(layout);

    restoreQueueRef.current = restoreQueueRef.current
      .catch(() => {})
      .then(async () => {
        if (generation !== restoreGenerationRef.current) return;

        const ws = wsRef.current;
        if (!ws) return;

        restoringRef.current = true;
        try {
          clearViewerSelections(ws);
          await flushWorkspace(ws);
          await ws.restoreWorkspace(themedLayout);
          await flushWorkspace(ws);
          if (syncUrl && generation === restoreGenerationRef.current) {
            await setUrlLayout(themedLayout);
          }
        } catch (error) {
          if (generation === restoreGenerationRef.current) {
            console.warn("Failed to restore workspace layout:", error);
          }
        } finally {
          if (generation === restoreGenerationRef.current) {
            restoringRef.current = false;
          }
        }
      });

    return restoreQueueRef.current;
  };

  const scheduleRestoreLayout = (layout, { syncUrl = false } = {}) => {
    pendingLayoutRef.current = layout;
    pendingSyncUrlRef.current = pendingSyncUrlRef.current || syncUrl;

    if (restoreTimerRef.current) {
      window.clearTimeout(restoreTimerRef.current);
    }

    restoreTimerRef.current = window.setTimeout(() => {
      restoreTimerRef.current = null;
      const nextLayout = pendingLayoutRef.current;
      const shouldSyncUrl = pendingSyncUrlRef.current;
      pendingLayoutRef.current = null;
      pendingSyncUrlRef.current = false;

      if (nextLayout) {
        restoreLayoutNow(nextLayout, { syncUrl: shouldSyncUrl });
      }
    }, 150);

    return restoreQueueRef.current;
  };

  // Imperative handle for Header and other consumers
  useImperativeHandle(
    ref,
    () => ({
      getLayouts: () => layouts,
      getActiveLayout: () => activeLayoutName,
      setActiveLayout: (name) => {
        if (layouts[name]) {
          setActiveLayoutName(name);
          scheduleRestoreLayout(layouts[name], { syncUrl: true });
        }
      },
      saveLayout: async () => {
        await restoreQueueRef.current.catch(() => {});
        const ws = wsRef.current;
        if (!ws) return;
        const config = stripTransientFields(await ws.saveWorkspace());
        saveCustomLayout(config);
        setLayouts((prev) => ({ ...prev, "Custom Layout": config }));
        setActiveLayoutName("Custom Layout");
      },
      exportLayout: async () => {
        await restoreQueueRef.current.catch(() => {});
        const ws = wsRef.current;
        if (!ws) return null;
        const config = stripTransientFields(await ws.saveWorkspace());
        return JSON.stringify(config).replace(
          /PERSPECTIVE_GENERATED_/g,
          "CSP_GATEWAY_GENERATED_",
        );
      },
    }),
    [layouts, activeLayoutName],
  );

  // Bootstrap: fetch tables → load client → restore layout
  // The critical ordering is: load(client) MUST complete before restore(layout).
  // PerspectiveWorkspace's React wrapper runs restore before load (both effects
  // fire on the same render, restore is first), which crashes. Using the raw
  // element lets us control the sequence explicitly.
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws) return;

    let cancelled = false;

    (async () => {
      // 1. Fetch tables from perspective server
      const { worker, websocket, tables } = await fetchTables();
      if (cancelled) return;

      // 2. Build default layout from tables
      const defaultLayout = buildEmptyLayout();
      const theme = getCurrentTheme();

      if (processTables) {
        processTables(defaultLayout, tables, theme);
      } else {
        const sortedNames = Object.keys(tables).sort();
        sortedNames.forEach((tableName, index) => {
          const { schema } = tables[tableName];
          const generated_id = `${tableName.toUpperCase()}_GENERATED_${index + 1}`;
          defaultLayout.layout.tabs.push(generated_id);
          defaultLayout.panels[generated_id] = getDefaultViewerConfig(
            tableName,
            schema,
            theme,
          );
        });
      }

      // 3. Fetch server + custom layouts and merge
      let serverLayouts = {};
      try {
        serverLayouts = await getServerLayouts();
      } catch (e) {
        console.warn("Failed to fetch server layouts:", e);
      }
      const customLayout = getCustomLayout();
      const allLayouts = {
        Default: defaultLayout,
        ...serverLayouts,
        ...customLayout,
      };
      if (cancelled) return;

      // 4. CRITICAL: load clients FIRST, then restore layout
      //    Load local worker first (has client-server tables as local copies).
      //    Load websocket second (for server-only tables).
      //    The viewer resolves a panel's `table` name across every loaded client in order, so
      //    client-server tables match on the local worker, giving each panel an independent
      //    local table (no shared views / no scroll sync between panels).
      await ws.load(worker);
      if (websocket !== worker) {
        await ws.load(websocket);
      }
      // 5. Restore: prefer URL layout (shared link), otherwise use Default
      const urlLayout = await getUrlLayout();
      const initialLayout = urlLayout || allLayouts["Default"];
      await restoreLayoutNow(initialLayout);
      if (cancelled) return;

      // 6. Update React state (for header dropdown, etc.)
      setLayouts(allLayouts);
      setActiveLayoutName(urlLayout ? null : "Default");
      initializedRef.current = true;

      // 7. Sync layout to URL on every workspace change
      ws.addEventListener("perspective-config-update", async () => {
        if (!initializedRef.current || restoringRef.current) return;
        try {
          const config = stripTransientFields(await ws.saveWorkspace());
          if (!restoringRef.current) {
            await setUrlLayout(config);
          }
        } catch (error) {
          console.warn("Failed to sync workspace layout to URL:", error);
        }
      });

      onReady?.();
    })();

    return () => {
      cancelled = true;
      if (restoreTimerRef.current) {
        window.clearTimeout(restoreTimerRef.current);
      }
    };
  }, []);

  // Theme is element-level in Perspective 5, so panels added later inherit it from the viewer and
  // no per-panel hook is needed; `applyTheme` restores it across every mounted viewer.
  return <perspective-viewer id="workspace" ref={wsRef} />;
});

/** Clone a layout config and fill in missing panel themes */
function applyThemeToLayout(layout) {
  const theme = getCurrentTheme();
  const cloned = structuredClone(layout);
  if (cloned?.panels) {
    Object.values(cloned.panels).forEach((panel) => {
      if (!panel.theme) {
        panel.theme = getViewerTheme(theme);
      }
      panel.plugin_config = {
        ...panel.plugin_config,
        edit_mode: "SELECT_REGION",
      };
    });
  }
  return cloned;
}

function clearViewerSelections(viewer) {
  try {
    if (viewer.getSelection?.()) {
      viewer.setSelection?.();
    }
  } catch {
    // The viewer may already be mid-delete during a rapid restore.
  }
}

async function flushWorkspace(workspace) {
  try {
    await workspace.flush?.();
  } catch {
    // Perspective can reject stale panel flushes while replacing layouts.
  }
}

export default Workspace;
