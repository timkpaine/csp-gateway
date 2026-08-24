/** Get the current theme from the DOM attribute */
export const getCurrentTheme = () =>
  document.documentElement.getAttribute("data-theme") || "light";

/** Return the perspective viewer theme name for a given app theme */
export const getViewerTheme = (theme) =>
  theme === "dark" ? "Pro Dark" : "Pro Light";

/** Clone a workspace layout and stamp every panel with the current app theme */
export const applyThemeToLayout = (layout, theme = getCurrentTheme()) => {
  const cloned = structuredClone(layout);
  for (const panel of Object.values(cloned?.panels || {})) {
    panel.theme = getViewerTheme(theme);
  }
  return cloned;
};

/** Read stored theme or detect from OS preference, apply to DOM, and return it */
export const getInitialTheme = () => {
  const theme =
    localStorage.getItem("theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light");
  document.documentElement.setAttribute("data-theme", theme);
  return theme;
};

/** Apply a theme to the DOM, persist it, and update all mounted perspective viewers */
export const applyTheme = async (
  theme,
  viewers = document.querySelectorAll("perspective-viewer"),
) => {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  const viewerTheme = getViewerTheme(theme);
  const viewerList = Array.from(viewers);
  for (const viewer of viewerList) {
    viewer.setAttribute("theme", viewerTheme);
  }
  if (typeof requestAnimationFrame === "function") {
    await Promise.race([
      new Promise((resolve) => requestAnimationFrame(resolve)),
      new Promise((resolve) => setTimeout(resolve, 32)),
    ]);
  }
  const results = await Promise.allSettled(
    viewerList.map(async (viewer) => {
      await viewer.restore({ theme: viewerTheme });
      const workspace = await viewer.saveWorkspace();
      await Promise.all(
        Object.keys(workspace.panels || {}).map((panel) =>
          viewer.restore({ theme: viewerTheme }, { panel }),
        ),
      );
    }),
  );
  const failures = results.filter((result) => result.status === "rejected");
  if (failures.length) {
    throw new AggregateError(
      failures.map((result) => result.reason),
      "Failed to theme one or more Perspective viewers",
    );
  }
};
