import { CUSTOM_LAYOUT_CONFIG_NAME } from "../../common";

/** Read custom layout from localStorage */
export const getCustomLayout = () => {
  const possibleLayout = window.localStorage.getItem(CUSTOM_LAYOUT_CONFIG_NAME);
  return possibleLayout ? { "Custom Layout": JSON.parse(possibleLayout) } : {};
};

/** Persist custom layout to localStorage */
export const saveCustomLayout = (layout) => {
  window.localStorage.setItem(
    CUSTOM_LAYOUT_CONFIG_NAME,
    JSON.stringify(layout),
  );
};

/** Fetch server-defined layouts from the gateway API */
export const getServerLayouts = async () => {
  const data = await fetch(
    `${window.location.protocol}//${window.location.host}/api/v1/perspective/layouts`,
  );
  if (data.status === 403) {
    window.location.replace(
      `${window.location.protocol}//${window.location.host}/login${window.location.search}`,
    );
  }
  const json = await data.json();
  Object.keys(json).forEach((key) => {
    json[key] = JSON.parse(json[key]);
  });
  return json;
};

/** Build an empty workspace layout shell that tables can be added into */
export const buildEmptyLayout = () => ({
  sizes: [1],
  detail: {
    main: {
      type: "tab-area",
      widgets: [],
      currentIndex: 0,
    },
  },
  master: {
    sizes: [],
    widgets: [],
  },
  mode: "globalFilters",
  viewers: {},
});
