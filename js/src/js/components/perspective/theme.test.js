import assert from "node:assert/strict";
import test from "node:test";

import { applyTheme, applyThemeToLayout } from "./theme.js";

test("applyThemeToLayout replaces stale panel themes", () => {
  const layout = {
    panels: {
      first: { theme: "Pro Light" },
      second: {},
    },
  };

  const themed = applyThemeToLayout(layout, "dark");

  assert.equal(themed.panels.first.theme, "Pro Dark");
  assert.equal(themed.panels.second.theme, "Pro Dark");
  assert.equal(layout.panels.first.theme, "Pro Light");
  assert.equal(layout.panels.second.theme, undefined);
});

test("applyTheme updates viewer chrome and every workspace panel", async (t) => {
  const attributes = new Map();
  const stored = new Map();
  const firstCalls = [];
  const secondCalls = [];
  const viewerAttributes = [new Map(), new Map()];
  const viewers = [
    {
      setAttribute: (name, value) => viewerAttributes[0].set(name, value),
      saveWorkspace: async () => ({ panels: { first: {}, second: {} } }),
      restore: async (...args) => firstCalls.push(args),
    },
    {
      setAttribute: (name, value) => viewerAttributes[1].set(name, value),
      saveWorkspace: async () => ({ panels: { third: {} } }),
      restore: async (...args) => secondCalls.push(args),
    },
  ];

  globalThis.document = {
    documentElement: {
      setAttribute: (name, value) => attributes.set(name, value),
    },
    querySelectorAll: () => viewers,
  };
  globalThis.localStorage = {
    setItem: (name, value) => stored.set(name, value),
  };
  t.after(() => {
    delete globalThis.document;
    delete globalThis.localStorage;
  });

  const operation = applyTheme("dark");
  assert.equal(attributes.get("data-theme"), "dark");
  assert.equal(stored.get("theme"), "dark");
  assert.equal(viewerAttributes[0].get("theme"), "Pro Dark");
  assert.equal(viewerAttributes[1].get("theme"), "Pro Dark");
  await operation;

  assert.deepEqual(firstCalls, [
    [{ theme: "Pro Dark" }],
    [{ theme: "Pro Dark" }, { panel: "first" }],
    [{ theme: "Pro Dark" }, { panel: "second" }],
  ]);
  assert.deepEqual(secondCalls, [
    [{ theme: "Pro Dark" }],
    [{ theme: "Pro Dark" }, { panel: "third" }],
  ]);
});

test("applyTheme continues after one viewer cannot be serialized", async (t) => {
  const brokenCalls = [];
  const workingCalls = [];
  const viewers = [
    {
      setAttribute: () => {},
      restore: async (...args) => brokenCalls.push(args),
      saveWorkspace: async () => {
        throw new Error("Panel has no table");
      },
    },
    {
      setAttribute: () => {},
      restore: async (...args) => workingCalls.push(args),
      saveWorkspace: async () => ({ panels: { panel: {} } }),
    },
  ];
  globalThis.document = {
    documentElement: { setAttribute: () => {} },
  };
  globalThis.localStorage = { setItem: () => {} };
  t.after(() => {
    delete globalThis.document;
    delete globalThis.localStorage;
  });

  await assert.rejects(applyTheme("light", viewers), AggregateError);

  assert.deepEqual(brokenCalls, [[{ theme: "Pro Light" }]]);
  assert.deepEqual(workingCalls, [
    [{ theme: "Pro Light" }],
    [{ theme: "Pro Light" }, { panel: "panel" }],
  ]);
});

test("applyTheme starts background panel restores concurrently", async (t) => {
  const started = [];
  const pending = [];
  const viewer = {
    setAttribute: () => {},
    saveWorkspace: async () => ({ panels: { first: {}, second: {} } }),
    restore: async (_config, options) => {
      if (!options?.panel) return;
      started.push(options.panel);
      await new Promise((resolve) => pending.push(resolve));
    },
  };
  globalThis.document = {
    documentElement: { setAttribute: () => {} },
  };
  globalThis.localStorage = { setItem: () => {} };
  t.after(() => {
    delete globalThis.document;
    delete globalThis.localStorage;
  });

  const operation = applyTheme("dark", [viewer]);
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(started, ["first", "second"]);
  pending.forEach((resolve) => resolve());
  await operation;
});
