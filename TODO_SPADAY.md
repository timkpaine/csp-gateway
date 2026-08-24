# Spaday follow-ups

## Parallelize Perspective panel theme restores

`spaday-perspective` 0.4.3 has the same panel-count-dependent theme lag as the legacy csp-gateway frontend had before its local fix.

The `<perspective-panel>` theme path currently:

1. calls `restore({theme})` for the element chrome and active panel;
1. calls `saveWorkspace()` to enumerate panels;
1. loops over every panel and awaits `restore({theme}, {panel})` sequentially.

Each restore restyles a Perspective plugin, so total latency grows with the number of tabs. Browser profiling against Perspective 5.2 showed a visibly delayed transition with eight panels. In one run, sequential completion took about 300 ms; issuing the panel restores with `Promise.all()` reduced it to about 150 ms. Exact timings vary with panel contents and render state.

Upstream action:

- Keep the initial bare `restore({theme})`; it updates element chrome and the active panel.
- After `saveWorkspace()`, restore background panel themes concurrently with `Promise.all()` rather than a serial `for ... of` loop.
- Verify concurrent restores remain safe while live tables update and while a layout replacement is queued.
- Add a multi-panel browser test that checks all saved panel themes after toggling the global theme.

## Bundle Perspective chart plugins

`spaday-perspective` 0.4.3 registers only the Datagrid plugin. A whole-element layout containing chart plugins such as `X Bar` or `Treemap` restores its geometry, but Perspective falls back to Datagrid for those panels because the requested plugins are unavailable.

csp-gateway temporarily works around this by building `spaday-charts.js` from `@perspective-dev/viewer-charts` and loading it as an additional Spaday component package. Remove that bundle and package registration once `spaday-perspective` provides chart plugins itself.

Upstream action:

- Bundle and register `@perspective-dev/viewer-charts` with `<perspective-panel>`, or expose a documented package/option that does so.
- Add a browser test that restores a multi-panel workspace containing Datagrid, `X Bar`, and `Treemap`, then verifies `saveWorkspace()` retains each requested plugin.
- Keep chart and Perspective viewer versions aligned so plugin registration uses the same Perspective runtime.
