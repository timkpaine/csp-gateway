import { bundle } from "./tools/bundle.mjs";
import { bundle_css } from "./tools/css.mjs";
import { node_modules_external } from "./tools/externals.mjs";

import fs from "fs";
import cpy from "cpy";
import path from "node:path";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

// Force all react imports to resolve to the same copy to avoid
// duplicate-React errors when dependencies (e.g. @perspective-dev/react)
// list a different React version.
const REACT_ALIAS = {
  react: path.dirname(require.resolve("react/package.json")),
  "react-dom": path.dirname(require.resolve("react-dom/package.json")),
  "react/jsx-runtime": require.resolve("react/jsx-runtime"),
};

const BUNDLES = [
  {
    entryPoints: ["./src/js/index.jsx"],
    plugins: [node_modules_external()],
    outfile: "dist/index.js",
    alias: REACT_ALIAS,
  },
  {
    entryPoints: ["./src/js/main.jsx"],
    outfile: "../csp_gateway/server/build/main.js",
    alias: REACT_ALIAS,
    publicPath: "static",
  },
];

const WASM_ASSETS = [
  "node_modules/@perspective-dev/server/dist/wasm/perspective-server.wasm",
  "node_modules/@perspective-dev/viewer/dist/wasm/perspective-viewer.wasm",
];

function copy_wasm_assets(outdir) {
  fs.mkdirSync(outdir, { recursive: true });
  for (const wasm of WASM_ASSETS) {
    fs.copyFileSync(wasm, path.join(outdir, path.basename(wasm)));
  }
}

async function build() {
  // Bundle css
  await bundle_css();

  // Copy images
  await cpy("src/img/*", "dist/", { flat: true });

  // Copy Perspective wasm assets for /static/*.wasm requests.
  copy_wasm_assets("dist");

  await Promise.all(BUNDLES.map(bundle)).catch(() => process.exit(1));

  // Copy servable assets to python extension (exclude esm/)
  fs.mkdirSync("../csp_gateway/server/build", { recursive: true });
  await cpy("dist/**/*", "../csp_gateway/server/build", { flat: true });
}

build();
