import { getarg } from "./getarg.mjs";

import { bundleAsync } from "lightningcss";
import fs from "fs";
import path from "path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const DEBUG = getarg("--debug");

const DEFAULT_RESOLVER = {
  read(file) {
    if (/^https?:\/\//.test(file)) {
      return "";
    }

    return fs.readFileSync(file, "utf8");
  },
  resolve(specifier, originatingFile) {
    if (/^https?:\/\//.test(specifier)) {
      return specifier;
    }

    if (specifier.startsWith("perspective-viewer-")) {
      const viewerCssDir = path.resolve(
        "node_modules/@perspective-dev/viewer/dist/css",
      );
      const normalized = specifier.replace(/^perspective-viewer-/, "");
      const normalizedPath = path.join(viewerCssDir, normalized);
      if (fs.existsSync(normalizedPath)) {
        return normalizedPath;
      }
      return path.join(viewerCssDir, specifier);
    }

    if (specifier === "react-modern-drawer.css") {
      return require.resolve("react-modern-drawer/dist/index.css");
    }

    if (!specifier.startsWith(".") && !path.isAbsolute(specifier)) {
      return require.resolve(specifier, {
        paths: [path.dirname(originatingFile), process.cwd()],
      });
    }

    return path.resolve(path.dirname(originatingFile), specifier);
  },
};

const EXTERNAL_IMPORT_RE =
  /^\s*@import\s+(?:url\()?(["']?)(https?:\/\/[^"')\s]+)\1\)?[^;]*;\s*$/gim;

const get_external_imports = (file) => {
  const css = fs.readFileSync(file, "utf8");
  return Array.from(css.matchAll(EXTERNAL_IMPORT_RE), ([statement]) =>
    statement.trim(),
  );
};

const bundle_one = async (file, resolver) => {
  const { code } = await bundleAsync({
    filename: path.resolve(file),
    minify: !DEBUG,
    sourceMap: false,
    resolver: resolver || DEFAULT_RESOLVER,
  });
  const outName = path.basename(file);
  fs.mkdirSync("./dist", { recursive: true });
  const externalImports = get_external_imports(file);
  const bundledCss = new TextDecoder().decode(code);
  const css = externalImports.length
    ? `${externalImports.join("\n")}\n${bundledCss}`
    : bundledCss;
  fs.writeFileSync(path.join("./dist", outName), css);
};

export const bundle_css = async (
  root = "src/css/index.css",
  resolver = null,
) => {
  const resolved = path.resolve(root);
  if (fs.statSync(resolved).isDirectory()) {
    const files = fs.readdirSync(resolved).filter((f) => f.endsWith(".css"));
    for (const file of files) {
      await bundle_one(path.join(root, file), resolver);
    }
  } else {
    await bundle_one(root, resolver);
  }
};
