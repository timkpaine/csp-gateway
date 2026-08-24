import assert from "node:assert/strict";
import test from "node:test";

import { downloadLayout } from "./download.js";

test("downloadLayout submits to the same-origin attachment endpoint", () => {
  const appended = [];
  const input = {};
  const form = {
    submitted: false,
    removed: false,
    appendChild(element) {
      this.input = element;
    },
    submit() {
      this.submitted = true;
    },
    remove() {
      this.removed = true;
    },
  };
  const documentRef = {
    createElement: (tag) => (tag === "form" ? form : input),
    body: { appendChild: (element) => appended.push(element) },
  };

  globalThis.window = {
    location: { protocol: "http:", host: "gateway.example:8080" },
  };
  try {
    downloadLayout('{"layout":{}}', documentRef);
  } finally {
    delete globalThis.window;
  }

  assert.equal(form.method, "POST");
  assert.equal(
    form.action,
    "http://gateway.example:8080/api/v1/perspective/download-layout",
  );
  assert.equal(form.target, "_blank");
  assert.equal(form.hidden, true);
  assert.equal(form.input, input);
  assert.equal(input.type, "hidden");
  assert.equal(input.name, "layout");
  assert.equal(input.value, '{"layout":{}}');
  assert.equal(form.submitted, true);
  assert.equal(form.removed, true);
  assert.deepEqual(appended, [form]);
});
