import { httpUrl } from "../../common.js";

export const downloadLayout = (json, documentRef = document) => {
  const form = documentRef.createElement("form");
  form.method = "POST";
  form.action = httpUrl("/api/v1/perspective/download-layout");
  form.target = "_blank";
  form.hidden = true;
  const input = documentRef.createElement("input");
  input.type = "hidden";
  input.name = "layout";
  input.value = json;
  form.appendChild(input);
  documentRef.body.appendChild(form);
  form.submit();
  form.remove();
};
