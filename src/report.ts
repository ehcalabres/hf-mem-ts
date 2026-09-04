import type { EstimateResult } from "./types.js";

function gib(bytes: number): string {
  return `${(bytes / 2 ** 30).toFixed(2)} GiB`;
}

function formatRows(rows: Array<[label: string, value: string]>): string[] {
  const width = Math.max(...rows.map(([label]) => label.length));
  return rows.map(([label, value]) => `${`${label}:`.padEnd(width + 2)}${value}`);
}

function modelReference(result: EstimateResult): string {
  const file = result.filename ? `, ${result.filename}` : "";
  return `${result.modelId}@${result.revision}${file}`;
}

function componentRows(result: EstimateResult): Array<[string, string]> {
  const totals = new Map<string, number>();
  for (const file of Object.values(result.files)) {
    for (const [name, component] of Object.entries(file.components)) {
      totals.set(name, (totals.get(name) ?? 0) + component.bytes);
    }
  }
  if (totals.size < 2) return [];
  return [...totals].map(([name, bytes]) => [
    `  ${name.replaceAll("_", " ").toUpperCase()}`,
    gib(bytes),
  ]);
}

export function formatResult(result: EstimateResult): string {
  const info: Array<[string, string]> = [
    ["Model ID", result.modelId],
    ["Revision", result.revision],
    ["Format", result.format],
  ];
  if (result.filename) info.push(["File", result.filename]);

  const memory: Array<[string, string]> = [];
  if (typeof result.weightsBytes === "number") {
    memory.push(["Model", gib(result.weightsBytes)]);
    memory.push(...componentRows(result));
    if (typeof result.kvCacheBytes === "number") memory.push(["KV cache", gib(result.kvCacheBytes)]);
  } else {
    for (const [filename, weights] of Object.entries(result.weightsBytes)) {
      const kv = typeof result.kvCacheBytes === "object" && result.kvCacheBytes
        ? result.kvCacheBytes[filename] ?? 0
        : 0;
      const detail = kv ? ` (${gib(weights)} weights + ${gib(kv)} KV)` : "";
      memory.push([filename, `${gib(weights + kv)}${detail}`]);
    }
  }

  if (result.mmproj) memory.push(["Multimodal projector", `${gib(result.mmproj.bytes)} (${result.mmproj.filename})`]);
  if (result.draft?.totalBytes !== null && result.draft) {
    if (typeof result.draft.weightsBytes === "number") {
      memory.push(["Draft model", `${gib(result.draft.weightsBytes)} (${modelReference(result.draft)})`]);
    }
    if (typeof result.draft.kvCacheBytes === "number") {
      memory.push(["Draft model KV cache", gib(result.draft.kvCacheBytes)]);
    }
  }
  memory.push(["Total", result.totalBytes === null ? "n/a (select a GGUF file)" : gib(result.totalBytes)]);

  return [
    "Model info",
    "----------",
    ...formatRows(info),
    "",
    "Total memory requirements",
    "-------------------------",
    ...formatRows(memory),
  ].join("\n");
}
