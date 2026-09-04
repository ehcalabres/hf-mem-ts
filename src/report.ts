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

function alternativeRows(result: EstimateResult, prefix = ""): Array<[string, string]> {
  if (typeof result.weightsBytes === "number") return [];
  return Object.entries(result.weightsBytes).map(([filename, weights]) => {
    const kv = typeof result.kvCacheBytes === "object" && result.kvCacheBytes
      ? result.kvCacheBytes[filename] ?? 0
      : 0;
    const detail = kv ? ` (${gib(weights)} weights + ${gib(kv)} cache)` : "";
    return [`${prefix}${filename}`, `${gib(weights + kv)}${detail}`];
  });
}

function cacheAssumptions(result: EstimateResult, label: string): string[] {
  const assumptions = new Set(Object.values(result.files).flatMap((file) => file.kvCache
    ? [`${label}: ${file.kvCache.dtype}, ${file.kvCache.maxModelLen} tokens per sequence, ${file.kvCache.batchSize} sequence(s).`]
    : []));
  return assumptions.size ? [...assumptions] : [`${label}: no cache estimate included.`];
}

export function formatResult(result: EstimateResult): string {
  const info: Array<[string, string]> = [
    ["Model ID", result.modelId],
    ["Revision", result.revision],
    ["Resolved revision", result.resolvedRevision],
    ["Format", result.format],
  ];
  if (result.filename) info.push(["File", result.filename]);

  const memory: Array<[string, string]> = [];
  if (typeof result.weightsBytes === "number") {
    memory.push(["Model", gib(result.weightsBytes)]);
    memory.push(...componentRows(result));
    if (typeof result.kvCacheBytes === "number") memory.push(["KV cache", gib(result.kvCacheBytes)]);
  } else {
    memory.push(...alternativeRows(result));
  }

  if (result.mmproj) memory.push(["Multimodal projector", `${gib(result.mmproj.bytes)} (${result.mmproj.filename})`]);
  if (result.draft) {
    if (typeof result.draft.weightsBytes === "number") {
      memory.push(["Draft model", `${gib(result.draft.weightsBytes)} (${modelReference(result.draft)})`]);
    } else {
      memory.push(["Draft model", modelReference(result.draft)]);
      memory.push(...alternativeRows(result.draft, "  "));
    }
    if (typeof result.draft.kvCacheBytes === "number") {
      memory.push(["Draft model KV cache", gib(result.draft.kvCacheBytes)]);
    }
  }
  const selections = [
    ...(typeof result.weightsBytes === "object" ? ["--gguf-file for the target"] : []),
    ...(result.draft && typeof result.draft.weightsBytes === "object" ? ["--draft-gguf-file for the draft"] : []),
  ];
  const missing = selections.length ? `select ${selections.join(" and ")}` : "incomplete estimate";
  memory.push(["Total", result.totalBytes === null ? `n/a (${missing})` : gib(result.totalBytes)]);

  const cacheDetails: string[] = [];
  for (const [label, model] of [["Target", result], ["Draft", result.draft]] as const) {
    if (!model) continue;
    for (const [filename, file] of Object.entries(model.files)) {
      const cache = file.kvCache;
      if (!cache) continue;
      cacheDetails.push(
        `${label} ${filename}: ${cache.layout}, ${cache.dtype} attention, ${cache.slidingWindowPolicy} allocation`,
        `  Context: ${cache.maxModelLen} tokens; batch: ${cache.batchSize}`,
        `  Layers: ${cache.fullAttentionLayers} full attention, ${cache.slidingAttentionLayers} sliding attention, ${cache.recurrentLayers} recurrent`,
        `  Attention payload: ${gib(cache.attentionBytes)} (${cache.attentionBytes} bytes)`,
        `  Persistent state: ${gib(cache.stateBytes)} (${cache.stateBytes} bytes)`,
      );
      if (cache.recurrentLayers) {
        cacheDetails.push(
          `  Convolution state: ${cache.convolutionBytes} bytes (${cache.convolutionDtype})`,
          `  Recurrent state: ${cache.recurrentBytes} bytes (${cache.recurrentDtype})`,
        );
      }
      cacheDetails.push(...cache.assumptions.map((assumption) => `  - ${assumption}`));
    }
  }

  return [
    "Model info",
    "----------",
    ...formatRows(info),
    "",
    "Estimated resident weights + cache",
    "----------------------------------",
    ...formatRows(memory),
    "",
    ...cacheAssumptions(result, "Target cache"),
    ...(result.draft ? cacheAssumptions(result.draft, "Draft cache") : []),
    "Excludes activations, temporary workspaces, allocator/framework overhead, and runtime weight conversion.",
    "Assumes the counted weights are resident; offloading and distributed replication are not modeled.",
    "Not a peak RAM/VRAM measurement or a guarantee that inference will fit.",
    ...(cacheDetails.length ? ["", "Cache assumptions", "-----------------", ...cacheDetails] : []),
  ].join("\n");
}
