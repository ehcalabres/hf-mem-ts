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
  const assumptions = new Set(Object.values(result.files).flatMap((file) => {
    const caches = file.kvCache ? [file.kvCache] : Object.values(file.kvCacheByTp ?? {});
    return caches.map((cache) => `${label}: ${cache.dtype}, ${cache.maxModelLen} tokens per sequence, ${cache.batchSize} sequence(s).`);
  }));
  return assumptions.size ? [...assumptions] : [`${label}: no cache estimate included.`];
}

function comparisonRows(result: EstimateResult, missing: string): Array<[string, string]> {
  const rows: Array<[string, string]> = [];
  const payload = (bytes: number): string => `${gib(bytes)} (${bytes} bytes)`;
  const cacheValue = (value: EstimateResult["kvCacheBytes"]): string => typeof value === "number"
    ? payload(value)
    : value ? "by GGUF file (below)" : "not included";
  const draftTp = result.draft && !result.draft.kvCacheBytesByTp
    ? Object.values(result.draft.files).find((file) => file.kvCache)?.kvCache?.tensorParallelSize
    : undefined;
  for (const [tp, total] of Object.entries(result.totalBytesByTp ?? {})) {
    const targetCache = result.kvCacheBytesByTp?.[tp] ?? null;
    const draftCache = result.draft?.kvCacheBytesByTp?.[tp] ?? result.draft?.kvCacheBytes ?? null;
    const values = [`target cache ${cacheValue(targetCache)}`];
    if (result.draft) values.push(`draft cache${draftTp !== undefined ? ` (fixed TP ${draftTp})` : ""} ${cacheValue(draftCache)}`);
    values.push(`combined total ${total === null ? `n/a (${missing})` : payload(total)}`);
    rows.push([`TP ${tp}`, values.join("; ")]);
    for (const [label, cache] of [["Target", targetCache], ["Draft", draftCache]] as const) {
      if (cache && typeof cache === "object") {
        for (const [filename, bytes] of Object.entries(cache)) {
          rows.push([`  TP ${tp} ${label} ${filename} cache`, payload(bytes)]);
        }
      }
    }
  }
  return rows;
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
    if (typeof result.kvCacheBytes === "number") memory.push(["KV cache payload", gib(result.kvCacheBytes)]);
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
      memory.push(["Draft model cache payload", gib(result.draft.kvCacheBytes)]);
    }
  }
  const selections = [
    ...(typeof result.weightsBytes === "object" ? ["--gguf-file for the target"] : []),
    ...(result.draft && typeof result.draft.weightsBytes === "object" ? ["--draft-gguf-file for the draft"] : []),
  ];
  const missing = selections.length ? `select ${selections.join(" and ")}` : "incomplete estimate";
  if (result.totalBytesByTp) {
    memory.push(...comparisonRows(result, missing));
  } else {
    memory.push(["Total", result.totalBytes === null ? `n/a (${missing})` : gib(result.totalBytes)]);
  }

  const cacheDetails: string[] = [];
  for (const [label, model] of [["Target", result], ["Draft", result.draft]] as const) {
    if (!model) continue;
    for (const [filename, file] of Object.entries(model.files)) {
      if (file.kvCacheByTp) {
        const entries = Object.entries(file.kvCacheByTp);
        const first = entries[0]?.[1];
        if (!first) continue;
        cacheDetails.push(
          `${label} ${filename}: ${first.layout}, ${first.dtype} attention, ${first.slidingWindowPolicy} allocation`,
          `  TP comparison: ${entries.map(([tp]) => tp).join(", ")}; payload summed across ranks, alternatives are not additive.`,
          `  Estimate: ${entries.some(([, cache]) => cache.approximate) ? "approximate (see assumptions)" : "metadata-derived (runtime overhead excluded)"}`,
          `  Context: ${first.maxModelLen} tokens; batch: ${first.batchSize}`,
          `  Layers: ${first.fullAttentionLayers} full attention, ${first.slidingAttentionLayers} sliding attention, ${first.recurrentLayers} recurrent`,
        );
        if (first.recurrentLayers) {
          cacheDetails.push(`  State precision: ${first.convolutionDtype} convolution, ${first.recurrentDtype} recurrent`);
        }
        const assumptions = new Map<string, string[]>();
        for (const [tp, cache] of entries) {
          for (const assumption of new Set(cache.assumptions)) {
            const tps = assumptions.get(assumption) ?? [];
            tps.push(tp);
            assumptions.set(assumption, tps);
          }
        }
        for (const [assumption, tps] of assumptions) {
          cacheDetails.push(`  - ${tps.length === entries.length ? "" : `[TP ${tps.join(", ")}] `}${assumption}`);
        }
        if (first.recurrentLayers) cacheDetails.push("  vLLM occupied-pool memory: not predicted; depends on cache groups, page padding and checkpoint policy.");
        cacheDetails.push("  Full per-TP attention/indexer/state breakdowns are available with --json.");
        continue;
      }
      const cache = file.kvCache;
      if (!cache) continue;
      cacheDetails.push(
        `${label} ${filename}: ${cache.layout}, ${cache.dtype} attention, ${cache.slidingWindowPolicy} allocation`,
        `  Estimate: ${cache.approximate ? "approximate (see assumptions)" : "metadata-derived (runtime overhead excluded)"}`,
        `  Context: ${cache.maxModelLen} tokens; batch: ${cache.batchSize}; tensor parallel: ${cache.tensorParallelSize}`,
        `  Layers: ${cache.fullAttentionLayers} full attention, ${cache.slidingAttentionLayers} sliding attention, ${cache.recurrentLayers} recurrent`,
        `  Attention payload: ${gib(cache.attentionBytes)} (${cache.attentionBytes} bytes)`,
        `    Indexer payload (included above): ${gib(cache.indexerBytes)} (${cache.indexerBytes} bytes)`,
        `  Persistent state: ${gib(cache.stateBytes)} (${cache.stateBytes} bytes)`,
      );
      if (cache.recurrentLayers) {
        cacheDetails.push(
          `  Convolution state: ${cache.convolutionBytes} bytes (${cache.convolutionDtype})`,
          `  Recurrent state: ${cache.recurrentBytes} bytes (${cache.recurrentDtype})`,
        );
      }
      cacheDetails.push(...cache.assumptions.map((assumption) => `  - ${assumption}`));
      if (cache.recurrentLayers) cacheDetails.push("  vLLM occupied-pool memory: not predicted; depends on cache groups, page padding and checkpoint policy.");
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
    ...(cacheDetails.length ? ["", "Cache assumptions", "-----------------", ...cacheDetails] : []),
    "",
    ...cacheAssumptions(result, "Target cache"),
    ...(result.draft ? cacheAssumptions(result.draft, "Draft cache") : []),
    "Excludes activations, temporary workspaces, allocator/framework overhead, and runtime weight conversion.",
    "Assumes the counted weights are resident; offloading and distributed replication are not modeled.",
    "Not a peak RAM/VRAM measurement or a guarantee that inference will fit.",
  ].join("\n");
}
