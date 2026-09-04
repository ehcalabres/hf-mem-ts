import { fetchGgufMetadata, estimateGgufKvCache } from "./gguf.js";
import { assertPositiveInteger, checkedFetch, fetchJson, mapLimit, readJson } from "./http.js";
import { estimateSafetensorsKvCache } from "./kv-cache.js";
import { fetchSafetensorsHeader, parseSafetensorsHeaders } from "./safetensors.js";
import type { DraftModelOptions, EstimateOptions, EstimateResult, FileEstimate, FetchLike, HubFile, MmprojEstimate, WeightMetadata } from "./types.js";

const SHARD = /(.+)-(\d+)-of-(\d+)\.gguf$/i;

function urlPath(path: string): string { return path.split("/").map(encodeURIComponent).join("/"); }

function requestHeaders(token?: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function listFiles(fetcher: FetchLike, hub: string, modelId: string, revision: string, headers: HeadersInit): Promise<string[]> {
  let url: string | null = `${hub}/api/models/${urlPath(modelId)}/tree/${encodeURIComponent(revision)}?recursive=true&expand=false&limit=1000`;
  const paths: string[] = [];
  while (url) {
    const response = await checkedFetch(fetcher, url, { headers });
    const files = await readJson<HubFile[]>(response);
    paths.push(...files.filter((file) => file.type === "file").map((file) => file.path));
    const link = response.headers.get("link");
    const next = link?.split(",").find((part) => /rel="?next"?/.test(part));
    const match = next?.match(/<([^>]+)>/);
    url = match?.[1] ?? null;
  }
  return [...new Set(paths)];
}

function resolveUrl(hub: string, modelId: string, revision: string, path: string): string {
  return `${hub}/${urlPath(modelId)}/resolve/${encodeURIComponent(revision)}/${urlPath(path)}`;
}

function emptyFile(metadata: WeightMetadata, kvCache: FileEstimate["kvCache"]): FileEstimate {
  return {
    parameters: metadata.parameters,
    bytes: metadata.bytes,
    components: metadata.components,
    kvCache,
  };
}

async function safetensorsPaths(
  fetcher: FetchLike, hub: string, modelId: string, revision: string, files: string[], headers: HeadersInit,
  concurrency: number,
): Promise<Array<{ path: string; component: string }>> {
  const fileSet = new Set(files);
  let canonical: string[] = [];
  let indexFiles: string[] = [];

  if (fileSet.has("model_index.json")) {
    const modelIndex = await fetchJson<Record<string, unknown>>(
      fetcher,
      resolveUrl(hub, modelId, revision, "model_index.json"),
      headers,
    );
    for (const component of Object.keys(modelIndex).filter((key) => !key.startsWith("_"))) {
      const prefix = `${component}/`;
      const single = [
        `${prefix}diffusion_pytorch_model.safetensors`,
        `${prefix}model.safetensors`,
      ].find((path) => fileSet.has(path));
      if (single) {
        canonical.push(single);
        continue;
      }
      const index = [
        `${prefix}diffusion_pytorch_model.safetensors.index.json`,
        `${prefix}model.safetensors.index.json`,
      ].find((path) => fileSet.has(path));
      if (index) indexFiles.push(index);
    }
  } else {
    canonical = files.filter((path) => /(?:^|\/)(?:model|diffusion_pytorch_model)\.safetensors$/.test(path));
    const canonicalSet = new Set(canonical);
    indexFiles = files.filter((path) =>
      (/(?:^|\/)model\.safetensors\.index\.json$/.test(path)
        || /(?:^|\/)diffusion_pytorch_model\.safetensors\.index\.json$/.test(path))
      && !canonicalSet.has(path.replace(/\.index\.json$/, ""))
    );
  }

  const indexed = await mapLimit(indexFiles, concurrency, async (indexPath) => {
    const index = await fetchJson<{ weight_map?: Record<string, string> }>(fetcher, resolveUrl(hub, modelId, revision, indexPath), headers);
    const directory = indexPath.includes("/") ? indexPath.slice(0, indexPath.lastIndexOf("/") + 1) : "";
    const component = directory.replace(/\/$/, "") || "Transformer";
    return [...new Set(Object.values(index.weight_map ?? {}))].map((filename) => ({ path: directory + filename, component }));
  });
  const results = [
    ...canonical.map((path) => ({ path, component: path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "Transformer" })),
    ...indexed.flat(),
  ];
  return [...new Map(results.map((item) => [item.path, item])).values()];
}

async function estimateSafetensors(
  options: Required<Pick<EstimateOptions, "modelId" | "revision" | "batchSize" | "concurrency" | "hubUrl">> & EstimateOptions,
  fetcher: FetchLike, files: string[], headers: HeadersInit,
): Promise<EstimateResult> {
  const paths = await safetensorsPaths(fetcher, options.hubUrl, options.modelId, options.revision, files, headers, options.concurrency);
  if (!paths.length) throw new Error(`No supported Safetensors weights found in ${options.modelId}@${options.revision}.`);
  const fetched = await mapLimit(paths, options.concurrency, async ({ path, component }) => ({
    component, header: await fetchSafetensorsHeader(fetcher, resolveUrl(options.hubUrl, options.modelId, options.revision, path), headers),
  }));
  // Offsets are file-relative: validate each shard before combining its statistics.
  const metadata: WeightMetadata = { parameters: 0, bytes: 0, components: Object.create(null) };
  for (const { component, header } of fetched) {
    const next = parseSafetensorsHeaders({ [component]: header });
    const current = metadata.components[component] ?? { parameters: 0, bytes: 0, dtypes: Object.create(null) };
    for (const [dtype, stats] of Object.entries(next.components[component]!.dtypes)) {
      const old = current.dtypes[dtype] ?? { parameters: 0, bytes: 0 };
      current.dtypes[dtype] = { parameters: old.parameters + stats.parameters, bytes: old.bytes + stats.bytes };
    }
    current.parameters += next.parameters;
    current.bytes += next.bytes;
    metadata.parameters += next.parameters;
    metadata.bytes += next.bytes;
    if (!Number.isSafeInteger(metadata.parameters) || !Number.isSafeInteger(metadata.bytes)) {
      throw new RangeError("Safetensors totals exceed JavaScript's safe integer range.");
    }
    metadata.components[component] = current;
  }
  let kvCache = null;
  if (options.kvCache) {
    if (!files.includes("config.json")) throw new Error("KV-cache estimation requested, but config.json was not found.");
    const config = await fetchJson<Record<string, unknown>>(fetcher, resolveUrl(options.hubUrl, options.modelId, options.revision, "config.json"), headers);
    kvCache = estimateSafetensorsKvCache(config, {
      batchSize: options.batchSize,
      metadata,
      ...(options.maxModelLen !== undefined ? { maxModelLen: options.maxModelLen } : {}),
      ...(options.kvCacheDtype !== undefined ? { dtype: options.kvCacheDtype } : {}),
    });
  }
  return {
    modelId: options.modelId, revision: options.revision, format: "safetensors", filename: null,
    weightsBytes: metadata.bytes, kvCacheBytes: kvCache?.bytes ?? null,
    totalBytes: metadata.bytes + (kvCache?.bytes ?? 0), files: { safetensors: emptyFile(metadata, kvCache) },
    mmproj: null, draft: null,
  };
}

function mergeMetadata(target: FileEstimate | undefined, next: FileEstimate): FileEstimate {
  if (!target) return next;
  const parameters = target.parameters + next.parameters;
  const bytes = target.bytes + next.bytes;
  if (!Number.isSafeInteger(parameters) || !Number.isSafeInteger(bytes)) {
    throw new RangeError("Combined model metadata exceeds JavaScript's safe integer range.");
  }
  const components = { ...target.components };
  for (const [name, component] of Object.entries(next.components)) {
    const current = components[name] ?? { parameters: 0, bytes: 0, dtypes: {} };
    for (const [dtype, stats] of Object.entries(component.dtypes)) {
      const old = current.dtypes[dtype] ?? { parameters: 0, bytes: 0 };
      current.dtypes[dtype] = { parameters: old.parameters + stats.parameters, bytes: old.bytes + stats.bytes };
    }
    current.parameters += component.parameters; current.bytes += component.bytes; components[name] = current;
  }
  return { parameters, bytes, components, kvCache: target.kvCache ?? next.kvCache };
}

async function estimateGguf(
  options: Required<Pick<EstimateOptions, "modelId" | "revision" | "batchSize" | "concurrency" | "hubUrl">> & EstimateOptions,
  fetcher: FetchLike, allFiles: string[], headers: HeadersInit,
): Promise<EstimateResult> {
  let paths = allFiles.filter((path) => path.toLowerCase().endsWith(".gguf") && !path.toLowerCase().includes("mmproj-"));
  if (!paths.length) throw new Error(`No GGUF files found in ${options.modelId}@${options.revision}.`);
  if (options.ggufFile) {
    const shard = options.ggufFile.match(SHARD);
    if (shard) {
      const prefix = shard[1]!;
      paths = paths.filter((path) => path.match(SHARD)?.[1] === prefix || path.match(SHARD)?.[1]?.endsWith(`/${prefix}`));
    } else {
      paths = paths.filter((path) => path === options.ggufFile || path.endsWith(`/${options.ggufFile}`));
    }
    if (!paths.length) throw new Error(`No GGUF file matching ${options.ggufFile} was found.`);
    if (!shard && paths.length > 1) throw new Error(`Multiple GGUF files matching ${options.ggufFile} were found; pass the full path.`);
  }
  const parsed = await mapLimit(paths, options.concurrency, async (path) => {
    const metadata = await fetchGgufMetadata(fetcher, resolveUrl(options.hubUrl, options.modelId, options.revision, path), headers);
    const shard = path.match(SHARD);
    const group = shard ? `${shard[1]}.gguf` : path;
    const shouldComputeKv = Boolean(options.kvCache && (!shard || shard[2] === "1"));
    const kv = shouldComputeKv ? estimateGgufKvCache(metadata.metadata, {
      batchSize: options.batchSize,
      ...(options.maxModelLen !== undefined ? { maxModelLen: options.maxModelLen } : {}),
      ...(options.kvCacheDtype !== undefined ? { dtype: options.kvCacheDtype } : {}),
    }) : null;
    return { group, estimate: emptyFile(metadata, kv) };
  });
  const grouped: Record<string, FileEstimate> = {};
  for (const item of parsed) grouped[item.group] = mergeMetadata(grouped[item.group], item.estimate);
  const names = Object.keys(grouped);
  const selected = options.ggufFile ? grouped[names[0]!]! : null;
  const weights = Object.fromEntries(names.map((name) => [name, grouped[name]!.bytes]));
  const caches = Object.fromEntries(names.filter((name) => grouped[name]!.kvCache).map((name) => [name, grouped[name]!.kvCache!.bytes]));
  return {
    modelId: options.modelId, revision: options.revision, format: "gguf", filename: options.ggufFile ? names[0]! : null,
    weightsBytes: selected ? selected.bytes : weights,
    kvCacheBytes: selected ? selected.kvCache?.bytes ?? null : Object.keys(caches).length ? caches : null,
    totalBytes: selected ? selected.bytes + (selected.kvCache?.bytes ?? 0) : null, files: grouped,
    mmproj: null, draft: null,
  };
}

function selectMmproj(files: string[], requested: string | false | undefined): string | null {
  if (requested === false) return null;
  const candidates = files.filter((path) => /(?:^|\/)mmproj[^/]*\.gguf$/i.test(path));
  if (typeof requested === "string") {
    const matches = candidates.filter((path) => path === requested || path.endsWith(`/${requested}`));
    if (!matches.length) throw new Error(`No mmproj GGUF file matching ${requested} was found.`);
    if (matches.length > 1) throw new Error(`Multiple mmproj files matching ${requested} were found; pass the full path.`);
    return matches[0]!;
  }
  if (!candidates.length) return null;
  if (candidates.length === 1) return candidates[0]!;
  const f16 = candidates.filter((path) => /^mmproj(?:.*[-_.])?f16\.gguf$/i.test(path.slice(path.lastIndexOf("/") + 1)));
  if (f16.length === 1) return f16[0]!;
  throw new Error(`Multiple mmproj files were found (${candidates.join(", ")}); pass mmprojFile explicitly or false to exclude them.`);
}

async function estimateMmproj(
  path: string,
  options: Required<Pick<EstimateOptions, "modelId" | "revision" | "hubUrl">>,
  fetcher: FetchLike,
  headers: HeadersInit,
): Promise<MmprojEstimate> {
  const metadata = await fetchGgufMetadata(fetcher, resolveUrl(options.hubUrl, options.modelId, options.revision, path), headers);
  return {
    modelId: options.modelId,
    revision: options.revision,
    filename: path,
    parameters: metadata.parameters,
    bytes: metadata.bytes,
    components: metadata.components,
  };
}

function draftOptions(input: EstimateOptions, draft: string | DraftModelOptions): EstimateOptions {
  const selected = typeof draft === "string" ? { modelId: draft } : draft;
  return {
    modelId: selected.modelId,
    revision: selected.revision ?? "main",
    mmprojFile: false,
    ...(input.token !== undefined ? { token: input.token } : {}),
    ...(input.fetch !== undefined ? { fetch: input.fetch } : {}),
    ...(input.hubUrl !== undefined ? { hubUrl: input.hubUrl } : {}),
    ...(input.concurrency !== undefined ? { concurrency: input.concurrency } : {}),
    ...(input.kvCache !== undefined ? { kvCache: input.kvCache } : {}),
    ...((selected.maxModelLen ?? input.maxModelLen) !== undefined ? { maxModelLen: selected.maxModelLen ?? input.maxModelLen } : {}),
    ...((selected.batchSize ?? input.batchSize) !== undefined ? { batchSize: selected.batchSize ?? input.batchSize } : {}),
    ...((selected.kvCacheDtype ?? input.kvCacheDtype) !== undefined ? { kvCacheDtype: selected.kvCacheDtype ?? input.kvCacheDtype } : {}),
    ...(selected.ggufFile !== undefined ? { ggufFile: selected.ggufFile } : {}),
  };
}

function draftMatchesTarget(
  input: EstimateOptions,
  options: Required<Pick<EstimateOptions, "modelId" | "revision" | "batchSize" | "kvCacheDtype">> & EstimateOptions,
): boolean {
  if (!input.draftModel) return false;
  const draft = typeof input.draftModel === "string" ? { modelId: input.draftModel } : input.draftModel;
  return draft.modelId === options.modelId
    && (draft.revision ?? "main") === options.revision
    && draft.ggufFile === input.ggufFile
    && (draft.maxModelLen ?? options.maxModelLen) === options.maxModelLen
    && (draft.batchSize ?? options.batchSize) === options.batchSize
    && (draft.kvCacheDtype ?? options.kvCacheDtype) === options.kvCacheDtype;
}

function withAccessories(base: EstimateResult, mmproj: MmprojEstimate | null, draft: EstimateResult | null): EstimateResult {
  const totalBytes = base.totalBytes === null || draft?.totalBytes === null
    ? null
    : base.totalBytes + (mmproj?.bytes ?? 0) + (draft?.totalBytes ?? 0);
  if (totalBytes !== null && !Number.isSafeInteger(totalBytes)) {
    throw new RangeError("Total model memory exceeds JavaScript's safe integer range.");
  }
  return { ...base, totalBytes, mmproj, draft };
}

export async function estimateModelMemory(input: EstimateOptions): Promise<EstimateResult> {
  if (!input.modelId?.includes("/")) throw new Error("modelId must be a Hugging Face repository ID such as owner/model.");
  const fetcher = input.fetch ?? globalThis.fetch;
  if (!fetcher) throw new Error("No global fetch implementation is available; pass options.fetch.");
  const options = {
    revision: "main", batchSize: 1, concurrency: 8, hubUrl: "https://huggingface.co", kvCacheDtype: "auto", ...input,
  };
  options.hubUrl = options.hubUrl.replace(/\/$/, "");
  assertPositiveInteger(options.batchSize, "batchSize");
  assertPositiveInteger(options.concurrency, "concurrency");
  if (options.maxModelLen !== undefined) assertPositiveInteger(options.maxModelLen, "maxModelLen");
  const headers = requestHeaders(options.token);
  const targetPromise = (async () => {
    const files = await listFiles(fetcher, options.hubUrl, options.modelId, options.revision, headers);
    const hasSafetensors = files.some((path) => /(?:^|\/)(?:model|diffusion_pytorch_model)\.safetensors(?:\.index\.json)?$/.test(path)) || files.includes("model_index.json");
    const useGguf = Boolean(input.ggufFile || !hasSafetensors);
    const basePromise = useGguf
      ? estimateGguf(options, fetcher, files, headers)
      : estimateSafetensors(options, fetcher, files, headers);
    const mmprojPath = useGguf ? selectMmproj(files, input.mmprojFile) : null;
    const mmprojPromise = mmprojPath ? estimateMmproj(mmprojPath, options, fetcher, headers) : Promise.resolve(null);
    const [base, mmproj] = await Promise.all([basePromise, mmprojPromise]);
    return { base, mmproj };
  })();
  const draftPromise = draftMatchesTarget(input, options)
    ? targetPromise.then(({ base }) => base)
    : input.draftModel
      ? estimateModelMemory(draftOptions(input, input.draftModel))
      : Promise.resolve(null);
  const [{ base, mmproj }, draft] = await Promise.all([targetPromise, draftPromise]);
  return withAccessories(base, mmproj, draft);
}
