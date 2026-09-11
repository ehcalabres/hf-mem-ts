import { fetchGgufMetadata, estimateGgufKvCache } from "./gguf.js";
import { assertPositiveInteger, checkedFetch, fetchJson, mapLimit, readJson } from "./http.js";
import { estimateSafetensorsKvCache } from "./kv-cache.js";
import { fetchSafetensorsHeader, parseSafetensorsHeaders } from "./safetensors.js";
import type { DraftModelOptions, EstimateOptions, EstimateResult, FileEstimate, FetchLike, HubFile, KvCacheEstimate, KvCacheOptions, MmprojEstimate, WeightMetadata } from "./types.js";
import { requestPolicy, transportFetch, type RequestPolicy } from "./transport.js";

type ModelOptions = Required<Pick<EstimateOptions, "modelId" | "revision" | "batchSize" | "concurrency" | "hubUrl">>
  & EstimateOptions & { requestedRevision: string };

const SHARD = /(.+)-(\d+)-of-(\d+)\.gguf$/i;
const DEFAULT_TENSOR_PARALLEL_SIZES = [1, 2, 4, 8] as const;
type FileCache = Pick<FileEstimate, "kvCache" | "kvCacheByTp">;

function urlPath(path: string): string { return path.split("/").map(encodeURIComponent).join("/"); }

function requestHeaders(token?: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function listFiles(fetcher: FetchLike, hub: string, modelId: string, revision: string, headers: HeadersInit): Promise<string[]> {
  let url: string | null = `${hub}/api/models/${urlPath(modelId)}/tree/${encodeURIComponent(revision)}?recursive=true&expand=false&limit=1000`;
  const paths: string[] = [];
  const first = new URL(url);
  const seen = new Set<string>();
  while (url) {
    const page = new URL(url);
    if (page.origin !== first.origin || page.pathname !== first.pathname || page.username || page.password) {
      throw new Error("Unsafe Hub pagination URL: next page must use the same repository tree and origin.");
    }
    page.hash = "";
    page.searchParams.sort();
    if (seen.has(page.href)) throw new Error("Hub pagination loop detected.");
    seen.add(page.href);
    const response = await checkedFetch(fetcher, url, { headers });
    const files = await readJson<HubFile[]>(response);
    paths.push(...files.filter((file) => file.type === "file").map((file) => file.path));
    const link = response.headers.get("link");
    const next = link?.split(",").find((part) => /rel="?next"?/.test(part));
    const match = next?.match(/<([^>]+)>/);
    url = match ? new URL(match[1]!, url).href : null;
  }
  return [...new Set(paths)];
}

function resolveUrl(hub: string, modelId: string, revision: string, path: string): string {
  return `${hub}/${urlPath(modelId)}/resolve/${encodeURIComponent(revision)}/${urlPath(path)}`;
}

function emptyFile(metadata: WeightMetadata, cache: FileCache): FileEstimate {
  return {
    parameters: metadata.parameters,
    bytes: metadata.bytes,
    components: metadata.components,
    ...cache,
  };
}

function estimateCaches(options: ModelOptions, estimate: (options: KvCacheOptions) => KvCacheEstimate): FileCache {
  const cacheOptions: KvCacheOptions = {
    batchSize: options.batchSize,
    ...(options.maxModelLen !== undefined ? { maxModelLen: options.maxModelLen } : {}),
    ...(options.kvCacheDtype !== undefined ? { dtype: options.kvCacheDtype } : {}),
    ...(options.slidingWindowPolicy !== undefined ? { slidingWindowPolicy: options.slidingWindowPolicy } : {}),
    ...(options.mlaLayout !== undefined ? { mlaLayout: options.mlaLayout } : {}),
    ...(options.recurrentStateDtype !== undefined ? { recurrentStateDtype: options.recurrentStateDtype } : {}),
  };
  if (options.tensorParallelSize !== undefined) {
    return {
      kvCache: estimate({ ...cacheOptions, tensorParallelSize: options.tensorParallelSize }),
      kvCacheByTp: null,
    };
  }
  return {
    kvCache: null,
    kvCacheByTp: Object.fromEntries(DEFAULT_TENSOR_PARALLEL_SIZES.map((tensorParallelSize) => [
      String(tensorParallelSize), estimate({ ...cacheOptions, tensorParallelSize }),
    ])),
  };
}

function checkedTotal(...values: number[]): number {
  let total = 0;
  for (const value of values) {
    total += value;
    if (!Number.isSafeInteger(total) || total < 0) throw new RangeError("Total model memory exceeds JavaScript's safe integer range.");
  }
  return total;
}

function memoryTotals(
  files: Record<string, FileEstimate>,
  selectedName: string | null,
): Pick<EstimateResult, "kvCacheBytes" | "kvCacheBytesByTp" | "totalBytes" | "totalBytesByTp"> {
  const selected = selectedName === null ? null : files[selectedName]!;
  const entries = Object.entries(files);
  const comparing = entries.some(([, file]) => file.kvCacheByTp !== null);
  if (comparing) {
    const kvCacheBytesByTp = Object.fromEntries(DEFAULT_TENSOR_PARALLEL_SIZES.map((tp) => [
      String(tp),
      selected ? selected.kvCacheByTp![tp]!.bytes
        : Object.fromEntries(entries.map(([name, file]) => [name, file.kvCacheByTp![tp]!.bytes])),
    ]));
    const totalBytesByTp = Object.fromEntries(DEFAULT_TENSOR_PARALLEL_SIZES.map((tp) => [
      String(tp), selected ? checkedTotal(selected.bytes, selected.kvCacheByTp![tp]!.bytes) : null,
    ]));
    return { kvCacheBytes: null, kvCacheBytesByTp, totalBytes: null, totalBytesByTp };
  }
  const caches = Object.fromEntries(entries.filter(([, file]) => file.kvCache !== null).map(([name, file]) => [name, file.kvCache!.bytes]));
  return {
    kvCacheBytes: selected ? selected.kvCache?.bytes ?? null : Object.keys(caches).length ? caches : null,
    kvCacheBytesByTp: null,
    totalBytes: selected ? checkedTotal(selected.bytes, selected.kvCache?.bytes ?? 0) : null,
    totalBytesByTp: null,
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
  options: ModelOptions,
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
  let cache: FileCache = { kvCache: null, kvCacheByTp: null };
  if (options.kvCache) {
    if (!files.includes("config.json")) throw new Error("KV-cache estimation requested, but config.json was not found.");
    const config = await fetchJson<Record<string, unknown>>(fetcher, resolveUrl(options.hubUrl, options.modelId, options.revision, "config.json"), headers);
    cache = estimateCaches(options, (cacheOptions) => estimateSafetensorsKvCache(config, cacheOptions));
  }
  const estimates = { safetensors: emptyFile(metadata, cache) };
  return {
    modelId: options.modelId, revision: options.requestedRevision, resolvedRevision: options.revision, format: "safetensors", filename: null,
    weightsBytes: metadata.bytes, ...memoryTotals(estimates, "safetensors"), files: estimates,
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
  return {
    parameters, bytes, components, kvCache: target.kvCache ?? next.kvCache,
    kvCacheByTp: target.kvCacheByTp ?? next.kvCacheByTp,
  };
}

function isMmproj(path: string): boolean {
  return /(?:^|\/)mmproj[^/]*\.gguf$/i.test(path);
}

function ggufPaths(allFiles: string[], requested: string | undefined): string[] {
  let paths = allFiles.filter((path) => path.toLowerCase().endsWith(".gguf") && !isMmproj(path));
  if (requested) {
    const matches = paths.filter((path) => path === requested || path.endsWith(`/${requested}`));
    if (!matches.length) throw new Error(`No GGUF file matching ${requested} was found.`);
    if (matches.length > 1) throw new Error(`Multiple GGUF files matching ${requested} were found; pass the full path.`);
    const selected = matches[0]!;
    const shard = selected.match(SHARD);
    paths = shard ? paths.filter((path) => path.match(SHARD)?.[1] === shard[1] || path === `${shard[1]}.gguf`) : [selected];
  }
  if (!paths.length) throw new Error("No GGUF model files found.");
  const groups = new Map<string, { count: number; indices: Set<number> }>();
  const singles = new Set(paths.filter((path) => !SHARD.test(path)));
  for (const path of paths) {
    const shard = path.match(SHARD);
    if (!shard) continue;
    const index = Number(shard[2]);
    const count = Number(shard[3]);
    const name = `${shard[1]}.gguf`;
    if (!Number.isSafeInteger(count) || count < 1 || !Number.isSafeInteger(index) || index < 1 || index > count) {
      throw new Error(`Invalid GGUF shard index or count: ${path}.`);
    }
    if (singles.has(name)) throw new Error(`Ambiguous GGUF group ${name}: both sharded and unsharded files exist.`);
    const group = groups.get(name) ?? { count, indices: new Set<number>() };
    if (group.count !== count) throw new Error(`Inconsistent GGUF shard counts for ${name}.`);
    if (group.indices.has(index)) throw new Error(`Duplicate GGUF shard index ${index} for ${name}.`);
    group.indices.add(index);
    groups.set(name, group);
  }
  for (const [name, group] of groups) {
    if (group.indices.size !== group.count) throw new Error(`Incomplete GGUF shard set ${name}: expected ${group.count}, found ${group.indices.size}.`);
  }
  return paths;
}

async function estimateGguf(
  options: ModelOptions,
  fetcher: FetchLike, paths: string[], headers: HeadersInit,
): Promise<EstimateResult> {
  const parsed = await mapLimit(paths, options.concurrency, async (path) => {
    const metadata = await fetchGgufMetadata(fetcher, resolveUrl(options.hubUrl, options.modelId, options.revision, path), headers);
    const shard = path.match(SHARD);
    const group = shard ? `${shard[1]}.gguf` : path;
    const shouldComputeKv = Boolean(options.kvCache && (!shard || Number(shard[2]) === 1));
    const cache = shouldComputeKv
      ? estimateCaches(options, (cacheOptions) => estimateGgufKvCache(metadata.metadata, cacheOptions))
      : { kvCache: null, kvCacheByTp: null };
    return { group, estimate: emptyFile(metadata, cache) };
  });
  const grouped: Record<string, FileEstimate> = {};
  for (const item of parsed) grouped[item.group] = mergeMetadata(grouped[item.group], item.estimate);
  const names = Object.keys(grouped);
  const selected = options.ggufFile ? grouped[names[0]!]! : null;
  const weights = Object.fromEntries(names.map((name) => [name, grouped[name]!.bytes]));
  return {
    modelId: options.modelId, revision: options.requestedRevision, resolvedRevision: options.revision, format: "gguf", filename: options.ggufFile ? names[0]! : null,
    weightsBytes: selected ? selected.bytes : weights,
    ...memoryTotals(grouped, options.ggufFile ? names[0]! : null), files: grouped,
    mmproj: null, draft: null,
  };
}

function selectMmproj(files: string[], requested: string | false | undefined): string | null {
  if (requested === false) return null;
  const candidates = files.filter(isMmproj);
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
  options: ModelOptions,
  fetcher: FetchLike,
  headers: HeadersInit,
): Promise<MmprojEstimate> {
  const metadata = await fetchGgufMetadata(fetcher, resolveUrl(options.hubUrl, options.modelId, options.revision, path), headers);
  return {
    modelId: options.modelId,
    revision: options.requestedRevision,
    resolvedRevision: options.revision,
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
    ...((selected.tensorParallelSize ?? input.tensorParallelSize) !== undefined ? { tensorParallelSize: selected.tensorParallelSize ?? input.tensorParallelSize } : {}),
    ...((selected.slidingWindowPolicy ?? input.slidingWindowPolicy) !== undefined ? { slidingWindowPolicy: selected.slidingWindowPolicy ?? input.slidingWindowPolicy } : {}),
    ...((selected.mlaLayout ?? input.mlaLayout) !== undefined ? { mlaLayout: selected.mlaLayout ?? input.mlaLayout } : {}),
    ...((selected.recurrentStateDtype ?? input.recurrentStateDtype) !== undefined ? { recurrentStateDtype: selected.recurrentStateDtype ?? input.recurrentStateDtype } : {}),
    ...(selected.ggufFile !== undefined ? { ggufFile: selected.ggufFile } : {}),
  };
}

function draftMatchesTarget(
  input: EstimateOptions,
  options: Required<Pick<EstimateOptions, "modelId" | "revision" | "batchSize" | "kvCacheDtype">> & EstimateOptions,
): boolean {
  if (!input.draftModel) return false;
  const draft = typeof input.draftModel === "string" ? { modelId: input.draftModel } : input.draftModel;
  const draftTp = draft.tensorParallelSize ?? options.tensorParallelSize;
  const sameTopology = draftTp === options.tensorParallelSize;
  const canSelectComparison = Boolean(options.kvCache && options.tensorParallelSize === undefined
    && DEFAULT_TENSOR_PARALLEL_SIZES.some((tp) => tp === draftTp));
  return draft.modelId === options.modelId
    && (draft.revision ?? "main") === options.revision
    && draft.ggufFile === input.ggufFile
    && (draft.maxModelLen ?? options.maxModelLen) === options.maxModelLen
    && (draft.batchSize ?? options.batchSize) === options.batchSize
    && (draft.kvCacheDtype ?? options.kvCacheDtype) === options.kvCacheDtype
    && (sameTopology || canSelectComparison)
    && (draft.slidingWindowPolicy ?? options.slidingWindowPolicy) === options.slidingWindowPolicy
    && (draft.mlaLayout ?? options.mlaLayout) === options.mlaLayout
    && (draft.recurrentStateDtype ?? options.recurrentStateDtype) === options.recurrentStateDtype;
}

function selectTopology(base: EstimateResult, tp: number): EstimateResult {
  const files = Object.fromEntries(Object.entries(base.files).map(([name, file]) => {
    const kvCache = file.kvCacheByTp?.[tp];
    if (!kvCache) throw new Error(`Missing TP=${tp} cache comparison for ${name}.`);
    return [name, { ...file, kvCache, kvCacheByTp: null }];
  }));
  return { ...base, files, ...memoryTotals(files, base.format === "safetensors" ? "safetensors" : base.filename) };
}

function withAccessories(base: EstimateResult, mmproj: MmprojEstimate | null, draft: EstimateResult | null): EstimateResult {
  const totalBytes = base.totalBytes === null || draft?.totalBytes === null
    ? null
    : checkedTotal(base.totalBytes, mmproj?.bytes ?? 0, draft?.totalBytes ?? 0);
  const comparison = base.totalBytesByTp ?? draft?.totalBytesByTp;
  const totalBytesByTp = comparison ? Object.fromEntries(Object.keys(comparison).map((tp) => {
    const targetTotal = base.totalBytesByTp ? base.totalBytesByTp[tp]! : base.totalBytes;
    const draftTotal = draft ? (draft.totalBytesByTp ? draft.totalBytesByTp[tp]! : draft.totalBytes) : 0;
    return [tp, targetTotal === null || draftTotal === null
      ? null : checkedTotal(targetTotal, mmproj?.bytes ?? 0, draftTotal)];
  })) : null;
  return { ...base, totalBytes, totalBytesByTp, mmproj, draft };
}

export async function estimateModelMemory(input: EstimateOptions): Promise<EstimateResult> {
  const fetcher = input.fetch ?? globalThis.fetch;
  if (!fetcher) throw new Error("No global fetch implementation is available; pass options.fetch.");
  return estimateModel(input, fetcher, requestPolicy(input));
}

async function estimateModel(input: EstimateOptions, rawFetch: FetchLike, policy: RequestPolicy): Promise<EstimateResult> {
  if (!input.modelId?.includes("/")) throw new Error("modelId must be a Hugging Face repository ID such as owner/model.");
  const options = {
    revision: "main", batchSize: 1, concurrency: 8, hubUrl: "https://huggingface.co", kvCacheDtype: "auto", ...input,
  };
  options.hubUrl = options.hubUrl.replace(/\/$/, "");
  assertPositiveInteger(options.batchSize, "batchSize");
  assertPositiveInteger(options.concurrency, "concurrency");
  if (options.maxModelLen !== undefined) assertPositiveInteger(options.maxModelLen, "maxModelLen");
  if (options.tensorParallelSize !== undefined) assertPositiveInteger(options.tensorParallelSize, "tensorParallelSize");
  const draftInput = input.draftModel ? draftOptions(input, input.draftModel) : null;
  if (draftInput?.tensorParallelSize !== undefined) assertPositiveInteger(draftInput.tensorParallelSize, "draft tensorParallelSize");
  const fetcher = transportFetch(rawFetch, options.concurrency, policy);
  const headers = requestHeaders(options.token);
  const targetPromise = (async () => {
    const info = await fetchJson<{ sha?: string }>(
      fetcher, `${options.hubUrl}/api/models/${urlPath(options.modelId)}/revision/${encodeURIComponent(options.revision)}`, headers,
    );
    if (typeof info.sha !== "string" || !/^[a-f0-9]{40}$/i.test(info.sha)) {
      throw new Error("Hub revision lookup did not return a valid immutable commit.");
    }
    const pinned = { ...options, revision: info.sha, requestedRevision: options.revision };
    const files = await listFiles(fetcher, options.hubUrl, options.modelId, pinned.revision, headers);
    const hasSafetensors = files.some((path) => /(?:^|\/)(?:model|diffusion_pytorch_model)\.safetensors(?:\.index\.json)?$/.test(path)) || files.includes("model_index.json");
    const useGguf = Boolean(input.ggufFile || !hasSafetensors);
    // Validate every selection before starting metadata work that can reject.
    const mmprojPath = useGguf ? selectMmproj(files, input.mmprojFile) : null;
    const paths = useGguf ? ggufPaths(files, input.ggufFile) : [];
    const basePromise = useGguf
      ? estimateGguf(pinned, fetcher, paths, headers)
      : estimateSafetensors(pinned, fetcher, files, headers);
    const mmprojPromise = mmprojPath ? estimateMmproj(mmprojPath, pinned, fetcher, headers) : Promise.resolve(null);
    const [base, mmproj] = await Promise.all([basePromise, mmprojPromise]);
    return { base, mmproj };
  })();
  const draftPromise = draftMatchesTarget(input, options)
    ? targetPromise.then(({ base }) => {
      const draftTp = typeof input.draftModel === "object" ? input.draftModel.tensorParallelSize : undefined;
      return base.kvCacheBytesByTp && draftTp !== undefined ? selectTopology(base, draftTp) : base;
    })
    : draftInput
      ? estimateModel(draftInput, rawFetch, policy)
      : Promise.resolve(null);
  const [{ base, mmproj }, draft] = await Promise.all([targetPromise, draftPromise]);
  return withAccessories(base, mmproj, draft);
}
