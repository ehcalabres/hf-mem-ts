export interface DtypeStats {
  parameters: number;
  bytes: number;
}

export interface ComponentStats {
  parameters: number;
  bytes: number;
  dtypes: Record<string, DtypeStats>;
}

export interface WeightMetadata {
  parameters: number;
  bytes: number;
  components: Record<string, ComponentStats>;
}

export interface KvCacheEstimate {
  bytes: number;
  dtype: string;
  maxModelLen: number;
  batchSize: number;
}

export interface FileEstimate extends WeightMetadata {
  kvCache: KvCacheEstimate | null;
}

export interface MmprojEstimate extends WeightMetadata {
  modelId: string;
  revision: string;
  filename: string;
}

export interface EstimateResult {
  modelId: string;
  revision: string;
  format: "safetensors" | "gguf";
  /** Set when a single GGUF (or one sharded GGUF set) was requested. */
  filename: string | null;
  weightsBytes: number | Record<string, number>;
  kvCacheBytes: number | Record<string, number> | null;
  /** Null when the result contains multiple alternative GGUF quantizations. */
  totalBytes: number | null;
  files: Record<string, FileEstimate>;
  /** Multimodal projector loaded alongside a GGUF model, if present and enabled. */
  mmproj: MmprojEstimate | null;
  /** Draft model loaded alongside the target model for speculative decoding. */
  draft: EstimateResult | null;
}

export type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export interface DraftModelOptions {
  modelId: string;
  revision?: string;
  ggufFile?: string;
  maxModelLen?: number;
  batchSize?: number;
  kvCacheDtype?: string;
}

export interface EstimateOptions {
  modelId: string;
  revision?: string;
  token?: string;
  /** Select a GGUF file. A shard name selects and combines its full shard set. */
  ggufFile?: string;
  /**
   * GGUF multimodal projector selection. Omit to auto-select a sole projector or
   * mmproj-F16; pass a filename explicitly, or false to exclude it.
   */
  mmprojFile?: string | false;
  /** Add the resident memory of a draft model used for speculative decoding. */
  draftModel?: string | DraftModelOptions;
  /** Include a KV-cache estimate. Defaults to false. */
  kvCache?: boolean;
  maxModelLen?: number;
  batchSize?: number;
  /** Safetensors aliases (auto, bfloat16, fp8...) or a GGUF dtype (F16, Q8_0...). */
  kvCacheDtype?: string;
  /** Override fetch, useful for SSR, tests, proxies, or non-browser runtimes. */
  fetch?: FetchLike;
  /** Maximum simultaneous metadata requests. Defaults to 8. */
  /** Maximum concurrent metadata tasks per model. Defaults to 8. */
  concurrency?: number;
  /** Override the Hub base URL. */
  hubUrl?: string;
}

export interface HubFile {
  path: string;
  type: string;
}
