export interface DtypeStats {
  /** Stored tensor elements, not necessarily logical model parameters for packed weights. */
  parameters: number;
  bytes: number;
}

export interface ComponentStats {
  /** Stored tensor elements, not necessarily logical model parameters for packed weights. */
  parameters: number;
  bytes: number;
  dtypes: Record<string, DtypeStats>;
}

export interface WeightMetadata {
  /** Stored tensor elements, not necessarily logical model parameters for packed weights. */
  parameters: number;
  bytes: number;
  components: Record<string, ComponentStats>;
}

export interface KvCacheEstimate {
  /** Aggregate cache tensor payload, not occupied backend pool bytes; excludes checkpoint/page overhead. */
  bytes: number;
  dtype: string;
  maxModelLen: number;
  batchSize: number;
  /** Aggregate cache payload across these tensor-parallel ranks; defaults to one rank. */
  tensorParallelSize: number;
  attentionBytes: number;
  /** Indexer keys and raw compression history, included in attentionBytes. */
  indexerBytes: number;
  stateBytes: number;
  convolutionBytes: number;
  recurrentBytes: number;
  convolutionDtype: string | null;
  recurrentDtype: string | null;
  layout: "attention" | "mla-compressed" | "mla-expanded" | "hybrid" | "recurrent";
  /** True when defaults, structural proxies, or packing approximations were needed; see assumptions. */
  approximate: boolean;
  slidingWindowPolicy: "optimized" | "full-context";
  fullAttentionLayers: number;
  slidingAttentionLayers: number;
  recurrentLayers: number;
  assumptions: string[];
}

export interface KvCacheOptions {
  maxModelLen?: number;
  batchSize?: number;
  dtype?: string;
  /** KV heads are sharded across ranks, with whole-head replication when necessary. */
  tensorParallelSize?: number;
  /** Backend allocation policy, not the attention mask. Defaults to optimized. */
  slidingWindowPolicy?: "optimized" | "full-context";
  /** MLA storage choice. Defaults to compressed (latent plus shared RoPE key). */
  mlaLayout?: "compressed" | "expanded";
  /** Recurrent storage precision; config mamba_ssm_dtype/state_dtype or F32 if omitted. */
  recurrentStateDtype?: string;
}

export interface FileEstimate extends WeightMetadata {
  kvCache: KvCacheEstimate | null;
}

export interface MmprojEstimate extends WeightMetadata {
  modelId: string;
  revision: string;
  /** Immutable Hub commit used for every file request. */
  resolvedRevision: string;
  filename: string;
}

export interface EstimateResult {
  modelId: string;
  revision: string;
  /** Immutable Hub commit; revision retains the requested branch, tag, or commit. */
  resolvedRevision: string;
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
  tensorParallelSize?: number;
  slidingWindowPolicy?: "optimized" | "full-context";
  mlaLayout?: "compressed" | "expanded";
  recurrentStateDtype?: string;
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
  /** Aggregate payload for this many tensor-parallel ranks (default 1); does not include engine pool padding. */
  tensorParallelSize?: number;
  /** Allocate window-limited attention caches or full context per attention layer. */
  slidingWindowPolicy?: "optimized" | "full-context";
  /** MLA backend storage layout; defaults to compressed, not universal across engines. */
  mlaLayout?: "compressed" | "expanded";
  /** Override recurrent state storage dtype to match the backend. */
  recurrentStateDtype?: string;
  /** Override fetch, useful for SSR, tests, proxies, or non-browser runtimes. */
  fetch?: FetchLike;
  /** Cancel target and draft requests, body reads, queued work, and retry delays. */
  signal?: AbortSignal;
  /** Per-request deadline including retries and body consumption; defaults to 30,000 ms. */
  requestTimeoutMs?: number;
  /** Retries before response delivery for transient GET/HEAD failures; defaults to 2 (maximum 10). */
  maxRetries?: number;
  /** Maximum concurrent metadata tasks per model. Defaults to 8. */
  concurrency?: number;
  /** Override the Hub base URL. */
  hubUrl?: string;
}

export interface HubFile {
  path: string;
  type: string;
}
