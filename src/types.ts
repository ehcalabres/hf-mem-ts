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
  /** Attention cache plus persistent convolution/recurrent state; excludes working memory. */
  bytes: number;
  dtype: string;
  maxModelLen: number;
  batchSize: number;
  attentionBytes: number;
  stateBytes: number;
  convolutionBytes: number;
  recurrentBytes: number;
  convolutionDtype: string | null;
  recurrentDtype: string | null;
  layout: "attention" | "mla-compressed" | "mla-expanded" | "qwen3.5-hybrid";
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
  /** Backend allocation policy, not the attention mask. Defaults to optimized. */
  slidingWindowPolicy?: "optimized" | "full-context";
  /** MLA storage choice. Defaults to compressed (latent plus shared RoPE key). */
  mlaLayout?: "compressed" | "expanded";
  /** Qwen3.5 recurrent storage precision; config mamba_ssm_dtype or F32 if omitted. */
  recurrentStateDtype?: string;
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
  /** Allocate window-limited attention caches or full context per attention layer. */
  slidingWindowPolicy?: "optimized" | "full-context";
  /** MLA backend storage layout; defaults to compressed, not universal across engines. */
  mlaLayout?: "compressed" | "expanded";
  /** Override Qwen3.5 recurrent state storage dtype to match the backend. */
  recurrentStateDtype?: string;
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
