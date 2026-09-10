import { SAFETENSORS_DTYPE_BYTES } from "./safetensors.js";
import type { KvCacheEstimate, KvCacheOptions } from "./types.js";

type JsonConfig = Record<string, unknown>;
type LayerKind = "full" | "sliding" | "recurrent" | "none";
type Approximate = (reason: string) => void;

const ALIASES: Record<string, readonly string[]> = {
  num_hidden_layers: ["decoder_layers", "num_decoder_layers", "n_layer", "n_layers", "num_layers", "num_blocks"],
  hidden_size: ["n_embd", "d_model", "dim", "model_dim", "embedding_size"],
  num_attention_heads: ["decoder_attention_heads", "n_head", "n_heads", "num_heads", "attn_config.n_heads"],
  num_key_value_heads: ["num_kv_heads", "n_head_kv", "n_kv_heads", "kv_n_heads", "multi_query_group_num", "attn_config.kv_n_heads"],
  max_position_embeddings: ["n_positions", "n_ctx", "max_seq_len", "max_sequence_length", "seq_length", "seq_len", "model_max_length"],
  head_dim: ["attention_head_dim", "d_head"],
  key_head_dim: ["key_dim", "key_length", "qk_head_dim"],
  value_head_dim: ["v_head_dim", "value_dim", "value_length"],
  global_head_dim: ["head_dim_global"],
  global_num_key_value_heads: ["num_key_value_heads_global", "num_global_key_value_heads", "global_num_kv_heads"],
  layer_types: ["layers_block_type", "block_types"],
  state_size: ["d_state", "ssm_cfg.d_state", "ssm_cfg.state_size"],
  intermediate_size: ["d_inner", "ssm_cfg.d_inner"],
  conv_kernel: ["d_conv", "conv_kernel_size", "ssm_cfg.d_conv"],
  expand: ["expand_factor", "ssm_cfg.expand"],
  n_groups: ["num_groups", "ssm_cfg.ngroups", "ssm_cfg.n_groups"],
  index_head_dim: ["indexer_head_dim"],
  index_kv_heads: ["indexer_kv_heads"],
  index_compress_ratio: ["indexer_compress_ratio"],
};

function record(value: unknown): value is JsonConfig {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function field(config: JsonConfig, path: string): unknown {
  let value: unknown = config;
  for (const key of path.split(".")) {
    if (!record(value) || !Object.hasOwn(value, key)) return undefined;
    value = value[key];
  }
  return value;
}

function normalize(raw: JsonConfig): JsonConfig {
  if (!record(raw)) throw new Error("Cache configuration must be an object.");
  let config = { ...raw };
  const seen = new Set<unknown>([raw]);
  for (;;) {
    const nested = ["text_config", "llm_config", "language_config", "language_model_config", "decoder"]
      .map((key) => config[key]).find(record);
    if (!nested) break;
    if (seen.has(nested)) throw new Error("Cyclic nested cache configuration.");
    seen.add(nested);
    for (const key of ["text_config", "llm_config", "language_config", "language_model_config", "decoder"]) delete config[key];
    // A text decoder's dtype takes precedence over an outer multimodal wrapper's dtype.
    if (nested.dtype != null || nested.torch_dtype != null) { delete config.dtype; delete config.torch_dtype; }
    config = { ...config, ...nested };
  }
  for (const [key, aliases] of Object.entries(ALIASES)) {
    if (config[key] == null) config[key] = aliases.map((alias) => field(config, alias)).find((value) => value != null);
  }
  if (config.num_key_value_heads == null && (config.multi_query === true || config.multi_query_attention === true || field(config, "attn_config.attn_type") === "multiquery_attention")) {
    config.num_key_value_heads = 1;
  }
  return config;
}

function positive(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) throw new RangeError(`${name} must be a positive safe integer.`);
  return value as number;
}

function safe(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError("Cache estimate exceeds JavaScript's safe integer range.");
  return value;
}

function product(...values: number[]): number {
  let result = 1;
  for (const value of values) result = safe(result * value);
  return result;
}

function sum(...values: number[]): number {
  let result = 0;
  for (const value of values) result = safe(result + value);
  return result;
}

function dtypeName(value: unknown): string {
  if (typeof value !== "string") throw new Error("Cache dtype must be a string.");
  const normalized = value.replace(/^torch\./, "").toLowerCase();
  const aliases: Record<string, string> = {
    bfloat16: "BF16", bf16: "BF16", float16: "F16", fp16: "F16", f16: "F16", half: "F16",
    float32: "F32", fp32: "F32", f32: "F32", fp8: "F8_E4M3", fp8_ds_mla: "F8_E4M3",
    fp8_inc: "F8_E4M3", fp8_e4m3: "F8_E4M3", fp8_e5m2: "F8_E5M2",
    float8_e4m3: "F8_E4M3", float8_e4m3fn: "F8_E4M3", float8_e5m2: "F8_E5M2",
    int8: "I8", uint8: "U8",
  };
  const dtype = Object.hasOwn(aliases, normalized) ? aliases[normalized]! : value.toUpperCase();
  if (!["BF16", "F16", "F32", "F8_E4M3", "F8_E5M2", "I8", "U8"].includes(dtype)) {
    throw new Error(`Unsupported cache dtype: ${value}; specify a supported storage precision.`);
  }
  return dtype;
}

function inferredDtype(values: unknown[], label: string, approximate: Approximate): string | undefined {
  let selected: string | undefined;
  for (const value of values) {
    if (value == null || (typeof value === "string" && value.toLowerCase() === "auto")) continue;
    let next: string;
    try { next = dtypeName(value); }
    catch { approximate(`${label} ${String(value)} is not understood; falling back to another declared precision or F16.`); continue; }
    if (selected && next !== selected) {
      approximate(`Conflicting ${label} declarations; using the wider storage precision (${selected}, ${next}).`);
      if (SAFETENSORS_DTYPE_BYTES[next]! > SAFETENSORS_DTYPE_BYTES[selected]!) selected = next;
    } else selected = next;
  }
  return selected;
}

function computeDtype(config: JsonConfig, approximate: Approximate): string {
  const dtype = inferredDtype([config.dtype, config.torch_dtype], "compute dtype", approximate);
  if (dtype) return dtype;
  approximate("Compute/cache precision is absent or unrecognized; assuming F16 (2 bytes per element), independently of weight quantization.");
  return "F16";
}

function cacheDtype(config: JsonConfig, requested: string | undefined, approximate: Approximate): string {
  if (requested !== undefined && requested.toLowerCase() !== "auto") return dtypeName(requested);
  const quantization = record(config.quantization_config) ? config.quantization_config : {};
  const scheme = quantization.kv_cache_scheme;
  let schemeDtype: string | undefined;
  if (record(scheme) && scheme.num_bits === 8) schemeDtype = scheme.type === "float" ? "F8_E4M3" : scheme.type === "int" ? "I8" : undefined;
  if (scheme != null && !schemeDtype) approximate("Unrecognized KV quantization scheme; using declared cache/compute precision instead.");
  return inferredDtype([config.kv_cache_dtype, quantization.kv_cache_dtype, schemeDtype], "explicit cache dtype", approximate) ?? computeDtype(config, approximate);
}

export function resolveSafetensorsKvDtype(config: JsonConfig, requested = "auto"): string {
  return cacheDtype(normalize(config), requested, () => {});
}

/** Estimate from tensor/cache capabilities, never from a model-name allowlist. */
export function estimateSafetensorsKvCache(rawConfig: JsonConfig, options: KvCacheOptions = {}): KvCacheEstimate {
  const config = normalize(rawConfig);
  const assumptions = new Set<string>(["Resident cache payload only; excludes runtime workspaces, allocator overhead, paging and offloading. Metadata-derived does not mean exact runtime allocation."]);
  let approximate = false;
  const note: Approximate = (reason) => { approximate = true; assumptions.add(reason); };
  const suppliedTypes = config.layer_types;
  if (suppliedTypes != null && (!Array.isArray(suppliedTypes) || !suppliedTypes.every((value) => typeof value === "string"))) throw new Error("layer_types must be an array of strings.");
  const layerTypes = (suppliedTypes as string[] | undefined) ?? [];
  const layers = positive(config.num_hidden_layers ?? (layerTypes.length || undefined), "num_hidden_layers (or layer_types length)");
  const rawLength = options.maxModelLen ?? config.max_position_embeddings;
  if (rawLength == null) note("Context length is absent; assuming 4096 tokens. Pass maxModelLen to match the workload.");
  const maxModelLen = positive(rawLength ?? 4096, "maxModelLen");
  const batchSize = positive(options.batchSize ?? 1, "batchSize");
  const tensorParallelSize = positive(options.tensorParallelSize ?? 1, "tensorParallelSize");
  const allocatedHeads = (count: number): number => {
    const allocated = product(Math.ceil(count / tensorParallelSize), tensorParallelSize);
    if (allocated !== count) note("Tensor parallelism replicates/rounds KV heads to at least one whole head per rank; aggregate payload includes those copies.");
    return allocated;
  };
  if (tensorParallelSize > 1) note(`Payload is summed across ${tensorParallelSize} tensor-parallel ranks: head-sharded attention, replicated compressed latents/index keys, and sharded recurrent state. Backend-specific padding/replicated auxiliary states are not modeled.`);
  const slidingWindowPolicy = options.slidingWindowPolicy ?? "optimized";
  if (!["optimized", "full-context"].includes(slidingWindowPolicy)) throw new Error("slidingWindowPolicy must be optimized or full-context.");
  const mlaLayout = options.mlaLayout ?? "compressed";
  if (!["compressed", "expanded"].includes(mlaLayout)) throw new Error("mlaLayout must be compressed or expanded.");
  if (options.recurrentStateDtype !== undefined) dtypeName(options.recurrentStateDtype);
  const dtype = cacheDtype(config, options.dtype, note);
  const width = SAFETENSORS_DTYPE_BYTES[dtype]!;
  if (layerTypes.length && layerTypes.length !== layers) note("layer_types length differs from the layer count; repeating/truncating the supplied schedule.");

  const at = (key: string, index: number, allowZero = false): number | undefined => {
    let value = config[key];
    if (Array.isArray(value)) {
      if (!value.length) { note(`${key} is an empty per-layer array; ignoring it.`); return undefined; }
      if (value.length !== layers) note(`${key} has a different length than the layer count; repeating/truncating its per-layer values.`);
      value = value[index % value.length];
    }
    if (value == null) return undefined;
    if (allowZero && value === 0) return undefined;
    return positive(value, key);
  };
  const hasLinear = ["linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"].every((key) => config[key] != null);
  const hasSsm = config.state_size != null;
  const stateHints = hasSsm || config.linear_num_value_heads != null || config.ssm_cfg != null;
  const interval = config.full_attention_interval == null ? undefined : positive(config.full_attention_interval, "full_attention_interval");
  const windowPattern = config.sliding_window_pattern == null ? undefined : positive(config.sliding_window_pattern, "sliding_window_pattern");
  const window = config.sliding_window == null || config.sliding_window === 0 || config.sliding_window === -1 ? undefined : positive(config.sliding_window, "sliding_window");
  const maxWindowLayers = config.max_window_layers == null ? undefined : safe(config.max_window_layers as number);
  if (maxWindowLayers !== undefined && window && config.use_sliding_window !== false && !layerTypes.length) note("max_window_layers is interpreted as the first windowed layers; provide explicit layer_types for a different schedule.");
  if (stateHints && !layerTypes.length && !interval) note("State dimensions have no mixed-attention schedule; treating all layers as recurrent.");
  if (config.mlp_only_layers != null) assumptions.add("MLP/MoE-only hints do not remove attention caches; only explicit mlp/feed_forward layer types do.");
  if ([config.num_kv_shared_layers, config.num_kv_shared_layers_pattern, config.shared_kv_layers].some((value) => value != null && value !== 0 && value !== false) || config.attention_k_eq_v === true) {
    note("Cache-sharing hints are present; assuming independent layer/key/value buffers, which can overestimate shared storage.");
  }
  if (config.attention_causal === false) note("Non-causal attention: estimating retained K/V projections; an encoder backend may not retain these caches.");
  const crossAttention = config.is_encoder_decoder === true || config.add_cross_attention === true || config.cross_attention_hidden_size != null;
  if (crossAttention) note("Encoder-decoder/cross-attention storage is approximated with an additional full-length decoder-sized K/V cache; source length is assumed equal to the requested context.");

  const kindAt = (index: number): LayerKind => {
    const type = layerTypes.length ? layerTypes[index % layerTypes.length]!.toLowerCase() : undefined;
    if (type) {
      if (["attention", "full_attention", "global_attention", "self_attention"].includes(type)) return "full";
      if (["sliding_attention", "sliding_window", "local_attention", "window_attention"].includes(type)) return "sliding";
      if (["linear_attention", "recurrent", "ssm", "mamba", "mamba2", "rwkv"].includes(type)) return "recurrent";
      if (["mlp", "feed_forward", "ffn"].includes(type)) return "none";
      note(`Unknown layer type ${type}: using a full-attention K/V proxy; additional model-specific state is not separately modeled.`);
      return "full";
    }
    if ([config.num_attention_heads, config.num_key_value_heads].some((value) => Array.isArray(value) && value.length && value[index % value.length] === 0)) return "recurrent";
    if (interval) return (index + 1) % interval === 0 ? "full" : "recurrent";
    if (windowPattern) return (index + 1) % windowPattern === 0 ? "full" : "sliding";
    if (stateHints) return "recurrent";
    if (window && config.use_sliding_window !== false && (maxWindowLayers === undefined || index < maxWindowLayers)) return "sliding";
    return "full";
  };

  const attentionElements = (index: number, global: boolean, recurrentProxy = false): { elements: number; mla: boolean } => {
    const get = (key: string): number | undefined => (global ? at(`global_${key}`, index) : undefined) ?? at(key, index, recurrentProxy);
    const rank = at("kv_lora_rank", index, true);
    const rope = at("qk_rope_head_dim", index, true);
    if (rank && mlaLayout === "compressed") {
      if (config.qk_rope_head_dim == null) note("MLA RoPE dimensions are absent; estimating the shared latent without an additional RoPE key.");
      assumptions.add("MLA capability fields select a shared compressed KV latent plus shared RoPE key per token; requires a compressed-cache backend.");
      return { elements: product(sum(rank, rope ?? 0), tensorParallelSize), mla: true };
    }
    const hidden = at("hidden_size", index);
    const explicitHeads = get("num_attention_heads");
    const explicitKvHeads = get("num_key_value_heads");
    const explicitDim = get("head_dim");
    let heads = explicitHeads ?? explicitKvHeads;
    if (!heads && hidden) {
      const referenceDim = explicitDim ?? get("key_head_dim") ?? get("value_head_dim") ?? hidden;
      heads = Math.ceil(hidden / referenceDim);
      note("Attention head count is missing; inferring an ungrouped MHA projection from hidden width and available head dimensions.");
    } else if (explicitHeads === undefined && explicitKvHeads !== undefined && !explicitDim) {
      note("Query-head count is missing; using KV heads as the query-head count when deriving head width. This can overestimate grouped-query storage.");
    }
    if (!heads) throw new Error("Insufficient cache dimensions: provide attention/KV heads and head dimensions, or hidden_size.");
    const kvHeads = explicitKvHeads ?? heads;
    if (explicitKvHeads === undefined) note("KV head count is absent; assuming multi-head attention (KV heads equal query heads), not grouped-query attention.");
    if (heads % kvHeads) note("Query/KV head counts do not divide evenly; using the explicitly supplied KV-head count.");
    let fallbackDim = explicitDim;
    if (!fallbackDim && hidden) {
      fallbackDim = Math.ceil(hidden / heads);
      if (hidden % heads) note("hidden_size / attention heads is fractional; rounding each head dimension upward.");
    }
    const nope = at("qk_nope_head_dim", index, true);
    const keyDim = get("key_head_dim") ?? (nope ? sum(nope, rope ?? 0) : undefined) ?? fallbackDim ?? rope;
    const valueDim = get("value_head_dim") ?? fallbackDim;
    if (!keyDim || !valueDim) throw new Error("Insufficient cache dimensions: provide key/value head dimensions, head_dim, or hidden_size.");
    if (rank) assumptions.add("Expanded MLA retains separate per-query-head keys and values, without also retaining the compressed latent.");
    else assumptions.add("Attention stores separate key and value projections per KV head.");
    return { elements: product(allocatedHeads(rank ? heads : kvHeads), sum(keyDim, valueDim)), mla: Boolean(rank) };
  };

  let attentionBytes = 0;
  let indexerBytes = 0;
  let convolutionBytes = 0;
  let recurrentBytes = 0;
  let convolutionDtype: string | null = null;
  let recurrentDtype: string | null = null;
  let fullAttentionLayers = 0;
  let slidingAttentionLayers = 0;
  let recurrentLayers = 0;
  let mlaLayers = 0;
  const perLayerFields = [
    "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "key_head_dim", "value_head_dim",
    "global_head_dim", "global_num_attention_heads", "global_num_key_value_heads", "global_key_head_dim", "global_value_head_dim",
    "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "index_head_dim", "index_kv_heads", "index_compress_ratio", "intermediate_size", "expand",
    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
    "state_size", "conv_kernel", "n_groups",
  ];
  const varying = layerTypes.length > 0 || interval !== undefined || windowPattern !== undefined || maxWindowLayers !== undefined || perLayerFields.some((key) => Array.isArray(config[key]));
  // Homogeneous stacks are O(1); reject resource-exhausting heterogeneous descriptions.
  if (varying && layers > 100_000) throw new RangeError("Per-layer cache descriptions are limited to 100000 layers.");
  const iterations = varying ? layers : 1;
  const copies = varying ? 1 : layers;
  for (let index = 0; index < iterations; index++) {
    const kind = kindAt(index);
    if (kind === "none") continue;
    if (kind === "recurrent") {
      recurrentLayers += copies;
      const inner = at("intermediate_size", index) ?? (hasSsm && config.hidden_size != null ? product(at("hidden_size", index)!, at("expand", index) ?? 2) : undefined);
      if (hasLinear || (hasSsm && inner)) {
        convolutionDtype ??= computeDtype(config, note);
        recurrentDtype ??= options.recurrentStateDtype !== undefined ? dtypeName(options.recurrentStateDtype) : inferredDtype([config.mamba_ssm_dtype, config.state_dtype], "recurrent dtype", note) ?? "F32";
        if (options.recurrentStateDtype === undefined && config.mamba_ssm_dtype == null && config.state_dtype == null) note("Recurrent storage precision is absent; assuming F32. Backends can use a different dtype; override recurrentStateDtype.");
        let convElements: number;
        let recurrentElements: number;
        if (hasLinear) {
          const keyHeads = allocatedHeads(at("linear_num_key_heads", index)!);
          const valueHeads = allocatedHeads(at("linear_num_value_heads", index)!);
          const keyDim = at("linear_key_head_dim", index)!;
          const valueDim = at("linear_value_head_dim", index)!;
          convElements = product(sum(product(2, keyHeads, keyDim), product(valueHeads, valueDim)), at("linear_conv_kernel_dim", index)!);
          recurrentElements = product(valueHeads, keyDim, valueDim);
          assumptions.add("Linear-attention dimensions imply a full-kernel convolution history plus per-value-head key-by-value recurrent matrix; no token-linear K/V buffer for those layers.");
        } else {
          const stateSize = at("state_size", index)!;
          const groups = at("n_groups", index, true) ?? 0;
          const kernel = at("conv_kernel", index) ?? 4;
          if (config.conv_kernel == null) note("SSM convolution width is absent; assuming 4 history slots.");
          if (config.intermediate_size == null && config.expand == null) note("SSM inner width is absent; assuming twice hidden_size.");
          convElements = product(sum(inner!, product(2, groups, stateSize)), kernel);
          recurrentElements = product(inner!, stateSize);
          note("SSM dimensions use inner-width × state-size recurrent storage and full-kernel convolution history; grouping/history layout can vary by backend.");
        }
        convolutionBytes = sum(convolutionBytes, product(copies, batchSize, convElements, SAFETENSORS_DTYPE_BYTES[convolutionDtype]!));
        recurrentBytes = sum(recurrentBytes, product(copies, batchSize, recurrentElements, SAFETENSORS_DTYPE_BYTES[recurrentDtype]!));
        continue;
      }
      note("Recurrent/linear layers lack complete state dimensions; representing them with a full-context attention-equivalent proxy, not silently assigning zero bytes. This can substantially overestimate recurrent storage.");
    } else if (kind === "full") fullAttentionLayers += copies;
    else slidingAttentionLayers += copies;
    const { elements, mla } = attentionElements(index, kind === "full", kind === "recurrent");
    if (mla) mlaLayers += copies;
    if (kind === "sliding" && !window) note("Sliding layers have no usable window length; allocating the full requested context.");
    const tokens = kind === "sliding" && slidingWindowPolicy === "optimized" ? Math.min(window ?? maxModelLen, maxModelLen) : maxModelLen;
    attentionBytes = sum(attentionBytes, product(copies, batchSize, elements, width, tokens));
    if (crossAttention) attentionBytes = sum(attentionBytes, product(copies, batchSize, elements, width, maxModelLen));
    const indexDim = at("index_head_dim", index);
    if (indexDim) {
      const indexHeads = allocatedHeads(at("index_kv_heads", index) ?? 1);
      const ratio = at("index_compress_ratio", index) ?? 1;
      // A compressed-key cache retains one key per group plus its open raw group.
      const rawRows = config.index_compress_ratio == null ? 0 : ratio;
      const rows = sum(Math.ceil(maxModelLen / ratio), rawRows);
      const layerIndexerBytes = product(copies, batchSize, indexHeads, indexDim, width, rows);
      indexerBytes = sum(indexerBytes, layerIndexerBytes);
      attentionBytes = sum(attentionBytes, layerIndexerBytes);
      note(`Indexer cache uses ${indexDim}-element keys, ${indexHeads} aggregate KV heads, one key per ${ratio} token(s) and ${rawRows} raw history rows. Query indexer heads do not multiply storage; speculative/MRoPE tails and backend page padding are excluded.`);
    }
  }
  if (slidingAttentionLayers) assumptions.add(`Sliding-window allocation: ${slidingWindowPolicy}; transient prefill and boundary buffers are excluded.`);
  if (recurrentLayers && (convolutionBytes || recurrentBytes)) assumptions.add(`Convolution storage ${convolutionDtype}; recurrent storage ${recurrentDtype}. Configured precision is an assumption, not a backend storage guarantee.`);
  if (recurrentLayers) note("Recurrent payload counts one state per sequence. Paged/prefix-caching engines can retain many checkpoints and pad heterogeneous cache groups; this payload is not an estimate of occupied vLLM pool blocks.");
  const stateBytes = sum(convolutionBytes, recurrentBytes);
  const layout: KvCacheEstimate["layout"] = recurrentLayers
    ? (fullAttentionLayers || slidingAttentionLayers ? "hybrid" : "recurrent")
    : mlaLayers ? (mlaLayout === "compressed" ? "mla-compressed" : "mla-expanded") : "attention";
  return { bytes: sum(attentionBytes, stateBytes), dtype, maxModelLen, batchSize, tensorParallelSize, attentionBytes, indexerBytes, stateBytes, convolutionBytes, recurrentBytes, convolutionDtype, recurrentDtype, layout, approximate, slidingWindowPolicy, fullAttentionLayers, slidingAttentionLayers, recurrentLayers, assumptions: [...assumptions] };
}
