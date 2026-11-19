import { bitLength, cdiv, count, int, isRecord, lcm, str, truthy } from "./py.js";
import type { Weights } from "./weights.js";

export type Config = Record<string, any>;
export type Layers = Record<string, number>;

export const KV_CACHE_DTYPES = ["auto", "bfloat16", "float16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"];
export const MAX_NUM_BATCHED_TOKENS = 8192;
export const BLOCK_SIZE = 16;
// DeepSeek-V4.1 in vLLM: packed FP8 (`fp8_ds_mla`) KV and indexer records per compressed token, fitted on vLLM 0.33
export const DSV4_RECORD = 584;
export const DSV4_INDEX_RECORD = 148;
// FP8 indexer key scale of the DeepSeek-V3.2 and GLM-5 sparse attention indexers
export const DSV4_INDEX_SCALE = 4;

const BYTES: Record<string, number> = { F64: 8, F32: 4, F16: 2, BF16: 2, F8_E4M3: 1, F8_E5M2: 1, FP8_DS_MLA: 1, I8: 1 };
const TORCH: Record<string, string> = {
  float32: "F32",
  float: "F32",
  float16: "F16",
  half: "F16",
  bfloat16: "BF16",
  float8_e4m3: "F8_E4M3",
  float8_e4m3fn: "F8_E4M3",
  float8_e5m2: "F8_E5M2",
};
const ALIASES: Record<string, string[]> = {
  num_hidden_layers: ["num_hidden_layers", "n_layers", "num_layers", "n_layer"],
  num_attention_heads: ["num_attention_heads", "n_heads", "n_head", "num_heads"],
  num_key_value_heads: ["num_key_value_heads", "n_kv_heads", "num_kv_heads", "multi_query_group_num"],
  hidden_size: ["hidden_size", "d_model", "n_embd", "dim"],
  max_position_embeddings: ["max_position_embeddings", "model_max_length", "n_positions", "max_seq_len", "seq_length"],
  sliding_window: ["sliding_window", "sliding_window_size", "attention_window_size"],
};
const DEFAULTS: Record<string, Config> = {
  gemma2: {
    head_dim: 256,
    sliding_window: 4096,
    sliding_window_pattern: 2,
    max_position_embeddings: 8192,
    vocab_size: 256000,
  },
  gemma3_text: {
    head_dim: 256,
    sliding_window: 4096,
    sliding_window_pattern: 6,
    max_position_embeddings: 131072,
    num_attention_heads: 8,
    num_key_value_heads: 4,
    vocab_size: 262208,
  },
  qwen2: {
    hidden_size: 4096,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    num_key_value_heads: 32,
    max_position_embeddings: 32768,
    vocab_size: 151936,
  },
};
const TEXT_KEYS = ["text_config", "language_config", "llm_config", "lfm", "thinker_config", "talker_config"];
const NON_TEXT = ["vision", "audio", "encoder", "codec", "depth", "speech", "image", "video", "visual", "detr"];
const ENCODER_ARCHS = [
  "ForMaskedLM",
  "ForSequenceClassification",
  "ForTokenClassification",
  "ForQuestionAnswering",
  "ForImageClassification",
  "ForMaskGeneration",
  "ForTextEncoding",
  "Embedding",
];
const GENERATIVE_ARCHS = ["ForCausalLM", "ForConditionalGeneration", "LMHeadModel"];
const TEXT_HEADS = [
  "ForMaskedLM",
  "ForSequenceClassification",
  "ForTokenClassification",
  "ForQuestionAnswering",
  "ForMultipleChoice",
];
const STATE = ["linear", "conv", "mamba", "ssm", "recurrent", "delta", "rwkv", "lightning"];
const NO_CACHE = ["mlp", "moe", "dummy"];
const SLIDING = ["sliding", "local", "chunked", "swa"];

/** Python's `isinstance(value, int)`, where booleans are integers too. */
function pyInt(value: unknown): value is number | boolean {
  return typeof value === "boolean" || (typeof value === "number" && Number.isInteger(value));
}

const endsWithAny = (text: string, suffixes: string[]) => suffixes.some((s) => text.endsWith(s));

function range(n: number): number[] {
  return Array.from({ length: Math.max(0, n) }, (_, i) => i);
}

export function normalize(input: Config): Config {
  let config: Config = Object.fromEntries(Object.entries(input).filter(([, v]) => v !== null && v !== undefined));
  const defaults = DEFAULTS[str(config.model_type)] ?? {};
  for (const [key, aliases] of Object.entries(ALIASES)) {
    if (config[key] === null || config[key] === undefined) {
      const alias = aliases.find((a) => pyInt(config[a]));
      config[key] = alias !== undefined ? config[alias] : (defaults[key] ?? null);
    }
  }
  config = { ...defaults, ...config };
  const linear = config.linear_attn_config;
  if (isRecord(linear) && (config.layer_types === null || config.layer_types === undefined)) {
    const full = new Set<unknown>(linear.full_attn_layers || []);
    config.layer_types = range(config.num_hidden_layers || 0).map((i) =>
      full.has(i + 1) ? "full_attention" : "linear_attention",
    );
    const heads = linear.num_heads ?? null;
    const headDim = linear.head_dim ?? null;
    const entries: [string, unknown][] = [
      ["linear_num_value_heads", heads],
      ["linear_num_key_heads", heads],
      ["linear_key_head_dim", headDim],
      ["linear_value_head_dim", headDim],
      ["linear_conv_kernel_dim", linear.short_conv_kernel_size ?? null],
      ["mamba_ssm_dtype", "float32"],
    ];
    for (const [key, value] of entries) if (!(key in config)) config[key] = value;
  }
  const dsa = config.dsa_config;
  if (isRecord(dsa) && (config.index_head_dim === null || config.index_head_dim === undefined)) {
    const indexed = new Set<unknown>(dsa.dsa_layers || []);
    config.index_head_dim = dsa.index_head_dim ?? null;
    config.indexer_types = range(config.num_hidden_layers || 0).map((i) => (indexed.has(i + 1) ? "full" : "none"));
  }
  if (config.num_hidden_layers === null || config.num_hidden_layers === undefined) {
    const layers = config.layer_types || config.layers_block_type || config.hybrid_override_pattern;
    config.num_hidden_layers = truthy(layers) ? layers.length : null;
  }
  return config;
}

function isText(config: unknown): config is Config {
  if (!isRecord(config)) return false;
  const normalized = normalize(config);
  return pyInt(normalized.num_hidden_layers) && pyInt(normalized.num_attention_heads);
}

function nested(config: Config, path = ""): [string, Config][] {
  const found: [string, Config][] = [];
  for (const [key, value] of Object.entries(config)) {
    if (isRecord(value)) {
      const child = `${path}.${key}`;
      if (isText(value)) found.push([child, value]);
      found.push(...nested(value, child));
    }
  }
  return found;
}

export function isGenerative(config: Config, files: Map<string, number>): boolean {
  const architectures: unknown[] = config.architectures || [];
  return (
    files.has("generation_config.json") || architectures.some((a) => endsWithAny(String(a), GENERATIVE_ARCHS))
  );
}

export function isTextEncoder(config: Config, files: Map<string, number>): boolean {
  const modelType = str(config.model_type || "").toLowerCase();
  if (
    truthy(config.is_encoder_decoder) ||
    ["vision", "audio", "wav2vec", "dinov", "vit", "clip"].some((hint) => modelType.includes(hint)) ||
    files.has("preprocessor_config.json") ||
    ["vision_config", "audio_config", ...TEXT_KEYS].some((k) => isRecord(config[k])) ||
    !isText(config)
  ) {
    return false;
  }
  const architectures = ((config.architectures || []) as unknown[]).map((a) => str(a));
  return (
    files.has("config_sentence_transformers.json") ||
    architectures.some((a) => endsWithAny(a, TEXT_HEADS)) ||
    (!isGenerative(config, files) && architectures.some((a) => a.endsWith("Model")))
  );
}

export function textConfig(config: Config): Config | null {
  const architectures = (config.architectures || []) as unknown[];
  if (architectures.some((arch) => ENCODER_ARCHS.some((hint) => String(arch).includes(hint)))) return null;

  let found: Config | null = null;
  for (const key of TEXT_KEYS) {
    if (isRecord(config[key])) {
      found = textConfig(config[key]) ?? (isText(config[key]) ? config[key] : null);
      if (found !== null) break;
    }
  }
  if (found === null && isText(config)) found = config;
  if (found === null) {
    const candidates = nested(config)
      .filter(([p]) => !NON_TEXT.some((h) => p.toLowerCase().includes(h)))
      .map(([, c]) => c);
    let best = -Infinity;
    for (const candidate of candidates) {
      const size = normalize(candidate).hidden_size || 0;
      if (size > best) {
        best = size;
        found = candidate;
      }
    }
  }
  if (found === null) return null;

  found = normalize(found);
  if ((found.dtype === null || found.dtype === undefined) && (found.torch_dtype === null || found.torch_dtype === undefined)) {
    for (const key of ["dtype", "torch_dtype"]) {
      if (config[key] !== null && config[key] !== undefined) found[key] = config[key];
    }
  }
  if (
    (found.quantization_config === null || found.quantization_config === undefined) &&
    config.quantization_config !== null &&
    config.quantization_config !== undefined
  ) {
    found.quantization_config = config.quantization_config;
  }
  return found;
}

export function kvDtype(config: Config, requested: string, weights: Weights, warnings: string[]): string {
  if (requested === "fp8_e5m2" || requested === "fp8_e4m3") return requested.toUpperCase().replace("FP8", "F8");
  if (requested === "fp8_ds_mla" || (requested === "auto" && isRecord(config.dsa_config))) {
    // vLLM's sparse MLA cache outside Blackwell: FP8 latent with a 4 B scale per 128 values and a BF16 RoPE key
    return truthy(config.kv_lora_rank) ? "FP8_DS_MLA" : "F8_E4M3";
  }
  if (requested === "fp8" || requested === "fp8_inc") return "F8_E4M3";
  if (requested === "bfloat16" || requested === "float16") return TORCH[requested]!;
  if (requested !== "auto") {
    throw new Error(
      `\`--kv-cache-dtype=${requested}\` is not a vLLM KV cache dtype, use one of ${KV_CACHE_DTYPES.join(", ")}.`,
    );
  }

  if (truthy(config.compress_ratios)) return "F8_E4M3";
  const quantization: Config = config.quantization_config || {};
  const scheme: Config = quantization.kv_cache_scheme || {};
  if (scheme.num_bits === 8 || quantization.kv_cache_quant_algo === "FP8") return "F8_E4M3";
  const dtype = config.dtype || config.torch_dtype;
  if (typeof dtype === "string") {
    const name = dtype.startsWith("torch.") ? dtype.slice(6) : dtype;
    return downcastF32(TORCH[name] ?? "F16");
  }

  let largest: string | null = null;
  for (const [d, w] of weights.dtypes) {
    if (["BF16", "F16", "F32"].includes(d) && (largest === null || w.params > weights.dtypes.get(largest)!.params)) {
      largest = d;
    }
  }
  const fallback = downcastF32(largest ?? "BF16");
  warnings.push(
    `The KV cache dtype could not be inferred from \`config.json\`, so it defaults to ${fallback}, set ` +
      "`--kv-cache-dtype` to override it.",
  );
  return fallback;
}

function downcastF32(dtype: string): string {
  return dtype === "F32" ? "F16" : dtype;
}

export function layerKinds(config: Config): string[] {
  const layers: number = config.num_hidden_layers;
  const types = config.layer_types || config.layers_block_type;
  let kinds: string[];
  const local = config.local_layer_ids;
  const pattern = config.sliding_window_pattern;
  const hybrid = config.hybrid_override_pattern;
  if (Array.isArray(types) && types.length) {
    kinds = types.map((type) => {
      const layer = str(type).toLowerCase();
      if (SLIDING.some((h) => layer.includes(h))) return "sliding";
      if (STATE.some((h) => layer.includes(h))) return "state";
      if (NO_CACHE.some((h) => layer.includes(h))) return "none";
      return "full";
    });
  } else if (Array.isArray(local)) {
    kinds = range(layers).map((i) => (local.includes(i) ? "sliding" : "full"));
  } else if (pyInt(pattern) && Number(pattern) > 0) {
    kinds = range(layers).map((i) => ((i + 1) % Number(pattern) === 0 ? "full" : "sliding"));
  } else if (typeof hybrid === "string") {
    kinds = [...hybrid].map((c) => (c === "*" ? "full" : c === "M" ? "state" : "none"));
  } else {
    kinds = layerConfigs(config).map((layer) =>
      truthy(layer.sliding_window || layer.attention_chunk_size) && layer.use_sliding_window !== false
        ? "sliding"
        : "full",
    );
  }
  let modes: unknown[] | null = config.kv_sharing_modes ?? null;
  const sources = config.kv_source_layers;
  if (modes === null && Array.isArray(sources)) {
    modes = sources.map((source, i) => (i === source ? null : "reuse"));
  }
  const shared: number = config.num_kv_shared_layers || 0;
  const list: unknown[] = truthy(modes) ? modes! : [];
  return kinds.map((kind, i) =>
    (i < list.length && (list[i] === "reuse" || list[i] === "reuse_but_indexer")) || (shared && i >= layers - shared)
      ? "none"
      : kind,
  );
}

function layerConfigs(config: Config): Config[] {
  // Transformers serializes a sparse mapping with integer or zero-padded string indices.
  const raw: Config = config.per_layer_config || {};
  const overrides = new Map<number, Config>();
  for (const [i, value] of Object.entries(raw)) overrides.set(Number.parseInt(i, 10), value);
  const layers = range(config.num_hidden_layers).map((i) => normalize({ ...config, ...(overrides.get(i) ?? {}) }));
  const rope = config.no_rope_layers;
  if (truthy(config.kv_lora_rank) && Array.isArray(rope)) {
    for (let i = 0; i < Math.min(layers.length, rope.length); i++) {
      if (!truthy(rope[i])) layers[i]!.qk_rope_head_dim = 0;
    }
  }
  return layers;
}

function state(config: Config, dtypeBytes: number): [number, number] {
  const hidden: number = config.hidden_size || 0;
  const ssmBytes = config.mamba_ssm_dtype === "float32" || config.model_type === "nemotron_h" ? 4 : dtypeBytes;
  const heads = config.linear_num_value_heads;
  if (truthy(heads)) {
    const keyHeads = config.linear_num_key_heads || heads;
    const keyDim = config.linear_key_head_dim || 0;
    const valueDim = config.linear_value_head_dim || 0;
    const conv = (2 * keyHeads * keyDim + heads * valueDim) * ((config.linear_conv_kernel_dim || 4) - 1);
    return [int(conv * dtypeBytes + heads * keyDim * valueDim * ssmBytes), truthy(config.linear_attn_config) ? 3 : 2];
  }
  const size = config.mamba_d_state || config.ssm_state_size || config.state_size;
  if (truthy(size)) {
    const mambaHeads = config.mamba_num_heads || config.mamba_n_heads || 0;
    const headDim = config.mamba_head_dim || config.mamba_d_head || 0;
    const inner = mambaHeads * headDim || (config.mamba_expand || config.expand || 2) * hidden;
    const groups = config.n_groups || config.mamba_n_groups || 1;
    const kernel = config.conv_kernel || config.mamba_d_conv || config.d_conv || 4;
    return [int((inner + 2 * groups * size) * (kernel - 1) * dtypeBytes + inner * size * ssmBytes), 3];
  }
  const cache = config.conv_L_cache;
  if (truthy(cache)) return [int(hidden * (cache - 1) * dtypeBytes), 2];
  return [0, 2];
}

export function kvHeads(input: Config): number {
  const config = normalize(input);
  if (truthy(config.kv_lora_rank)) return 1;
  if (truthy(config.per_layer_config)) {
    return Math.max(...layerConfigs(config).map((layer) => kvHeads({ ...layer, per_layer_config: null })));
  }
  return config.num_global_key_value_heads || config.num_key_value_heads || config.num_attention_heads;
}

function compressed(config: Config, maxModelLen: number, batchedTokens: number): [number, Layers] {
  const ratios: number[] = config.compress_ratios;
  const sources: number[] = config.kv_source_layer_ids;
  const window = config.sliding_window || 128;
  const sliding = Math.min(window - 1 + 2 * batchedTokens, maxModelLen);
  let sum = 0;
  for (const i of sources) if (truthy(ratios[i])) sum += 1 / ratios[i]!;
  const perToken = (DSV4_RECORD + DSV4_INDEX_RECORD) * sum;
  const size = ratios.length * cdiv(sliding, 32) * 32 * DSV4_RECORD + perToken * maxModelLen;
  return [int(size), { compressed: sources.length, sliding: ratios.length, full: 0, state: 0, none: 0 }];
}

function packedMla(
  config: Config,
  dtype: string,
  length: number,
  batchSize: number,
  tensorParallelSize: number,
): [number, Layers] | null {
  if (!(truthy(config.dsa_config) && truthy(config.linear_attn_config) && truthy(config.kv_lora_rank))) return null;
  const kinds = layerKinds(config);
  const full = count(kinds, "full");
  const stateLayers = count(kinds, "state");
  if (
    !full ||
    !stateLayers ||
    count(kinds, "sliding") ||
    count((config.indexer_types || []) as unknown[], "full") !== full
  ) {
    return null;
  }
  // vLLM's Kimi K3.1 cache: a block row holds every MLA and indexer page. State groups share the same
  // pool, so every checkpoint reserves an entire row, including its padding. MLA is replicated across
  // tensor-parallel ranks; KDA states shard, changing the block size on each rank.
  const rank: number = config.kv_lora_rank;
  const rope: number = config.qk_rope_head_dim || 0;
  const token = dtype === "FP8_DS_MLA" ? rank + cdiv(rank, 128) * 4 + 2 * rope : (rank + rope) * (BYTES[dtype] ?? 2);
  const rowToken = full * (token + config.index_head_dim + DSV4_INDEX_SCALE);
  const [rawState, pages] = state(config, 2);
  const stateBytes = cdiv(rawState, tensorParallelSize);
  const groups = cdiv(stateLayers, full);
  const statesPerRow = cdiv(stateLayers, groups);
  // Sparse MLA and indexer kernels use 64-token blocks; vLLM rounds the manager block to a power of two.
  const units = Math.max(1, cdiv(statesPerRow * stateBytes, 64 * rowToken));
  const block = 64 * 2 ** bitLength(units - 1);
  let row = block * rowToken;
  if (block > 64) {
    // Both cache views must split into whole 64-token kernel pages within the shared row.
    const alignment = lcm(64 * token, 64 * (config.index_head_dim + DSV4_INDEX_SCALE));
    row = cdiv(row, alignment) * alignment;
  }
  const size = row * (cdiv(length, block) + groups * pages);
  const layers = { full, sliding: 0, state: stateLayers, none: kinds.length - full - stateLayers };
  return [size * tensorParallelSize * batchSize, layers];
}

const PER_LAYER_DROPPED = [
  "per_layer_config",
  "num_kv_shared_layers",
  "kv_sharing_modes",
  "kv_source_layers",
  "no_rope_layers",
  "global_head_dim",
  "num_global_key_value_heads",
];

export function kvCache(
  input: Config,
  dtype: string,
  maxModelLen: number,
  batchSize: number,
  batchedTokens = MAX_NUM_BATCHED_TOKENS,
  tensorParallelSize = 1,
): [number, Layers] {
  const config = normalize(input);
  const packed = packedMla(config, dtype, maxModelLen, batchSize, tensorParallelSize);
  if (packed) return packed;
  if (
    (truthy(config.per_layer_config) || (truthy(config.kv_lora_rank) && truthy(config.no_rope_layers))) &&
    !layerKinds(config).includes("state")
  ) {
    const kinds = layerKinds(config);
    let size = 0;
    const counts: Layers = { full: 0, sliding: 0, state: 0, none: 0 };
    const layers = layerConfigs(config);
    for (let i = 0; i < Math.min(layers.length, kinds.length); i++) {
      const layer = layers[i]!;
      const kind = kinds[i]!;
      counts[kind] = (counts[kind] ?? 0) + 1;
      // Explicit per-layer dimensions supersede the legacy global-attention overrides.
      for (const key of PER_LAYER_DROPPED) delete layer[key];
      layer.num_hidden_layers = 1;
      layer.layer_types = [kind === "none" ? "dummy" : `${kind}_attention`];
      if (Array.isArray(config.indexer_types)) layer.indexer_types = config.indexer_types.slice(i, i + 1);
      size += kvCache(layer, dtype, maxModelLen, batchSize, batchedTokens, tensorParallelSize)[0];
    }
    return [size, counts];
  }
  if (tensorParallelSize > 1) {
    if (truthy(config.linear_attn_config)) {
      const sharded = {
        ...config,
        linear_num_value_heads: cdiv(config.linear_num_value_heads, tensorParallelSize),
        linear_num_key_heads: cdiv(config.linear_num_key_heads, tensorParallelSize),
      };
      const [size, layers] = kvCache(sharded, dtype, maxModelLen, batchSize, batchedTokens);
      return [size * tensorParallelSize, layers];
    }
    const [size, layers] = kvCache(config, dtype, maxModelLen, batchSize, batchedTokens);
    return [int(size * Math.max(1.0, tensorParallelSize / kvHeads(config))), layers];
  }
  if (truthy(config.compress_ratios) && config.kv_source_layer_ids !== null && config.kv_source_layer_ids !== undefined) {
    const [size, counts] = compressed(config, maxModelLen, batchedTokens);
    return [size * batchSize, counts];
  }
  const heads: number = config.num_attention_heads;
  const cacheHeads: number = config.num_key_value_heads || heads;
  const headDim: number = config.head_dim || Math.floor((config.hidden_size || 0) / heads);
  let width: number;
  let fullWidth: number;
  const rank = config.kv_lora_rank;
  if (truthy(rank)) {
    const rope = config.qk_rope_head_dim || 0;
    width = fullWidth = dtype === "FP8_DS_MLA" ? rank + cdiv(rank, 128) * 4 + 2 * rope : rank + rope;
  } else {
    width = 2 * (config.swa_num_key_value_heads || cacheHeads) * (config.swa_head_dim || headDim);
    fullWidth = 2 * (config.num_global_key_value_heads || cacheHeads) * (config.global_head_dim || headDim);
  }

  let indexer = 0;
  const index = config.index_head_dim;
  if (truthy(index)) {
    const types = config.indexer_types;
    const indexed = Array.isArray(types) ? count(types as unknown[], "full") : layerKinds(config).length;
    indexer = indexed * cdiv(maxModelLen, BLOCK_SIZE) * BLOCK_SIZE * (index + DSV4_INDEX_SCALE) * batchSize;
  }
  const kinds = layerKinds(config);
  const full = count(kinds, "full");
  const sliding = count(kinds, "sliding");
  const stateLayers = count(kinds, "state");
  const layers = { full, sliding, state: stateLayers, none: kinds.length - full - sliding - stateLayers };
  const window = config.sliding_window || config.attention_chunk_size || maxModelLen;
  const slidingTokens = Math.min(window - 1 + 2 * batchedTokens, maxModelLen);
  const slidingPages = cdiv(slidingTokens, BLOCK_SIZE) + Number(slidingTokens < maxModelLen);
  const dtypeBytes = BYTES[dtype] ?? 2;
  const token = int(width * dtypeBytes);
  const fullToken = int(fullWidth * dtypeBytes);
  if (!stateLayers) {
    const size = full * cdiv(maxModelLen, BLOCK_SIZE) * fullToken + sliding * slidingPages * token;
    return [size * BLOCK_SIZE * batchSize + indexer, layers];
  }

  const [stateBytes, statePages] = state(config, truthy(config.linear_attn_config) ? 2 : dtypeBytes);
  if (!(full + sliding)) return [stateLayers * statePages * stateBytes * batchSize + indexer, layers];
  const alignment = truthy(config.linear_attn_config) ? 128 : BLOCK_SIZE;
  const block = Math.max(alignment, cdiv(cdiv(stateBytes, token), alignment) * alignment);
  const group = Math.min(...[full, sliding, stateLayers].filter((n) => n));
  let size = cdiv(full, group) * cdiv(maxModelLen, block) * fullToken;
  size += cdiv(sliding, group) * (cdiv(slidingTokens, block) + Number(slidingTokens < maxModelLen)) * token;
  size += cdiv(stateLayers, group) * statePages * token;
  return [group * size * block * batchSize + indexer, layers];
}
