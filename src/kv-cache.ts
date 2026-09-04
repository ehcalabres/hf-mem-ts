import { SAFETENSORS_DTYPE_BYTES } from "./safetensors.js";
import type { KvCacheEstimate, KvCacheOptions } from "./types.js";

type JsonConfig = Record<string, unknown>;

function positive(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) throw new RangeError(`${name} must be a positive safe integer.`);
  return value as number;
}

function integer(config: JsonConfig, key: string): number | undefined {
  return config[key] === undefined ? undefined : positive(config[key], key);
}

function required(config: JsonConfig, key: string): number {
  if (config[key] === undefined) throw new Error(`Cache configuration lacks ${key}.`);
  return positive(config[key], key);
}

function product(...values: number[]): number {
  let result = 1;
  for (const value of values) {
    result *= value;
    if (!Number.isSafeInteger(result) || result < 0) throw new RangeError("Cache estimate exceeds JavaScript's safe integer range.");
  }
  return result;
}

function sum(...values: number[]): number {
  let result = 0;
  for (const value of values) {
    result += value;
    if (!Number.isSafeInteger(result) || result < 0) throw new RangeError("Cache estimate exceeds JavaScript's safe integer range.");
  }
  return result;
}

function dtypeName(value: unknown): string {
  if (typeof value !== "string") throw new Error("Cache dtype must be a string.");
  const normalized = value.replace(/^torch\./, "").toLowerCase();
  const aliases: Record<string, string> = {
    bfloat16: "BF16", bf16: "BF16", float16: "F16", fp16: "F16", f16: "F16",
    float32: "F32", fp32: "F32", f32: "F32", fp8: "F8_E4M3", fp8_ds_mla: "F8_E4M3",
    fp8_inc: "F8_E4M3", fp8_e4m3: "F8_E4M3", fp8_e5m2: "F8_E5M2",
    float8_e4m3: "F8_E4M3", float8_e4m3fn: "F8_E4M3", float8_e5m2: "F8_E5M2",
  };
  const dtype = aliases[normalized] ?? value.toUpperCase();
  if (!["BF16", "F16", "F32", "F8_E4M3", "F8_E5M2"].includes(dtype)) {
    throw new Error(`Unsupported Safetensors KV-cache dtype: ${value}; specify a floating-point cache precision.`);
  }
  return dtype;
}

function consistentDtype(values: unknown[], description: string): string | undefined {
  let dtype: string | undefined;
  for (const value of values) {
    if (value === undefined || value === null || value === "auto") continue;
    const next = dtypeName(value);
    if (dtype !== undefined && dtype !== next) throw new Error(`Conflicting ${description}; pass kvCacheDtype explicitly for the attention cache.`);
    dtype = next;
  }
  return dtype;
}

function computeDtype(config: JsonConfig): string {
  const dtype = consistentDtype([config.dtype, config.torch_dtype], "configured compute dtypes");
  if (!dtype) throw new Error("Could not infer the compute/cache dtype; pass kvCacheDtype explicitly.");
  return dtype;
}

export function resolveSafetensorsKvDtype(config: JsonConfig, requested = "auto"): string {
  if (requested.toLowerCase() !== "auto") return dtypeName(requested);
  const quantization = config.quantization_config as JsonConfig | undefined;
  const scheme = quantization?.kv_cache_scheme as JsonConfig | undefined;
  let schemeDtype: string | undefined;
  if (scheme != null) {
    if (scheme.num_bits !== 8 || scheme.type !== "float") {
      throw new Error("Unsupported explicit kv_cache_scheme; pass kvCacheDtype for a supported backend cache precision.");
    }
    schemeDtype = "F8_E4M3";
  }
  // Weight quant_method, format and tensor dtypes do not specify cache precision.
  return consistentDtype([config.kv_cache_dtype, quantization?.kv_cache_dtype, schemeDtype], "explicit cache dtypes") ?? computeDtype(config);
}

function unsupported(detail: string): never {
  throw new Error(`Unsupported cache layout: ${detail}. Disable kvCache or provide a supported architecture configuration; changing dtype cannot repair a layout.`);
}

export function estimateSafetensorsKvCache(rawConfig: JsonConfig, options: KvCacheOptions = {}): KvCacheEstimate {
  const text = rawConfig.text_config;
  const config = text && typeof text === "object" && !Array.isArray(text) ? { ...rawConfig, ...text as JsonConfig } : rawConfig;
  const layers = required(config, "num_hidden_layers");
  const maxModelLen = positive(options.maxModelLen ?? config.max_position_embeddings ?? config.n_positions ?? config.max_seq_len, "maxModelLen");
  const batchSize = positive(options.batchSize ?? 1, "batchSize");
  const slidingWindowPolicy = options.slidingWindowPolicy ?? "optimized";
  if (!["optimized", "full-context"].includes(slidingWindowPolicy)) throw new Error("slidingWindowPolicy must be optimized or full-context.");
  const mlaLayout = options.mlaLayout ?? "compressed";
  if (!["compressed", "expanded"].includes(mlaLayout)) throw new Error("mlaLayout must be compressed or expanded.");
  const modelType = String(config.model_type ?? "");
  const qwen35 = ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"].includes(modelType);
  const mla = config.kv_lora_rank !== undefined;
  if (config.is_encoder_decoder === true || config.cross_attention_hidden_size !== undefined) unsupported("encoder-decoder/cross-attention state");
  if ((config.num_kv_shared_layers as number) > 0 || (config.num_kv_shared_layers_pattern as number) > 0) unsupported("cross-layer cache sharing");
  if (!qwen35 && (/mamba|rwkv|recurrent|jamba|zamba|nemotron_h|falcon_h1/.test(modelType) || config.ssm_cfg !== undefined || config.linear_num_value_heads !== undefined)) unsupported(`${modelType || "model-specific"} recurrent state`);
  if (!mla && (config.qk_rope_head_dim !== undefined || config.qk_nope_head_dim !== undefined)) unsupported("incomplete MLA configuration (missing kv_lora_rank)");
  if (config.head_dim_global !== undefined || config.num_key_value_heads_global !== undefined || config.attention_k_eq_v === true) unsupported("model-specific global head dimensions or shared key/value storage");
  if (!qwen35 && !mla && modelType && !["llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe"].includes(modelType)) unsupported(`unrecognized architecture ${modelType}`);
  if (!mla && (config.v_head_dim !== undefined || config.key_head_dim !== undefined || config.value_head_dim !== undefined)) unsupported("nonstandard separate key/value head dimensions without a supported layout");
  if (Array.isArray(config.mlp_only_layers) && config.mlp_only_layers.length) unsupported("MLP-only layers without an explicit supported cache schedule");

  let fullAttentionLayers = 0;
  let slidingAttentionLayers = 0;
  let recurrentLayers = 0;
  if (config.layer_types !== undefined) {
    if (!Array.isArray(config.layer_types) || config.layer_types.length !== layers) throw new Error("layer_types must contain one entry per hidden layer.");
    for (const type of config.layer_types) {
      if (["attention", "full_attention", "global_attention"].includes(String(type))) fullAttentionLayers++;
      else if (type === "sliding_attention") slidingAttentionLayers++;
      else if (type === "linear_attention" && qwen35) recurrentLayers++;
      else unsupported(`layer type ${String(type)} in ${modelType || "unknown model"}`);
    }
  } else if (qwen35) {
    const interval = required(config, "full_attention_interval");
    fullAttentionLayers = Math.floor(layers / interval);
    recurrentLayers = layers - fullAttentionLayers;
  } else if (config.sliding_window_pattern !== undefined) {
    const pattern = required(config, "sliding_window_pattern");
    fullAttentionLayers = Math.floor(layers / pattern);
    slidingAttentionLayers = layers - fullAttentionLayers;
  } else if (config.sliding_window != null && config.use_sliding_window !== false) {
    if (config.max_window_layers !== undefined) unsupported("sliding window without explicit layer_types (max_window_layers semantics vary)");
    slidingAttentionLayers = layers;
  } else {
    fullAttentionLayers = layers;
  }
  if (mla && recurrentLayers) unsupported("mixed MLA and recurrent layers");
  const window = slidingAttentionLayers ? required(config, "sliding_window") : maxModelLen;
  const layerTokens = sum(product(fullAttentionLayers, maxModelLen), product(slidingAttentionLayers, slidingWindowPolicy === "full-context" ? maxModelLen : Math.min(window, maxModelLen)));
  const dtype = resolveSafetensorsKvDtype(config, options.dtype);
  const assumptions = ["Resident cache payload at the requested length; no allocator, workspace, paging, padding, offloading or prefix/cross-layer sharing adjustments."];
  let layout: KvCacheEstimate["layout"] = "attention";
  let elementsPerToken = 0;
  if (fullAttentionLayers || slidingAttentionLayers) {
    const heads = required(config, "num_attention_heads");
    const kvHeads = integer(config, "num_key_value_heads") ?? heads;
    if (heads % kvHeads !== 0) throw new RangeError("num_attention_heads must be divisible by num_key_value_heads.");
    if (mla) {
      const rank = required(config, "kv_lora_rank");
      const rope = required(config, "qk_rope_head_dim");
      layout = mlaLayout === "compressed" ? "mla-compressed" : "mla-expanded";
      elementsPerToken = mlaLayout === "compressed" ? sum(rank, rope) : product(heads, sum(required(config, "qk_nope_head_dim"), rope, required(config, "v_head_dim")));
      assumptions.push(mlaLayout === "compressed" ? "MLA stores one compressed KV latent plus one shared RoPE key per token, not per head; requires a compressed-cache backend." : "MLA stores expanded per-head keys (non-RoPE plus RoPE) and values; no compressed latent is retained.");
    } else {
      const explicitHeadDim = integer(config, "head_dim");
      const headDim = explicitHeadDim ?? positive(required(config, "hidden_size") / heads, "hidden_size / num_attention_heads");
      elementsPerToken = product(kvHeads, sum(headDim, headDim));
      assumptions.push("Attention stores separate keys and values for each KV head.");
    }
  }
  if (slidingAttentionLayers) assumptions.push(slidingWindowPolicy === "optimized" ? "Sliding layers allocate min(context, window) token slots, including in hybrid models; transient prefill and window-boundary buffers are excluded." : "All attention layers allocate full-context token slots, even when attention is windowed.");
  const attentionBytes = product(layerTokens, elementsPerToken, SAFETENSORS_DTYPE_BYTES[dtype]!, batchSize);
  let convolutionBytes = 0;
  let recurrentBytes = 0;
  let convolutionDtype: string | null = null;
  let recurrentDtype: string | null = null;
  if (recurrentLayers) {
    layout = "qwen3.5-hybrid";
    const keyHeads = required(config, "linear_num_key_heads");
    const valueHeads = required(config, "linear_num_value_heads");
    if (valueHeads % keyHeads !== 0) throw new RangeError("linear_num_value_heads must be divisible by linear_num_key_heads.");
    const keyDim = required(config, "linear_key_head_dim");
    const valueDim = required(config, "linear_value_head_dim");
    const kernel = required(config, "linear_conv_kernel_dim");
    convolutionDtype = computeDtype(config);
    recurrentDtype = dtypeName(options.recurrentStateDtype ?? config.mamba_ssm_dtype ?? "F32");
    if (!["F16", "BF16", "F32"].includes(convolutionDtype) || !["F16", "BF16", "F32"].includes(recurrentDtype)) unsupported("Qwen3.5 state precision must be F16, BF16 or F32");
    convolutionBytes = product(recurrentLayers, batchSize, sum(product(2, keyHeads, keyDim), product(valueHeads, valueDim)), kernel, SAFETENSORS_DTYPE_BYTES[convolutionDtype]!);
    recurrentBytes = product(recurrentLayers, batchSize, valueHeads, keyDim, valueDim, SAFETENSORS_DTYPE_BYTES[recurrentDtype]!);
    assumptions.push(`Qwen3.5 Gated DeltaNet: convolution state [batch, 2*keyHeads*keyDim + valueHeads*valueDim, kernel] in ${convolutionDtype}; recurrent state [batch, valueHeads, keyDim, valueDim] in ${recurrentDtype}. Recurrent dtype uses the explicit override, then mamba_ssm_dtype, then an F32 policy; config does not guarantee backend storage precision. No per-token linear-attention KV cache.`);
  }
  const stateBytes = sum(convolutionBytes, recurrentBytes);
  return { bytes: sum(attentionBytes, stateBytes), dtype, maxModelLen, batchSize, attentionBytes, stateBytes, convolutionBytes, recurrentBytes, convolutionDtype, recurrentDtype, layout, slidingWindowPolicy, fullAttentionLayers, slidingAttentionLayers, recurrentLayers, assumptions };
}
