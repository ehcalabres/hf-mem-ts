import { SAFETENSORS_DTYPE_BYTES } from "./safetensors.js";
import type { KvCacheEstimate, WeightMetadata } from "./types.js";

type JsonConfig = Record<string, unknown>;

function integer(config: JsonConfig, key: string): number | undefined {
  const value = config[key];
  return Number.isSafeInteger(value) && (value as number) > 0 ? value as number : undefined;
}

export function resolveSafetensorsKvDtype(config: JsonConfig, requested = "auto", metadata?: WeightMetadata): string {
  const normalized = requested.toLowerCase();
  if (normalized !== "auto") {
    const aliases: Record<string, string> = {
      bfloat16: "BF16", bf16: "BF16", float16: "F16", fp16: "F16", f16: "F16",
      float32: "F32", fp32: "F32", f32: "F32", fp8: "F8_E4M3", fp8_ds_mla: "F8_E4M3",
      fp8_inc: "F8_E4M3", fp8_e4m3: "F8_E4M3", fp8_e5m2: "F8_E5M2",
    };
    const dtype = aliases[normalized] ?? requested.toUpperCase();
    if (SAFETENSORS_DTYPE_BYTES[dtype] === undefined) throw new Error(`Unsupported Safetensors KV-cache dtype: ${requested}.`);
    return dtype;
  }

  const quantization = config.quantization_config as JsonConfig | undefined;
  const method = String(quantization?.quant_method ?? "").toLowerCase();
  if (method === "fp8" || method === "modelopt") {
    const format = String(quantization?.fmt ?? quantization?.format ?? "").toLowerCase();
    if (format.includes("e5m2")) return "F8_E5M2";
    if (format.includes("e4m3")) return "F8_E4M3";
    const scheme = quantization?.kv_cache_scheme as JsonConfig | undefined;
    if (scheme?.num_bits === 8 && scheme.type === "float") return "F8_E4M3";
    const used = Object.values(metadata?.components ?? {}).flatMap((component) => Object.keys(component.dtypes));
    const fp8 = ["F8_E4M3", "F8_E5M2"].sort((a, b) => used.filter((v) => v === b).length - used.filter((v) => v === a).length)[0];
    if (fp8 && used.includes(fp8)) return fp8;
    return "F8_E4M3";
  }
  const configured = String(config.torch_dtype ?? config.dtype ?? "").replace(/^torch\./, "").toLowerCase();
  const aliases: Record<string, string> = {
    bfloat16: "BF16", float16: "F16", float32: "F32", float8_e4m3: "F8_E4M3",
    float8_e4m3fn: "F8_E4M3", float8_e5m2: "F8_E5M2", int8: "I8",
  };
  if (aliases[configured]) return aliases[configured];
  throw new Error("Could not infer the KV-cache dtype; pass kvCacheDtype explicitly.");
}

export function estimateSafetensorsKvCache(
  rawConfig: JsonConfig,
  options: { maxModelLen?: number; batchSize?: number; dtype?: string; metadata?: WeightMetadata } = {},
): KvCacheEstimate {
  let config = rawConfig;
  if (rawConfig.text_config && typeof rawConfig.text_config === "object") {
    config = { ...rawConfig, ...(rawConfig.text_config as JsonConfig) };
  }
  const hiddenSize = integer(config, "hidden_size");
  const layers = integer(config, "num_hidden_layers");
  const heads = integer(config, "num_attention_heads");
  const maxModelLen = options.maxModelLen ?? integer(config, "max_position_embeddings") ?? integer(config, "n_positions") ?? integer(config, "max_seq_len");
  const batchSize = options.batchSize ?? 1;
  if (!hiddenSize || !layers || !heads || !maxModelLen) {
    throw new Error("config.json lacks hidden_size, num_hidden_layers, num_attention_heads, or a context length; pass maxModelLen if needed.");
  }
  if (!Number.isSafeInteger(batchSize) || batchSize <= 0) throw new RangeError("batchSize must be a positive safe integer.");
  const kvHeads = integer(config, "num_key_value_heads") ?? heads;
  const headDim = integer(config, "head_dim") ?? Math.floor(hiddenSize / heads);

  let fullLayers = layers;
  let slidingLayers = 0;
  const pattern = integer(config, "sliding_window_pattern");
  if (pattern) {
    fullLayers = Math.floor(layers / pattern);
    slidingLayers = layers - fullLayers;
  } else if (Array.isArray(config.layer_types)) {
    fullLayers = config.layer_types.filter((type) => ["attention", "full_attention", "global_attention"].includes(String(type))).length;
    slidingLayers = layers - fullLayers;
  }
  const slidingWindow = integer(config, "sliding_window") ?? maxModelLen;
  const hybrid = fullLayers > 0 && slidingLayers > 0;
  const totalLayerTokens = fullLayers * maxModelLen + slidingLayers * (hybrid ? maxModelLen : Math.min(slidingWindow, maxModelLen));
  const dtype = resolveSafetensorsKvDtype(config, options.dtype, options.metadata);
  const bytes = 2 * totalLayerTokens * kvHeads * headDim * SAFETENSORS_DTYPE_BYTES[dtype]! * batchSize;
  if (!Number.isSafeInteger(bytes)) throw new RangeError("KV-cache estimate exceeds JavaScript's safe integer range.");
  return { bytes, dtype, maxModelLen, batchSize };
}
