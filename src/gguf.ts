import { assertPositiveInteger, fetchRange } from "./http.js";
import { estimateSafetensorsKvCache } from "./kv-cache.js";
import type { ComponentStats, FetchLike, KvCacheEstimate, KvCacheOptions, WeightMetadata } from "./types.js";

// GGML block layouts: gguf-py/gguf/constants.py and ggml/src/ggml-common.h.
// Q8_1 uses the C layout (two fp16 scales); the Python table still lists fp32 scales.
const DTYPE_BLOCKS: Readonly<Record<string, readonly [blockElements: number, blockBytes: number]>> = {
  F32: [1, 4], F16: [1, 2], Q4_0: [32, 18], Q4_1: [32, 20], Q5_0: [32, 22], Q5_1: [32, 24],
  Q8_0: [32, 34], Q8_1: [32, 36], Q2_K: [256, 84], Q3_K: [256, 110], Q4_K: [256, 144],
  Q5_K: [256, 176], Q6_K: [256, 210], Q8_K: [256, 292], IQ2_XXS: [256, 66], IQ2_XS: [256, 74],
  IQ3_XXS: [256, 98], IQ1_S: [256, 50], IQ4_NL: [32, 18], IQ3_S: [256, 110], IQ2_S: [256, 82],
  IQ4_XS: [256, 136], I8: [1, 1], I16: [1, 2], I32: [1, 4], I64: [1, 8], F64: [1, 8],
  IQ1_M: [256, 56], BF16: [1, 2], TQ1_0: [256, 54], TQ2_0: [256, 66], MXFP4: [32, 17],
  NVFP4: [64, 36], Q1_0: [128, 18], Q2_0: [64, 18],
};

export const GGUF_DTYPE_BITS: Readonly<Record<string, number>> = Object.freeze(Object.assign(
  Object.create(null) as Record<string, number>,
  Object.fromEntries(Object.entries(DTYPE_BLOCKS).map(([name, [elements, bytes]]) => [name, bytes * 8 / elements])),
));

const DTYPE_NAMES: Readonly<Record<number, string>> = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
  10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
  17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
  24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
  34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
};

class TruncatedGgufError extends RangeError {}

const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

class Cursor {
  offset = 0;
  readonly view: DataView;
  constructor(readonly bytes: Uint8Array, readonly maxBytes: number) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  need(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.maxBytes - this.offset) {
      throw new RangeError("GGUF metadata exceeds the byte budget.");
    }
    if (length > this.bytes.byteLength - this.offset) throw new TruncatedGgufError("Truncated GGUF metadata.");
  }
  u8(): number { this.need(1); return this.view.getUint8(this.offset++); }
  i8(): number { this.need(1); return this.view.getInt8(this.offset++); }
  u16(): number { this.need(2); const v = this.view.getUint16(this.offset, true); this.offset += 2; return v; }
  i16(): number { this.need(2); const v = this.view.getInt16(this.offset, true); this.offset += 2; return v; }
  u32(): number { this.need(4); const v = this.view.getUint32(this.offset, true); this.offset += 4; return v; }
  i32(): number { this.need(4); const v = this.view.getInt32(this.offset, true); this.offset += 4; return v; }
  f32(): number { this.need(4); const v = this.view.getFloat32(this.offset, true); this.offset += 4; return v; }
  u64(): number { this.need(8); const v = this.view.getBigUint64(this.offset, true); this.offset += 8; return safe(v); }
  i64(): number { this.need(8); const v = this.view.getBigInt64(this.offset, true); this.offset += 8; return safe(v); }
  f64(): number { this.need(8); const v = this.view.getFloat64(this.offset, true); this.offset += 8; return v; }
  string(): string { const length = this.u64(); this.need(length); const value = UTF8_DECODER.decode(this.bytes.subarray(this.offset, this.offset + length)); this.offset += length; return value; }
}

function safe(value: bigint): number {
  if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) throw new RangeError("GGUF integer exceeds JavaScript's safe integer range.");
  return Number(value);
}

const VALUE_MIN_BYTES = [1, 1, 2, 2, 4, 4, 4, 1, 8, 12, 8, 8, 8] as const;

function readValue(cursor: Cursor, type: number, depth = 0): unknown {
  switch (type) {
    case 0: return cursor.u8(); case 1: return cursor.i8(); case 2: return cursor.u16(); case 3: return cursor.i16();
    case 4: return cursor.u32(); case 5: return cursor.i32(); case 6: return cursor.f32();
    case 7: { const value = cursor.u8(); if (value > 1) throw new Error("Invalid GGUF boolean."); return Boolean(value); }
    case 8: return cursor.string();
    case 9: {
      if (depth >= 16) throw new RangeError("GGUF metadata arrays exceed the nesting limit.");
      const childType = cursor.u32();
      const minimum = VALUE_MIN_BYTES[childType];
      if (minimum === undefined) throw new Error(`Unsupported GGUF metadata value type: ${childType}.`);
      const length = cursor.u64();
      if (length > 1_000_000) throw new RangeError("GGUF metadata array exceeds the element limit.");
      cursor.need(length * minimum);
      const values = new Array<unknown>(length);
      for (let i = 0; i < length; i++) values[i] = readValue(cursor, childType, depth + 1);
      return values;
    }
    case 10: return cursor.u64(); case 11: return cursor.i64(); case 12: return cursor.f64();
    default: throw new Error(`Unsupported GGUF metadata value type: ${type}.`);
  }
}

function safeCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError("GGUF count or byte total exceeds JavaScript's safe integer range.");
  return value;
}

export interface ParsedGguf extends WeightMetadata {
  metadata: Record<string, unknown>;
}

export function parseGguf(bytes: Uint8Array): ParsedGguf {
  return parseMetadata(bytes, Math.max(bytes.byteLength, 100_000_000));
}

function parseMetadata(bytes: Uint8Array, maxBytes: number): ParsedGguf {
  const cursor = new Cursor(bytes, maxBytes);
  if (cursor.u32() !== 0x46554747) throw new Error("Not a GGUF file (magic number mismatch).");
  const version = cursor.u32();
  if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version: ${version}.`);
  const tensorCount = cursor.u64();
  const metadataCount = cursor.u64();
  cursor.need(safeCount(tensorCount * 32 + metadataCount * 13));
  const metadata: Record<string, unknown> = Object.create(null);
  for (let i = 0; i < metadataCount; i++) {
    const key = cursor.string();
    if (Object.hasOwn(metadata, key)) throw new Error(`Duplicate GGUF metadata key: ${key}.`);
    metadata[key] = readValue(cursor, cursor.u32());
  }
  const component: ComponentStats = { parameters: 0, bytes: 0, dtypes: Object.create(null) };
  const names = new Set<string>();
  for (let i = 0; i < tensorCount; i++) {
    const name = cursor.string();
    if (names.has(name)) throw new Error(`Duplicate GGUF tensor name: ${name}.`);
    names.add(name);
    const dimensions = cursor.u32();
    if (dimensions < 1 || dimensions > 4) throw new Error(`Invalid dimensions for GGUF tensor ${name}.`);
    let count = 1;
    let rowElements = 0;
    for (let dimension = 0; dimension < dimensions; dimension++) {
      const size = cursor.u64();
      if (size === 0) throw new Error(`Invalid zero dimension for GGUF tensor ${name}.`);
      if (dimension === 0) rowElements = size;
      count = safeCount(count * size);
    }
    const type = cursor.u32();
    const offset = cursor.u64();
    const dtype = Object.hasOwn(DTYPE_NAMES, type) ? DTYPE_NAMES[type] : undefined;
    if (!dtype || !Object.hasOwn(DTYPE_BLOCKS, dtype)) throw new Error(`Unsupported GGUF tensor type: ${type}.`);
    const [blockElements, blockBytes] = DTYPE_BLOCKS[dtype]!;
    if (rowElements % blockElements !== 0) throw new Error(`GGUF tensor ${name} row is not aligned to its ${dtype} block size.`);
    const tensorBytes = safeCount(count / blockElements * blockBytes);
    safeCount(offset + tensorBytes);
    const stats = component.dtypes[dtype] ?? { parameters: 0, bytes: 0 };
    stats.parameters = safeCount(stats.parameters + count);
    stats.bytes = safeCount(stats.bytes + tensorBytes);
    component.dtypes[dtype] = stats;
    component.parameters = safeCount(component.parameters + count);
    component.bytes = safeCount(component.bytes + tensorBytes);
  }
  return { parameters: component.parameters, bytes: component.bytes, components: { Transformer: component }, metadata };
}

export async function fetchGgufMetadata(fetcher: FetchLike, url: string, headers: HeadersInit = {}, maxBytes = 100_000_000): Promise<ParsedGguf> {
  assertPositiveInteger(maxBytes, "maxBytes");
  let size = Math.min(1_000_000, maxBytes);
  for (;;) {
    const { bytes, eof } = await fetchRange(fetcher, url, 0, size - 1, headers);
    try { return parseMetadata(bytes, maxBytes); }
    catch (error) {
      if (!(error instanceof TruncatedGgufError) || size >= maxBytes || eof) throw error;
      size = size > maxBytes / 2 ? maxBytes : size * 2;
    }
  }
}

export function estimateGgufKvCache(
  metadata: Record<string, unknown>,
  options: KvCacheOptions = {},
): KvCacheEstimate {
  const assumptions: string[] = [];
  let architecture = metadata["general.architecture"];
  if (architecture !== undefined && (typeof architecture !== "string" || !architecture.trim())) {
    throw new Error("GGUF general.architecture must be a nonempty string.");
  }
  const hasDimensions = (namespace: string): boolean => ["embedding_length", "attention.head_count",
    "attention.head_count_kv", "attention.key_length", "attention.key_length_mla", "attention.kv_lora_rank",
    "ssm.state_size", "ssm.inner_size"].some((suffix) => metadata[`${namespace}.${suffix}`] !== undefined);
  if (architecture === undefined || metadata[`${architecture}.block_count`] === undefined || !hasDimensions(architecture as string)) {
    const namespaces = Object.keys(metadata).filter((key) => /^[^.]+\.block_count$/.test(key))
      .map((key) => key.slice(0, -".block_count".length)).filter(hasDimensions);
    if (namespaces.length !== 1) {
      throw new Error("GGUF cache estimation requires one unambiguous namespace with block_count and cache dimensions.");
    }
    architecture = namespaces[0]!;
    assumptions.push(`GGUF cache dimensions select the ${architecture} namespace because general.architecture does not identify usable block metadata.`);
  }
  const field = (...suffixes: string[]): unknown => {
    for (const suffix of suffixes) {
      const value = metadata[`${architecture}.${suffix}`];
      if (value !== undefined) return value;
    }
    return undefined;
  };
  const positive = (value: unknown, name: string): number => {
    if (!Number.isSafeInteger(value) || (value as number) <= 0) throw new RangeError(`${name} must be a positive safe integer.`);
    return value as number;
  };
  const layers = positive(field("block_count"), "block_count");
  const config: Record<string, unknown> = { num_hidden_layers: layers, torch_dtype: "float16" };
  const mappings: ReadonlyArray<readonly [string, ...string[]]> = [
    ["hidden_size", "embedding_length"],
    ["max_position_embeddings", "context_length"],
    ["num_attention_heads", "attention.head_count"],
    ["num_key_value_heads", "attention.head_count_kv"],
    ["key_head_dim", "attention.key_length_mla", "attention.key_length"],
    ["value_head_dim", "attention.value_length_mla", "attention.value_length"],
    ["global_num_attention_heads", "attention.head_count_global"],
    ["global_num_key_value_heads", "attention.head_count_kv_global"],
    ["global_key_head_dim", "attention.key_length_global"],
    ["global_value_head_dim", "attention.value_length_global"],
    ["sliding_window", "attention.sliding_window"],
    ["sliding_window_pattern", "attention.sliding_window_pattern"],
    ["full_attention_interval", "full_attention_interval", "attention.full_attention_interval"],
    ["layer_types", "layer_types", "attention.layer_types"],
    ["kv_lora_rank", "attention.kv_lora_rank"],
    ["qk_rope_head_dim", "rope.dimension_count"],
    ["state_size", "ssm.state_size"],
    ["intermediate_size", "ssm.inner_size"],
    ["conv_kernel", "ssm.conv_kernel"],
    ["n_groups", "ssm.group_count"],
    ["linear_num_key_heads", "linear_num_key_heads", "attention.linear_num_key_heads"],
    ["linear_num_value_heads", "linear_num_value_heads", "attention.linear_num_value_heads"],
    ["linear_key_head_dim", "linear_key_head_dim", "attention.linear_key_head_dim"],
    ["linear_value_head_dim", "linear_value_head_dim", "attention.linear_value_head_dim"],
    ["linear_conv_kernel_dim", "linear_conv_kernel_dim", "attention.linear_conv_kernel_dim"],
  ];
  for (const [key, ...suffixes] of mappings) {
    const value = field(...suffixes);
    if (value !== undefined) config[key] = value;
  }
  // GGUF's unsuffixed dimensions describe full attention; _swa overrides sliding layers.
  for (const [key, suffix] of [["key_head_dim", "key_length"], ["value_head_dim", "value_length"],
    ["num_attention_heads", "head_count"], ["num_key_value_heads", "head_count_kv"]] as const) {
    const sliding = field(`attention.${suffix}_swa`);
    if (sliding !== undefined) {
      config[`global_${key}`] ??= config[key];
      config[key] = sliding;
    }
  }
  if (config.n_groups === 0) delete config.n_groups;
  if (config.kv_lora_rank === 0) delete config.kv_lora_rank;
  if (config.kv_lora_rank === undefined) delete config.qk_rope_head_dim;
  if (config.sliding_window === 0) delete config.sliding_window;
  const hasSsm = Object.keys(metadata).some((key) => key.startsWith(`${architecture}.ssm.`));
  if (hasSsm) {
    // time_step_rank is NOT a portable head count: GGUF uses it for several different SSM layouts.
    assumptions.push("GGUF SSM dimensions use generic inner-size × state-size recurrent storage and convolution channels including grouped state inputs; time_step_rank is not treated as an attention head count. Backend history may retain kernel−1 rather than kernel convolution slots.");
  }
  const recurrent = field("attention.recurrent_layers");
  const slidingPattern = config.sliding_window_pattern;
  const headFields = [config.num_attention_heads, config.num_key_value_heads];
  const hasZeroHeads = headFields.some((value) => Array.isArray(value) ? value.includes(0) : value === 0);
  const at = (value: unknown, index: number): unknown => Array.isArray(value) ? value[index % value.length] : value;
  const flag = (value: unknown, name: string): boolean => {
    if (value !== true && value !== false && value !== 0 && value !== 1) {
      throw new RangeError(`${name} must contain boolean or 0/1 flags.`);
    }
    return value === true || value === 1;
  };
  if (recurrent !== undefined || hasZeroHeads || Array.isArray(slidingPattern) || slidingPattern === 0) {
    if (layers > 100_000) throw new RangeError("Per-layer cache descriptions are limited to 100000 layers.");
    for (const [name, value] of [["attention.recurrent_layers", recurrent],
      ["attention.sliding_window_pattern", slidingPattern], ["layer_types", config.layer_types],
      ["attention.head_count", config.num_attention_heads], ["attention.head_count_kv", config.num_key_value_heads]] as const) {
      if (Array.isArray(value) && !value.length) throw new RangeError(`${name} must not be empty.`);
      if (Array.isArray(value) && value.length !== layers) assumptions.push(`${name} has a different length than block_count; repeating/truncating its per-layer schedule.`);
    }
    if (config.layer_types !== undefined && !Array.isArray(config.layer_types)) {
      throw new RangeError("layer_types must be an array.");
    }
    const fullInterval = config.full_attention_interval === undefined ? undefined
      : positive(config.full_attention_interval, "full_attention_interval");
    const slidingInterval = slidingPattern === undefined || Array.isArray(slidingPattern) || slidingPattern === 0
      ? undefined : positive(slidingPattern, "attention.sliding_window_pattern");
    config.layer_types = Array.from({ length: layers }, (_, index) => {
      if (recurrent !== undefined && flag(at(recurrent, index), "attention.recurrent_layers")) return "ssm";
      if (headFields.some((value) => at(value, index) === 0)) return "ssm";
      if (Array.isArray(config.layer_types)) return at(config.layer_types, index);
      if (recurrent === undefined && fullInterval !== undefined && (index + 1) % fullInterval !== 0) return "ssm";
      const sliding = Array.isArray(slidingPattern) ? flag(at(slidingPattern, index), "attention.sliding_window_pattern")
        : slidingPattern === 0 || (slidingInterval !== undefined ? (index + 1) % slidingInterval !== 0 : config.sliding_window !== undefined);
      return sliding ? "sliding_attention" : "full_attention";
    });
    delete config.sliding_window_pattern;
    if (hasZeroHeads) assumptions.push("Zero GGUF attention-head entries identify recurrent layers; missing recurrent dimensions use an attention-sized proxy rather than dropping their state.");
  }
  if (field("attention.causal") === false || Object.keys(metadata).some((key) =>
    key.startsWith(`${architecture}.`) && /(^|\.)(encoder|cross_attention|decoder_start_token_id)(\.|$)/.test(key))) {
    assumptions.push("Noncausal, encoder or cross-attention metadata is estimated using autoregressive resident-cache dimensions; encoder/source lengths and backend cache lifetime are not modeled.");
  }
  if (Object.keys(metadata).some((key) => key.startsWith(`${architecture}.`) && /shared_kv|kv_shared/.test(key))) {
    assumptions.push("GGUF KV sharing is not deducted; each layer is allocated independently.");
  }
  if (field("attention.kv_lora_rank_swa", "attention.key_length_mla_swa", "attention.value_length_mla_swa", "rope.dimension_count_swa") !== undefined) {
    assumptions.push("Sliding-specific MLA/RoPE metadata uses the common compressed latent layout; backend-specific mixed latent layouts are not modeled.");
  }
  const requested = options.dtype === undefined ? "auto" : options.dtype;
  if (typeof requested !== "string" || !requested.trim()) throw new Error("GGUF KV-cache dtype must be a nonempty string.");
  const selected = requested.toUpperCase() === "AUTO"
    ? field("attention.kv_cache_dtype", "kv_cache_dtype") ?? metadata["general.kv_cache_dtype"] ?? "F16"
    : requested;
  let dtype = typeof selected === "string" ? selected.toUpperCase() : "";
  if (!Object.hasOwn(GGUF_DTYPE_BITS, dtype)) {
    if (requested.toUpperCase() !== "AUTO") throw new Error(`Unsupported GGUF KV-cache dtype: ${requested}.`);
    assumptions.push(`Unrecognized GGUF cache precision ${String(selected)}; assuming F16.`);
    dtype = "F16";
  }
  const bits = GGUF_DTYPE_BITS[dtype]!;
  const estimate = estimateSafetensorsKvCache(config, { ...options, dtype: "F16" });
  const attentionBytes = Math.ceil(estimate.attentionBytes / 16 * bits);
  const bytes = attentionBytes + estimate.stateBytes;
  if (!Number.isSafeInteger(attentionBytes) || !Number.isSafeInteger(bytes) || attentionBytes < 0 || bytes < 0) {
    throw new RangeError("Cache estimate exceeds JavaScript's safe integer range.");
  }
  const quantized = DTYPE_BLOCKS[dtype]![0] > 1;
  if (quantized) {
    assumptions.push(`GGUF ${dtype} uses ${bits} effective bits per attention-cache element including block scales, rounded up to whole bytes; row packing, block alignment and backend padding are unknown, so this is approximate, not a backend dtype-support guarantee.`);
  }
  return {
    ...estimate, dtype, bytes, attentionBytes,
    approximate: estimate.approximate || assumptions.length > 0,
    assumptions: [...estimate.assumptions,
      `GGUF ${architecture} is a metadata namespace, not a cache-layout restriction. Auto cache precision uses explicit cache dtype metadata or F16, independent of weight quantization; only attention storage is rescaled to ${dtype}.`,
      ...assumptions],
  };
}
