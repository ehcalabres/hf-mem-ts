import { assertPositiveInteger, fetchRange } from "./http.js";
import type { ComponentStats, FetchLike, KvCacheEstimate, WeightMetadata } from "./types.js";

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

const KV_SUFFIXES = ["block_count", "head_count_kv", "head_count", "embedding_length", "context_length"] as const;

export function estimateGgufKvCache(
  metadata: Record<string, unknown>,
  options: { maxModelLen?: number; batchSize?: number; dtype?: string } = {},
): KvCacheEstimate {
  const values: Record<string, number> = {};
  for (const suffix of KV_SUFFIXES) {
    const entry = Object.entries(metadata).find(([key]) => key.endsWith(suffix));
    if (entry && Number.isSafeInteger(entry[1]) && (entry[1] as number) > 0) values[suffix] = entry[1] as number;
  }
  if (options.maxModelLen !== undefined) values.context_length = options.maxModelLen;
  const missing = KV_SUFFIXES.filter((key) => !values[key]);
  if (missing.length) throw new Error(`GGUF metadata lacks KV-cache fields: ${missing.join(", ")}.`);
  const dtype = (options.dtype ?? "F16").toUpperCase() === "AUTO" ? "F16" : (options.dtype ?? "F16").toUpperCase();
  const bits = GGUF_DTYPE_BITS[dtype];
  if (bits === undefined) throw new Error(`Unsupported GGUF KV-cache dtype: ${dtype}.`);
  const batchSize = options.batchSize ?? 1;
  if (!Number.isSafeInteger(batchSize) || batchSize <= 0) throw new RangeError("batchSize must be a positive safe integer.");
  const maxModelLen = values.context_length!;
  const headDim = Math.floor(values.embedding_length! / values.head_count!);
  const bytes = Math.floor(values.block_count! * 2 * values.head_count_kv! * headDim * maxModelLen * batchSize * bits / 8);
  if (!Number.isSafeInteger(bytes)) throw new RangeError("KV-cache estimate exceeds JavaScript's safe integer range.");
  return { bytes, dtype, maxModelLen, batchSize };
}
