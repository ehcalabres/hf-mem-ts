import { fetchRange } from "./http.js";
import type { ComponentStats, FetchLike, KvCacheEstimate, WeightMetadata } from "./types.js";

export const GGUF_DTYPE_BITS: Readonly<Record<string, number>> = {
  F32: 32, F16: 16, Q4_0: 4.5, Q4_1: 5, Q5_0: 5.5, Q5_1: 6, Q8_0: 8.5, Q8_1: 9,
  Q2_K: 2.625, Q3_K: 3.4375, Q4_K: 4.5, Q5_K: 5.5, Q6_K: 6.5625, Q8_K: 8.03125,
  IQ2_XXS: 2.06, IQ2_XS: 2.31, IQ3_XXS: 3.06, IQ1_S: 1.56, IQ4_NL: 4.5, IQ3_S: 3.44,
  IQ2_S: 2.5, IQ4_XS: 4.25, I8: 8, I16: 16, I32: 32, I64: 64, F64: 64, IQ1_M: 1.75,
  BF16: 16, TQ1_0: 1.6875, TQ2_0: 2.0625, MXFP4: 4.25,
};

const DTYPE_NAMES: Readonly<Record<number, string>> = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
  10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
  17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
  24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
  34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
};

class Cursor {
  offset = 0;
  readonly view: DataView;
  constructor(readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  need(length: number): void {
    if (this.offset + length > this.bytes.byteLength) throw new RangeError("Truncated GGUF metadata.");
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
  string(): string { const length = this.u64(); this.need(length); const value = new TextDecoder().decode(this.bytes.subarray(this.offset, this.offset + length)); this.offset += length; return value; }
}

function safe(value: bigint): number {
  if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) throw new RangeError("GGUF integer exceeds JavaScript's safe integer range.");
  return Number(value);
}

function readValue(cursor: Cursor, type: number): unknown {
  switch (type) {
    case 0: return cursor.u8(); case 1: return cursor.i8(); case 2: return cursor.u16(); case 3: return cursor.i16();
    case 4: return cursor.u32(); case 5: return cursor.i32(); case 6: return cursor.f32(); case 7: return Boolean(cursor.u8());
    case 8: return cursor.string();
    case 9: { const childType = cursor.u32(); const length = cursor.u64(); const values = new Array<unknown>(length); for (let i = 0; i < length; i++) values[i] = readValue(cursor, childType); return values; }
    case 10: return cursor.u64(); case 11: return cursor.i64(); case 12: return cursor.f64();
    default: throw new Error(`Unsupported GGUF metadata value type: ${type}.`);
  }
}

export interface ParsedGguf extends WeightMetadata {
  metadata: Record<string, unknown>;
}

export function parseGguf(bytes: Uint8Array): ParsedGguf {
  if (new TextDecoder().decode(bytes.subarray(0, 4)) !== "GGUF") throw new Error("Not a GGUF file (magic number mismatch).");
  const cursor = new Cursor(bytes);
  cursor.offset = 4;
  const version = cursor.u32();
  if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version: ${version}.`);
  const tensorCount = cursor.u64();
  const metadataCount = cursor.u64();
  const metadata: Record<string, unknown> = {};
  for (let i = 0; i < metadataCount; i++) {
    const key = cursor.string();
    metadata[key] = readValue(cursor, cursor.u32());
  }
  const component: ComponentStats = { parameters: 0, bytes: 0, dtypes: {} };
  for (let i = 0; i < tensorCount; i++) {
    cursor.string();
    const dimensions = cursor.u32();
    let count = 1;
    for (let dimension = 0; dimension < dimensions; dimension++) count *= cursor.u64();
    const type = cursor.u32();
    cursor.u64(); // data offset
    if (!Number.isSafeInteger(count)) throw new RangeError("GGUF tensor parameter count exceeds JavaScript's safe integer range.");
    const dtype = DTYPE_NAMES[type];
    const bits = dtype ? GGUF_DTYPE_BITS[dtype] : undefined;
    if (!dtype || bits === undefined) throw new Error(`Unsupported GGUF tensor type: ${type}.`);
    const tensorBytes = Math.floor((count * bits) / 8);
    const stats = component.dtypes[dtype] ?? { parameters: 0, bytes: 0 };
    stats.parameters += count;
    stats.bytes += tensorBytes;
    component.dtypes[dtype] = stats;
    component.parameters += count;
    component.bytes += tensorBytes;
  }
  return { parameters: component.parameters, bytes: component.bytes, components: { Transformer: component }, metadata };
}

export async function fetchGgufMetadata(fetcher: FetchLike, url: string, headers: HeadersInit = {}, maxBytes = 100_000_000): Promise<ParsedGguf> {
  let size = 1_000_000;
  for (;;) {
    const bytes = await fetchRange(fetcher, url, 0, size - 1, headers);
    try { return parseGguf(bytes); }
    catch (error) {
      if (!(error instanceof RangeError) || size >= maxBytes) throw error;
      size = Math.min(size * 2, maxBytes);
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
