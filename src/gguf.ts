import { basename, dirname, escapeRegExp, int, isInt, product, str, truthy } from "./py.js";
import type { Source } from "./source.js";
import { Weights, choices } from "./weights.js";

export const FIRST_READ = 1 << 20;
export const PREFIX_READ = 1 << 17;
export const MAX_READ = 256 << 20;
export const LISTING_READ = 16 << 20;
/** Read sizes used at runtime; mutable so tests can shrink them (Python monkeypatches the module constants). */
export const reads = { FIRST_READ, PREFIX_READ, MAX_READ, LISTING_READ };
export const SHARD = /-(\d+)-of-(\d+)\.gguf$/;
const SKIP = ["mmproj", "imatrix"];
const QUANT = /(?:UD-)?(?:I?Q\d\w*?|T?Q\d_\d|[BM]?F\d+|NVFP4|MXFP4(?:_MOE)?)(?=(?:-\d+-of-\d+)?\.gguf$)/i;
const MMPROJ = ["Q8_0", "F16", "BF16", "F32"];

// ggml type id -> (name, block size, bytes per block)
export const TYPES: Record<number, [string, number, number]> = {
  0: ["F32", 1, 4],
  1: ["F16", 1, 2],
  2: ["Q4_0", 32, 18],
  3: ["Q4_1", 32, 20],
  6: ["Q5_0", 32, 22],
  7: ["Q5_1", 32, 24],
  8: ["Q8_0", 32, 34],
  9: ["Q8_1", 32, 36],
  10: ["Q2_K", 256, 84],
  11: ["Q3_K", 256, 110],
  12: ["Q4_K", 256, 144],
  13: ["Q5_K", 256, 176],
  14: ["Q6_K", 256, 210],
  15: ["Q8_K", 256, 292],
  16: ["IQ2_XXS", 256, 66],
  17: ["IQ2_XS", 256, 74],
  18: ["IQ3_XXS", 256, 98],
  19: ["IQ1_S", 256, 50],
  20: ["IQ4_NL", 32, 18],
  21: ["IQ3_S", 256, 110],
  22: ["IQ2_S", 256, 82],
  23: ["IQ4_XS", 256, 136],
  24: ["I8", 1, 1],
  25: ["I16", 1, 2],
  26: ["I32", 1, 4],
  27: ["I64", 1, 8],
  28: ["F64", 1, 8],
  29: ["IQ1_M", 256, 56],
  30: ["BF16", 1, 2],
  34: ["TQ1_0", 256, 54],
  35: ["TQ2_0", 256, 66],
  39: ["MXFP4", 32, 17],
};
export const KV_DTYPES: Record<string, number> = Object.fromEntries(
  Object.values(TYPES).map(([name, block, size]) => [name, size / block]),
);
// KV cache types accepted by llama.cpp's `--cache-type-k` / `--cache-type-v`
export const CACHE_TYPES = ["F16", "BF16", "F32", "Q8_0", "Q4_0", "Q4_1", "IQ4_NL", "Q5_0", "Q5_1"] as const;

const SCALAR: Record<number, [number, (view: DataView, offset: number) => number | boolean]> = {
  0: [1, (v, o) => v.getUint8(o)],
  1: [1, (v, o) => v.getInt8(o)],
  2: [2, (v, o) => v.getUint16(o, true)],
  3: [2, (v, o) => v.getInt16(o, true)],
  4: [4, (v, o) => v.getUint32(o, true)],
  5: [4, (v, o) => v.getInt32(o, true)],
  6: [4, (v, o) => v.getFloat32(o, true)],
  7: [1, (v, o) => v.getUint8(o) !== 0],
  10: [8, (v, o) => Number(v.getBigUint64(o, true))],
  11: [8, (v, o) => Number(v.getBigInt64(o, true))],
  12: [8, (v, o) => v.getFloat64(o, true)],
};
const SWA_PATTERN: Record<string, number> = { gemma2: 2, gemma3: 6, gemma3n: 5, cohere2: 4 };

const DECODER = new TextDecoder("utf-8", { ignoreBOM: true });

class Truncated extends Error {}

/** Python's `KeyError` on an unknown value type. */
class Invalid extends Error {}

function view(buf: Uint8Array): DataView {
  return new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
}

function u32(buf: Uint8Array, offset: number): number {
  if (offset + 4 > buf.length) throw new Truncated();
  return view(buf).getUint32(offset, true);
}

function u64(buf: Uint8Array, offset: number): number {
  if (offset + 8 > buf.length) throw new Truncated();
  return Number(view(buf).getBigUint64(offset, true));
}

function scalar(kind: number): [number, (view: DataView, offset: number) => number | boolean] {
  const entry = SCALAR[kind];
  if (entry === undefined) throw new Invalid(String(kind));
  return entry;
}

function skipStrings(buf: Uint8Array, offset: number, count: number): number {
  for (let i = 0; i < count; i++) offset += 8 + u64(buf, offset);
  if (offset > buf.length) throw new Truncated();
  return offset;
}

function value(buf: Uint8Array, offset: number, kind: number, keep: boolean): [any, number] {
  if (kind === 8) {
    const length = u64(buf, offset);
    offset += 8;
    if (offset + length > buf.length) throw new Truncated();
    return [keep ? DECODER.decode(buf.subarray(offset, offset + length)) : null, offset + length];
  }
  if (kind === 9) {
    const item = u32(buf, offset);
    const count = u64(buf, offset + 4);
    offset += 12;
    const entry = SCALAR[item];
    if (entry !== undefined) {
      const [size, get] = entry;
      const end = offset + count * size;
      if (end > buf.length) throw new Truncated();
      let values: (number | boolean)[] | null = null;
      if (keep) {
        const v = view(buf);
        values = [];
        for (let i = 0; i < count; i++) values.push(get(v, offset + i * size));
      }
      return [values, end];
    }
    if (item === 8 && !keep) return [null, skipStrings(buf, offset, count)];
    const items: any[] = [];
    for (let i = 0; i < count; i++) {
      let item_: any;
      [item_, offset] = value(buf, offset, item, keep);
      items.push(item_);
    }
    return [keep ? items : null, offset];
  }
  const [size, get] = scalar(kind);
  if (offset + size > buf.length) throw new Truncated();
  return [get(view(buf), offset), offset + size];
}

export class Header {
  constructor(
    public metadata: Record<string, any>,
    public weights: Weights,
    public dataStart: number,
    public offloaded: Weights = new Weights(),
  ) {}
}

function metadata_(
  buf: Uint8Array,
  prefix = false,
  partial = false,
): [Record<string, any>, number, number, boolean] {
  if (buf.length < 4 || DECODER.decode(buf.subarray(0, 4)) !== "GGUF") {
    throw new Error("not a valid GGUF file (magic number mismatch)");
  }
  u32(buf, 4);
  const tensorCount = u64(buf, 8);
  const kvCount = u64(buf, 16);
  let offset = 24;
  const metadata: Record<string, any> = {};
  let start = offset;
  try {
    for (let i = 0; i < kvCount; i++) {
      start = offset;
      let key: string;
      [key, offset] = value(buf, offset, 8, true);
      if (prefix && key.startsWith("tokenizer.")) return [metadata, offset, tensorCount, true];
      const kind = u32(buf, offset);
      let item: any;
      [item, offset] = value(buf, offset + 4, kind, !key.startsWith("tokenizer."));
      metadata[key] = item;
    }
  } catch (e) {
    if (!(e instanceof Truncated) || !partial) throw e;
    return [metadata, start, tensorCount, false];
  }
  return [metadata, offset, tensorCount, false];
}

export function parseHeader(buf: Uint8Array, fileSize: number): Header {
  let [metadata, offset, tensorCount] = metadata_(buf);
  const tensors: [number, number, number, string][] = [];
  const names = new Set<string>();
  for (let i = 0; i < tensorCount; i++) {
    let tensor: string;
    [tensor, offset] = value(buf, offset, 8, true);
    names.add(tensor);
    const dims = u32(buf, offset);
    if (offset + 4 + 8 * dims > buf.length) throw new Truncated();
    const shape: number[] = [];
    for (let d = 0; d < dims; d++) shape.push(u64(buf, offset + 4 + 8 * d));
    offset += 4 + 8 * dims;
    const kind = u32(buf, offset);
    const dataOffset = u64(buf, offset + 4);
    offset += 12;
    tensors.push([dataOffset, kind, product(shape), tensor]);
  }

  const alignment: number = "general.alignment" in metadata ? metadata["general.alignment"] : 32;
  const dataStart = Math.ceil(offset / alignment) * alignment;
  tensors.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || (a[3] < b[3] ? -1 : a[3] > b[3] ? 1 : 0));
  const ends = [...tensors.slice(1).map((t) => t[0]), fileSize - dataStart];
  const weights = new Weights();
  const offloaded = new Weights();
  tensors.forEach(([start, kind, params, tensor], i) => {
    const type = TYPES[kind];
    const [name, nbytes] =
      type !== undefined
        ? [type[0], Math.ceil(params / type[1]) * type[2]]
        : [`TYPE_${kind}`, ends[i]! - start];
    // llama.cpp keeps the token embeddings in host memory, and a copy on the GPU when tied to the output
    if (tensor === "token_embd.weight") {
      offloaded.add(name, params, nbytes);
      if (names.has("output.weight")) return;
    }
    weights.add(name, params, nbytes);
  });
  return new Header(metadata, weights, dataStart, offloaded);
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

export async function readHeader(source: Source, path: string): Promise<Header> {
  const size = source.files.get(path)!;
  let buf: Uint8Array = new Uint8Array(0);
  let end = Math.min(reads.FIRST_READ, size);
  for (;;) {
    buf = concat(buf, await source.read(path, buf.length, end));
    try {
      return parseHeader(buf, size);
    } catch (e) {
      if (e instanceof Truncated) {
        if (end >= Math.min(reads.MAX_READ, size)) {
          throw new Error(`\`${path}\` has a metadata section larger than ${reads.MAX_READ >> 20} MiB.`);
        }
        end = Math.min(end * 2, reads.MAX_READ, size);
      } else if (e instanceof Invalid) {
        throw new Error(`\`${path}\` is not a valid GGUF file (${e.message}).`, { cause: e });
      } else {
        throw e;
      }
    }
  }
}

function complete(metadata: Record<string, any>): boolean {
  const arch = str(metadata["general.architecture"]);
  const keys = ["block_count", "context_length", "embedding_length", "attention.head_count"];
  if (Object.keys(metadata).some((k) => k.startsWith(`${arch}.ssm.`)) || `${arch}.full_attention_interval` in metadata) {
    keys.push("ssm.conv_kernel", "ssm.state_size", "ssm.inner_size", "ssm.group_count");
  }
  return keys.every((k) => `${arch}.${k}` in metadata);
}

export async function readMetadata(source: Source, path: string): Promise<[Record<string, any>, boolean]> {
  const size = source.files.get(path)!;
  let buf: Uint8Array = new Uint8Array(0);
  let end = Math.min(reads.PREFIX_READ, size);
  let prefix = true;
  for (;;) {
    if (buf.length < end) buf = concat(buf, await source.read(path, buf.length, end));
    let parsed: [Record<string, any>, number, number, boolean] | null = null;
    try {
      parsed = metadata_(buf, prefix);
    } catch (e) {
      if (e instanceof Invalid) throw new Error(`\`${path}\` is not a valid GGUF file (${e.message}).`, { cause: e });
      if (!(e instanceof Truncated)) throw e;
    }
    if (parsed !== null) {
      const [metadata, , , tokenizer] = parsed;
      if (!prefix || !tokenizer || complete(metadata)) return [metadata, tokenizer || !prefix];
      prefix = false;
      continue;
    }
    const [metadata, offset] = metadata_(buf, prefix, true);
    let key: string;
    try {
      key = value(buf, offset, 8, true)[0];
    } catch (e) {
      if (!(e instanceof Truncated)) throw e;
      key = "tokenizer.";
    }
    const tokenizer = key.startsWith("tokenizer.");
    if (end >= Math.min(reads.LISTING_READ, size) || (!tokenizer && buf.length - offset > reads.PREFIX_READ)) {
      return [metadata, tokenizer];
    }
    end = Math.min(Math.max(end * 2, reads.FIRST_READ), reads.LISTING_READ, size);
  }
}

export function ggufPaths(files: Map<string, number>): string[] {
  return [...files.keys()]
    .filter((p) => p.endsWith(".gguf") && !SKIP.some((t) => basename(p).toLowerCase().includes(t)))
    .sort();
}

export function groupFiles(paths: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  for (const path of paths) {
    const match = SHARD.exec(path);
    const name = match ? path.slice(0, match.index) + ".gguf" : path;
    const group = groups.get(name);
    if (group === undefined) groups.set(name, [path]);
    else group.push(path);
  }
  return groups;
}

export function selectFile(groups: Map<string, string[]>, file: string): [string, string[]] {
  let target = file.replace(/^\/+|\/+$/g, "");
  const match = SHARD.exec(target);
  if (match) target = target.slice(0, match.index) + ".gguf";
  let found = new Map([...groups].filter(([name]) => name === target || name.startsWith(target + "/")));
  if (found.size === 0) found = new Map([...groups].filter(([name]) => basename(name) === target));
  if (found.size === 0) {
    const matches = tagged(groups, target);
    if (matches.length) {
      const key = (n: string): [number, string] => [basename(n).length, n];
      const name = matches.reduce((best, n) => {
        const [a, b] = [key(n), key(best)];
        return a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]) ? n : best;
      });
      return [name, groups.get(name)!];
    }
  }
  if (found.size === 0) {
    throw new Error(
      `No GGUF file matching \`${file}\` found, use a quantization tag or one of ${choices([...groups.keys()])}.`,
    );
  }
  if (found.size > 1) throw new Error(`Multiple GGUF files match \`${file}\`, use one of ${choices([...found.keys()])}.`);
  return [...found][0]!;
}

export function tagged(groups: Map<string, string[]>, tag: string): string[] {
  const pattern = new RegExp(`(?<![a-z0-9_])${escapeRegExp(tag)}(?![a-z0-9_])`, "i");
  return [...groups.keys()].filter((name) => pattern.test(basename(name)));
}

export function mmproj(files: Map<string, number>, name: string): string | null {
  const directory = dirname(name);
  const candidates = [...files.keys()]
    .filter(
      (p) =>
        p.endsWith(".gguf") &&
        basename(p).toLowerCase().includes("mmproj") &&
        (dirname(p) === directory || dirname(p) === ""),
    )
    .sort();
  if (!candidates.length) return null;
  const quant = QUANT.exec(basename(name));
  for (const tag of [...(quant ? [quant[0]] : []), ...MMPROJ]) {
    const matches = tagged(new Map(candidates.map((p) => [p, []])), tag);
    if (matches.length) return matches[0]!;
  }
  return candidates[0]!;
}

function get(metadata: Record<string, any>, key: string, fallback: any = null): any {
  return key in metadata ? metadata[key] : fallback;
}

function stateBytes(metadata: Record<string, any>, arch: string): number {
  const state = get(metadata, `${arch}.ssm.state_size`);
  if (truthy(state)) {
    const inner = get(metadata, `${arch}.ssm.inner_size`, 0);
    const groups = get(metadata, `${arch}.ssm.group_count`) || 1;
    const kernel = get(metadata, `${arch}.ssm.conv_kernel`, 4);
    return 4 * ((kernel - 1) * (inner + 2 * groups * state) + inner * state);
  }
  const cache = get(metadata, `${arch}.shortconv.l_cache`);
  if (truthy(cache)) return 4 * get(metadata, `${arch}.embedding_length`, 0) * (cache - 1);
  return 0;
}

export function kvCache(
  metadata: Record<string, any>,
  maxModelLen: number,
  batchSize: number,
  dtype: string,
): [number, Record<string, number>] | null {
  const arch = str(get(metadata, "general.architecture"));
  let blocks = get(metadata, `${arch}.block_count`);
  let heads = get(metadata, `${arch}.attention.head_count`);
  const state = stateBytes(metadata, arch);
  if (!isInt(blocks) || !(truthy(heads) || state)) return null;
  blocks -= get(metadata, `${arch}.nextn_predict_layers`, 0);
  heads = (Array.isArray(heads) ? Math.max(...heads) : heads) || 0;

  let kvHeads = get(metadata, `${arch}.attention.head_count_kv`, heads);
  if (Array.isArray(kvHeads)) {
    kvHeads = kvHeads.slice(0, Math.max(0, blocks));
  } else {
    const interval = get(metadata, `${arch}.full_attention_interval`);
    const kv = kvHeads;
    kvHeads = Array.from({ length: Math.max(0, blocks) }, (_, i) => (!interval || (i + 1) % interval === 0 ? kv : 0));
  }
  const kvList: number[] = kvHeads;

  let widths: number[];
  const rank = get(metadata, `${arch}.attention.kv_lora_rank`);
  if (truthy(rank)) {
    const rope = get(metadata, `${arch}.rope.dimension_count`, 0);
    widths = kvList.map((h) => (h ? rank + rope : 0));
  } else {
    const embedding = get(metadata, `${arch}.embedding_length`, 0);
    const key = get(metadata, `${arch}.attention.key_length`, heads ? Math.floor(embedding / heads) : 0);
    const value_ = get(metadata, `${arch}.attention.value_length`, key);
    widths = kvList.map((h) => h * (key + value_));
  }

  const window = get(metadata, `${arch}.attention.sliding_window`);
  const pattern = get(metadata, `${arch}.attention.sliding_window_pattern`);
  let sliding: boolean[];
  const swa = SWA_PATTERN[arch];
  if (Array.isArray(pattern) && pattern.length >= blocks) {
    sliding = pattern.slice(0, Math.max(0, blocks)).map((p) => truthy(p));
  } else if (truthy(window) && swa !== undefined) {
    sliding = Array.from({ length: Math.max(0, blocks) }, (_, i) => (i + 1) % swa !== 0);
  } else {
    sliding = Array.from({ length: Math.max(0, blocks) }, () => false);
  }
  const swaTokens = Math.min(maxModelLen, (window || maxModelLen) + 512);

  const pairs = widths.slice(0, sliding.length).map((w, i): [number, boolean] => [w, sliding[i]!]);
  const tokens = pairs.reduce((total, [w, s]) => total + w * (s ? swaTokens : maxModelLen), 0);
  const empty = widths.filter((w) => !w).length;
  const recurrent = state ? empty : 0;
  const layers = {
    full: pairs.filter(([w, s]) => w && !s).length,
    sliding: pairs.filter(([w, s]) => w && s).length,
    state: recurrent,
    none: empty - recurrent,
  };
  return [int((tokens * KV_DTYPES[dtype]! + recurrent * state) * batchSize), layers];
}

export async function readGroup(source: Source, paths: string[]): Promise<Header> {
  const headers = await Promise.all(paths.map((p) => readHeader(source, p)));
  const weights = new Weights();
  const offloaded = new Weights();
  for (const header of headers) {
    weights.merge(header.weights);
    offloaded.merge(header.offloaded);
  }
  const first = headers[0]!;
  return new Header(first.metadata, weights, first.dataStart, offloaded);
}
