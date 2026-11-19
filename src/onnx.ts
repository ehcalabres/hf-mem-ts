import { basename, dirname, product } from "./py.js";
import type { Source } from "./source.js";
import { Weights, choices } from "./weights.js";

export const CHUNK = 1 << 20;
export let SMALL_FILE = 16 << 20;
export const TAIL_SIZE = 1 << 20;
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto (TensorProto.DataType)
export const TYPES: Record<number, [string, number]> = {
  1: ["F32", 32],
  2: ["U8", 8],
  3: ["I8", 8],
  4: ["U16", 16],
  5: ["I16", 16],
  6: ["I32", 32],
  7: ["I64", 64],
  9: ["BOOL", 8],
  10: ["F16", 16],
  11: ["F64", 64],
  12: ["U32", 32],
  13: ["U64", 64],
  16: ["BF16", 16],
  17: ["F8_E4M3", 8],
  18: ["F8_E4M3", 8],
  19: ["F8_E5M2", 8],
  20: ["F8_E5M2", 8],
  21: ["U4", 4],
  22: ["I4", 4],
  23: ["F4", 4],
};
const VARIANT = /_(fp16|q8|int8|uint8|q4|q4f16|bnb4|quantized)$/;
const DATA = /(?<=\.onnx)[._]data(?:_\d+)?$/;
// New Transformers exporters flatten Cache objects as `cache.layers.0.keys`, sometimes
// prefixed with `input.` to disambiguate graph inputs from outputs.
const CACHE_NAME = "(?:input\\.)?(?:past_|cache\\.|cache_params\\.)";
const CACHE_NAME_RE = new RegExp(`^${CACHE_NAME}`);
const CACHE_INPUT = new RegExp(`\\x5a[\\x80-\\xff]{0,3}[\\x00-\\x7f]\\x0a[\\x00-\\x7f]${CACHE_NAME}`, "g");
const PARTS = ["embed_tokens", "vision_encoder", "audio_encoder", "speech_encoder", "encoder_model"];
const DECODERS = ["decoder_model_merged", "decoder_model", "decoder_with_past_model"];
// Weight input of the ONNX Runtime ops that pack sub-byte weights into UINT8 blobs
const QUANTIZED: Record<string, number> = { MatMulNBits: 1, GatherBlockQuantized: 0 };

export function setSmallFile(value: number): void {
  SMALL_FILE = value;
}

export class Graph {
  weights: Weights = new Weights();
  cache: [string, number, number, boolean][] = [];
  packed: Map<string, number> = new Map();
}

function compare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

export function groupFiles(files: Map<string, number>): Map<string, string[]> {
  const variants = new Map<string, [string, string, Map<string, string>]>();
  const data = new Map<string, string[]>();
  for (const path of [...files.keys()].sort(compare)) {
    let match: RegExpExecArray | null;
    if (path.endsWith(".onnx")) {
      const stem = path.slice(0, -5);
      const directory = dirname(stem);
      const name = basename(stem);
      const found = VARIANT.exec(name);
      const [variant, part] = found ? [found[1]!, name.slice(0, found.index)] : ["", name];
      const key = JSON.stringify([directory, variant]);
      let entry = variants.get(key);
      if (entry === undefined) {
        entry = [directory, variant, new Map()];
        variants.set(key, entry);
      }
      entry[2].set(part, path);
    } else if ((match = DATA.exec(path))) {
      const key = path.slice(0, match.index);
      let list = data.get(key);
      if (list === undefined) {
        list = [];
        data.set(key, list);
      }
      list.push(path);
    }
  }
  const groups = new Map<string, string[]>();
  for (const [directory, variant, parts] of variants.values()) {
    for (const [part, path] of parts) {
      if (!PARTS.includes(part) && !DECODERS.includes(part)) groups.set(path, [path]);
    }
    const decoders = parts.has("decoder_model_merged") ? ["decoder_model_merged"] : DECODERS.slice(1);
    const loaded = [...PARTS, ...decoders].filter((p) => parts.has(p)).map((p) => parts.get(p)!);
    if (loaded.length) {
      const names = loaded.map((p) => {
        const name = basename(p).slice(0, -5);
        const suffix = `_${variant}`;
        return name.endsWith(suffix) ? name.slice(0, -suffix.length) : name;
      });
      const name = names.join("+") + (variant ? `_${variant}` : "");
      const label = loaded.length === 1 ? loaded[0]! : directory ? `${directory}/${name}` : name;
      groups.set(label, loaded);
    }
  }
  return new Map(
    [...groups.entries()]
      .sort((a, b) => compare(a[0], b[0]))
      .map(([label, paths]) => [label, [...paths, ...paths.flatMap((p) => data.get(p) ?? [])]]),
  );
}

export function selectFile(groups: Map<string, string[]>, file: string): [string, string[]] {
  const target = file.replace(/^\/+|\/+$/g, "");
  let found = [...groups.entries()].filter(
    ([label, paths]) => label === target || paths.includes(target) || label.startsWith(target + "/"),
  );
  if (!found.length) found = [...groups.entries()].filter(([label]) => basename(label) === target);
  if (!found.length) {
    throw new Error(`No ONNX file matching \`${file}\` found, use one of ${choices([...groups.keys()])}.`);
  }
  if (found.length === 1) return found[0]!;
  throw new Error(`Multiple ONNX files match \`${file}\`, use one of ${choices(found.map(([label]) => label))}.`);
}

const decoder = new TextDecoder();

class Reader {
  source: Source;
  path: string;
  size: number;
  chunk: number;
  start = 0;
  buffer: Uint8Array = new Uint8Array(0);

  constructor(source: Source, path: string) {
    this.source = source;
    this.path = path;
    this.size = source.files.get(path)!;
    this.chunk = this.size <= SMALL_FILE ? this.size : CHUNK;
  }

  async read(pos: number, n: number): Promise<Uint8Array> {
    if (pos < this.start || pos + n > this.start + this.buffer.length) {
      this.start = pos;
      this.buffer = await this.source.read(this.path, pos, Math.min(this.size, pos + Math.max(n, this.chunk)));
    }
    return this.buffer.subarray(pos - this.start, pos - this.start + n);
  }

  async varint(pos: number): Promise<[number, number]> {
    let value = 0;
    const data = await this.read(pos, Math.min(10, this.size - pos));
    for (let i = 0; i < data.length; i++) {
      const byte = data[i]!;
      value += (byte & 0x7f) * 2 ** (7 * i);
      if (byte < 0x80) return [value, pos + i + 1];
    }
    throw new Error(`\`${this.path}\` is not a valid ONNX file.`);
  }

  async *fields(start: number, end: number): AsyncGenerator<[number, number, number, number]> {
    let pos = start;
    while (pos < end) {
      let tag: number;
      [tag, pos] = await this.varint(pos);
      const number = Math.floor(tag / 8);
      const wire = tag % 8;
      if (wire === 0) {
        let value: number;
        [value, pos] = await this.varint(pos);
        yield [number, wire, value, pos];
      } else if (wire === 2) {
        let length: number;
        [length, pos] = await this.varint(pos);
        yield [number, wire, pos, pos + length];
        pos += length;
      } else if (wire === 1 || wire === 5) {
        pos += wire === 1 ? 8 : 4;
      } else {
        throw new Error(`\`${this.path}\` is not a valid ONNX file.`);
      }
    }
    if (pos !== end) throw new Error(`\`${this.path}\` is not a valid ONNX file.`);
  }
}

function packed(data: Uint8Array): number[] {
  const values: number[] = [];
  let value = 0;
  let shift = 0;
  for (const byte of data) {
    value += (byte & 0x7f) * 2 ** shift;
    shift += 7;
    if (byte < 0x80) {
      values.push(value);
      value = shift = 0;
    }
  }
  return values;
}

async function tensor(reader: Reader, start: number, end: number, graph: Graph): Promise<void> {
  let dtype = 0;
  const dims: number[] = [];
  let name = "";
  for await (const [number, wire, a, b] of reader.fields(start, end)) {
    if (number === 1) {
      dims.push(...(wire === 0 ? [a] : packed(await reader.read(a, b - a))));
    } else if (number === 2) {
      dtype = a;
    } else if (number === 8) {
      name = decoder.decode(await reader.read(a, b - a));
    }
  }
  const [label, bits] = TYPES[dtype] ?? [`TYPE${dtype}`, 8];
  const elements = product(dims);
  graph.weights.add(label, elements * (graph.packed.get(name) ?? 1), Math.ceil((elements * bits) / 8));
}

async function node(reader: Reader, start: number, end: number, graph: Graph): Promise<void> {
  const inputs: string[] = [];
  let op = "";
  let bits = 4;
  for await (const [number, , a, b] of reader.fields(start, end)) {
    if (number === 1) {
      inputs.push(decoder.decode(await reader.read(a, b - a)));
    } else if (number === 4) {
      op = decoder.decode(await reader.read(a, b - a));
      if (!(op in QUANTIZED)) return;
    } else if (number === 5) {
      let attribute = "";
      let value: number | null = null;
      for await (const [key, wire, c, d] of reader.fields(a, b)) {
        if (key === 1) {
          attribute = decoder.decode(await reader.read(c, d - c));
        } else if (key === 3 && wire === 0) {
          value = c;
        }
      }
      if (attribute === "bits" && value) bits = value;
    }
  }
  const index = QUANTIZED[op];
  if (index !== undefined && inputs.length > index && 0 < bits && bits < 8) {
    graph.packed.set(inputs[index]!, Math.floor(8 / bits));
  }
}

async function dimensions(reader: Reader, start: number, end: number): Promise<(number | null)[]> {
  const dims: (number | null)[] = [];
  for await (const [number, , a, b] of reader.fields(start, end)) {
    if (number === 1) {
      let value: number | null = null;
      for await (const [key, wire, v] of reader.fields(a, b)) {
        if (key === 1 && wire === 0) value = v;
      }
      dims.push(value);
    }
  }
  return dims;
}

async function input(reader: Reader, start: number, end: number, graph: Graph): Promise<void> {
  let name = "";
  let dtype = 0;
  let dims: (number | null)[] = [];
  for await (const [number, , a, b] of reader.fields(start, end)) {
    if (number === 1) {
      name = decoder.decode(await reader.read(a, b - a));
    } else if (number === 2 && CACHE_NAME_RE.test(name)) {
      for await (const [kind, , c, d] of reader.fields(a, b)) {
        if (kind === 1) {
          for await (const [key, , e, f] of reader.fields(c, d)) {
            if (key === 1) {
              dtype = e;
            } else if (key === 2) {
              dims = await dimensions(reader, e, f);
            }
          }
        }
      }
    }
  }
  if (CACHE_NAME_RE.test(name) && dims.length) {
    const [dtypeName, bits] = TYPES[dtype] ?? ["F32", 32];
    const fixed = product(dims.filter((d): d is number => !!d));
    const encoder = name.includes(".encoder.") || name.includes(".cross_attention_cache.");
    graph.cache.push([dtypeName, Math.floor((fixed * bits) / 8), dims.filter((d) => !d).length, encoder]);
  }
}

async function walk(reader: Reader, graph: Graph): Promise<void> {
  for await (const [number, wire, start, end] of reader.fields(0, reader.size)) {
    if (number !== 7 || wire !== 2) continue;
    for await (const [fieldNumber, fieldWire, a, b] of reader.fields(start, end)) {
      if (fieldNumber === 1 && fieldWire === 2) {
        await node(reader, a, b, graph);
      } else if (fieldNumber === 5 && fieldWire === 2) {
        await tensor(reader, a, b, graph);
      } else if (fieldNumber === 11 && fieldWire === 2) {
        await input(reader, a, b, graph);
      }
    }
    return;
  }
  throw new Error(`\`${reader.path}\` is not a valid ONNX file.`);
}

async function tail(reader: Reader, graph: Graph): Promise<void> {
  const start = Math.max(0, reader.size - TAIL_SIZE);
  const text = Buffer.from(await reader.read(start, reader.size - start)).toString("latin1");
  const seen = new Set<string>();
  for (const match of text.matchAll(CACHE_INPUT)) {
    const [length, pos] = await reader.varint(start + match.index! + 1);
    if (pos + length <= reader.size) {
      const key = text.slice(match.index, pos - start + length);
      if (!seen.has(key)) {
        seen.add(key);
        await input(reader, pos, pos + length, graph);
      }
    }
  }
}

export async function readGroup(source: Source, paths: string[]): Promise<Graph> {
  const read = async (path: string): Promise<Graph> => {
    const reader = new Reader(source, path);
    const graph = new Graph();
    if (reader.size <= SMALL_FILE) {
      await walk(reader, graph);
    } else {
      const data = paths.filter((p) => p.startsWith(path) && DATA.test(p));
      graph.weights.bytes = [path, ...data].reduce((total, p) => total + source.files.get(p)!, 0);
      await tail(reader, graph);
    }
    return graph;
  };

  const merged = new Graph();
  for (const graph of await Promise.all(paths.filter((p) => p.endsWith(".onnx")).map(read))) {
    merged.weights.merge(graph.weights);
    merged.cache.push(...graph.cache);
  }
  return merged;
}

export function kvCache(
  graph: Graph,
  maxModelLen: number,
  batchSize: number,
  encoderLength: number | null = null,
): [number, string, number] | null {
  if (!graph.cache.length) return null;
  let perToken = 0;
  let fixed = 0;
  for (const [, b, symbolic, encoder] of graph.cache) {
    if (symbolic >= 2 && !encoder) perToken += b;
    if (encoder || symbolic < 2) fixed += b * (symbolic >= 2 ? encoderLength || maxModelLen : 1);
  }
  const counts = new Map<string, number>();
  for (const [dtype] of graph.cache) counts.set(dtype, (counts.get(dtype) ?? 0) + 1);
  let best = "";
  let most = 0;
  for (const [dtype, n] of counts) {
    if (n > most) {
      best = dtype;
      most = n;
    }
  }
  return [(perToken * maxModelLen + fixed) * batchSize, best, perToken * batchSize];
}
