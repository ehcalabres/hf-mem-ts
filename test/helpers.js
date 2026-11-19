import { mkdirSync, mkdtempSync, openSync, closeSync, writeSync, ftruncateSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

import { Estimate, Listing, run } from "../dist/index.js";
import { TYPES } from "../dist/gguf.js";

const BITS = { F64: 64, F32: 32, F16: 16, BF16: 16, I64: 64, I32: 32, F8_E4M3: 8, F8_E8M0: 8, U8: 8, I8: 8 };

export function tmpPath() {
  return mkdtempSync(join(tmpdir(), "hf-mem-ts-"));
}

const prod = (values) => values.reduce((a, b) => a * b, 1);

function writeSparse(path, head, dataSize) {
  mkdirSync(dirname(path), { recursive: true });
  const fd = openSync(path, "w");
  try {
    writeSync(fd, head);
    ftruncateSync(fd, head.length + dataSize);
  } finally {
    closeSync(fd);
  }
}

/** Python `json.dumps` default separators, so header bytes match the Python tests. */
function pyJson(value) {
  if (Array.isArray(value)) return `[${value.map(pyJson).join(", ")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .map(([k, v]) => `${JSON.stringify(k)}: ${pyJson(v)}`)
      .join(", ")}}`;
  }
  return JSON.stringify(value);
}

/** `tensors`: object or array of `[name, [dtype, shape]]`. */
export function writeSafetensors(path, tensors) {
  const header = { __metadata__: { format: "pt" } };
  let offset = 0;
  for (const [name, [dtype, shape]] of Array.isArray(tensors) ? tensors : Object.entries(tensors)) {
    const size = (prod(shape) * BITS[dtype]) / 8;
    header[name] = { dtype, shape, data_offsets: [offset, offset + size] };
    offset += size;
  }
  const raw = Buffer.from(pyJson(header));
  const head = Buffer.alloc(8 + raw.length);
  head.writeBigUInt64LE(BigInt(raw.length), 0);
  raw.copy(head, 8);
  writeSparse(path, head, offset);
  return offset;
}

export function writeJson(path, data) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(data));
}

function u32(value) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(value);
  return b;
}

function u64(value) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(value));
  return b;
}

function string(value) {
  const raw = Buffer.from(value);
  return Buffer.concat([u64(raw.length), raw]);
}

/** GGUF metadata value; integers are I64, other numbers F32 (`{ float: x }` forces F32 for whole numbers). */
function value(v) {
  if (typeof v === "boolean") return Buffer.concat([u32(7), Buffer.from([v ? 1 : 0])]);
  if (typeof v === "number" && Number.isInteger(v)) {
    const b = Buffer.alloc(8);
    b.writeBigInt64LE(BigInt(v));
    return Buffer.concat([u32(11), b]);
  }
  if (typeof v === "number" || (v && typeof v === "object" && "float" in v)) {
    const b = Buffer.alloc(4);
    b.writeFloatLE(typeof v === "number" ? v : v.float);
    return Buffer.concat([u32(6), b]);
  }
  if (typeof v === "string") return Buffer.concat([u32(8), string(v)]);
  if (Array.isArray(v)) {
    if (v.every((x) => typeof x === "string")) {
      return Buffer.concat([u32(9), u32(8), u64(v.length), ...v.map(string)]);
    }
    if (v.every((x) => typeof x === "boolean")) {
      return Buffer.concat([u32(9), u32(7), u64(v.length), Buffer.from(v.map((x) => (x ? 1 : 0)))]);
    }
    const body = Buffer.alloc(4 * v.length);
    v.forEach((x, i) => body.writeInt32LE(x, 4 * i));
    return Buffer.concat([u32(9), u32(5), u64(v.length), body]);
  }
  throw new TypeError(String(v));
}

/** `tensors`: array of `[name, ggmlType, shape]`. */
export function writeGguf(path, metadata, tensors) {
  const kv = Buffer.concat(Object.entries(metadata).map(([k, v]) => Buffer.concat([string(k), value(v)])));
  const infos = [];
  let offset = 0;
  for (const [name, kind, shape] of tensors) {
    const [, block, size] = TYPES[kind] ?? ["", 1, 1];
    const nbytes = Math.ceil(prod(shape) / block) * size;
    infos.push(string(name), u32(shape.length), ...shape.map(u64), u32(kind), u64(offset));
    offset += Math.ceil(nbytes / 32) * 32;
  }
  let head = Buffer.concat([
    Buffer.from("GGUF"),
    u32(3),
    u64(tensors.length),
    u64(Object.keys(metadata).length),
    kv,
    ...infos,
  ]);
  head = Buffer.concat([head, Buffer.alloc((32 - (head.length % 32)) % 32)]);
  writeSparse(path, head, offset);
  return offset;
}

export function llamaConfig(overrides = {}) {
  return {
    architectures: ["LlamaForCausalLM"],
    hidden_size: 256,
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    max_position_embeddings: 1024,
    torch_dtype: "bfloat16",
    ...overrides,
  };
}

export function kimiConfig(full = 3, state = 9, sparse = true) {
  const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
  const config = llamaConfig({
    model_type: "kimi_linear",
    architectures: ["KimiLinearForCausalLM"],
    hidden_size: 7168,
    num_hidden_layers: full + state,
    num_attention_heads: 96,
    num_key_value_heads: 96,
    kv_lora_rank: 512,
    qk_rope_head_dim: 64,
    max_position_embeddings: 1048576,
    linear_attn_config: {
      full_attn_layers: range(1, full + 1),
      kda_layers: range(full + 1, full + state + 1),
      num_heads: 96,
      head_dim: 128,
      short_conv_kernel_size: 4,
    },
  });
  if (sparse) config.dsa_config = { dsa_layers: range(1, full + 1), index_head_dim: 128 };
  return config;
}

export function llamaTensors(layers = 4) {
  const tensors = { "model.embed_tokens.weight": ["BF16", [1000, 256]] };
  for (let i = 0; i < layers; i++) {
    tensors[`model.layers.${i}.self_attn.q_proj.weight`] = ["BF16", [256, 256]];
    tensors[`model.layers.${i}.mlp.up_proj.weight`] = ["BF16", [512, 256]];
    tensors[`model.layers.${i}.input_layernorm.weight`] = ["F32", [256]];
  }
  return tensors;
}

export async function estimate(model, options) {
  const result = await run(model, options);
  assert.ok(result instanceof Estimate);
  return result;
}

export async function listing(model, options) {
  const result = await run(model, options);
  assert.ok(result instanceof Listing);
  return result;
}

function varint(v) {
  const out = [];
  for (;;) {
    out.push((v & 0x7f) | (v > 0x7f ? 0x80 : 0));
    v = Math.floor(v / 128);
    if (!v) return Buffer.from(out);
  }
}

function field(number, v) {
  if (typeof v === "number") return Buffer.concat([varint(number * 8), varint(v)]);
  return Buffer.concat([varint(number * 8 + 2), varint(v.length), v]);
}

const ONNX_BITS = { 1: 32, 2: 8, 7: 64, 10: 16, 16: 16, 22: 4 };

/**
 * `initializers`: `[name, dtype, dims]`; `inputs`: `[name, dtype, dims (int or symbolic string)]`;
 * `nodes`: `[op, inputNames, bits]`.
 */
export function writeOnnx(path, initializers, inputs = [], external = null, nodes = []) {
  const parts = [];
  let data = 0;
  for (const [op, names, bits] of nodes) {
    const attribute = Buffer.concat([field(1, Buffer.from("bits")), field(3, bits)]);
    parts.push(
      field(
        1,
        Buffer.concat([
          ...names.map((n) => field(1, Buffer.from(n))),
          field(4, Buffer.from(op)),
          field(5, attribute),
        ]),
      ),
    );
  }
  for (const [name, dtype, dims] of initializers) {
    const size = Math.ceil((prod(dims) * ONNX_BITS[dtype]) / 8);
    let tensor = Buffer.concat([...dims.map((d) => field(1, d)), field(2, dtype), field(8, Buffer.from(name))]);
    if (external) {
      const entries = [
        ["location", external],
        ["offset", String(data)],
        ["length", String(size)],
      ];
      tensor = Buffer.concat([
        tensor,
        ...entries.map(([k, v]) => field(13, Buffer.concat([field(1, Buffer.from(k)), field(2, Buffer.from(v))]))),
        field(14, 1),
      ]);
    } else {
      tensor = Buffer.concat([tensor, field(9, Buffer.alloc(size))]);
    }
    parts.push(field(5, tensor));
    data += size;
  }
  for (const [name, dtype, shapeDims] of inputs) {
    const shape = Buffer.concat(
      shapeDims.map((d) => field(1, typeof d === "number" ? field(1, d) : field(2, Buffer.from(d)))),
    );
    parts.push(
      field(
        11,
        Buffer.concat([
          field(1, Buffer.from(name)),
          field(2, field(1, Buffer.concat([field(1, dtype), field(2, shape)]))),
        ]),
      ),
    );
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, Buffer.concat([field(1, 8), field(7, Buffer.concat(parts))]));
  if (external) {
    const fd = openSync(join(dirname(path), external), "w");
    ftruncateSync(fd, data);
    closeSync(fd);
  }
  return data;
}
