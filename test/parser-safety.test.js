import assert from "node:assert/strict";
import test from "node:test";
import { fetchGgufMetadata, fetchSafetensorsHeader, GGUF_DTYPE_BITS, parseGguf, parseSafetensorsHeaders } from "../dist/index.js";
import { fetchJson, fetchRange } from "../dist/http.js";

class Writer {
  chunks = [];
  u32(value) { const bytes = Buffer.alloc(4); bytes.writeUInt32LE(value); this.chunks.push(bytes); }
  u64(value) { const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(BigInt(value)); this.chunks.push(bytes); }
  string(value) { const bytes = Buffer.from(value); this.u64(bytes.length); this.chunks.push(bytes); }
  finish() { return Buffer.concat(this.chunks); }
}

function gguf(tensors = [], entries = []) {
  const writer = new Writer();
  writer.chunks.push(Buffer.from("GGUF")); writer.u32(3); writer.u64(tensors.length); writer.u64(entries.length);
  for (const [key, type, writeValue] of entries) { writer.string(key); writer.u32(type); writeValue(writer); }
  for (const [index, tensor] of tensors.entries()) {
    writer.string(`weight.${index}`); writer.u32(tensor.shape.length);
    for (const dimension of tensor.shape) writer.u64(dimension);
    writer.u32(tensor.type); writer.u64(tensor.offset ?? 0);
  }
  return writer.finish();
}

function ranged(file, init) {
  const match = new Headers(init.headers).get("range").match(/^bytes=(\d+)-(\d+)$/);
  const start = Number(match[1]);
  const end = Math.min(Number(match[2]), file.length - 1);
  return new Response(file.subarray(start, end + 1), {
    status: 206, headers: { "Content-Range": `bytes ${start}-${end}/${file.length}` },
  });
}

function safetensors(header) {
  const json = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(json.length));
  return Buffer.concat([prefix, json]);
}

// Independent byte counts from GGML's block struct sizes, including scales and sums.
for (const [type, dtype, elements, expectedBytes] of [
  [0, "F32", 1, 4], [1, "F16", 1, 2], [2, "Q4_0", 32, 18], [3, "Q4_1", 32, 20],
  [6, "Q5_0", 32, 22], [7, "Q5_1", 32, 24], [8, "Q8_0", 32, 34], [9, "Q8_1", 32, 36],
  [10, "Q2_K", 256, 84], [11, "Q3_K", 256, 110], [12, "Q4_K", 256, 144],
  [13, "Q5_K", 256, 176], [14, "Q6_K", 256, 210], [15, "Q8_K", 256, 292],
  [16, "IQ2_XXS", 256, 66], [17, "IQ2_XS", 256, 74], [18, "IQ3_XXS", 256, 98],
  [19, "IQ1_S", 256, 50], [20, "IQ4_NL", 32, 18], [21, "IQ3_S", 256, 110],
  [22, "IQ2_S", 256, 82], [23, "IQ4_XS", 256, 136], [24, "I8", 1, 1],
  [25, "I16", 1, 2], [26, "I32", 1, 4], [27, "I64", 1, 8], [28, "F64", 1, 8],
  [29, "IQ1_M", 256, 56], [30, "BF16", 1, 2], [34, "TQ1_0", 256, 54],
  [35, "TQ2_0", 256, 66], [39, "MXFP4", 32, 17], [40, "NVFP4", 64, 36],
  [41, "Q1_0", 128, 18], [42, "Q2_0", 64, 18],
]) {
  test(`GGUF ${dtype} includes exact block storage`, () => {
    const parsed = parseGguf(gguf([{ type, shape: [elements] }]));
    assert.equal(parsed.parameters, elements);
    assert.equal(parsed.bytes, expectedBytes);
    assert.equal(parsed.components.Transformer.dtypes[dtype].bytes, expectedBytes);
  });
}

test("GGUF exported precision retains exact fractional bits", () => {
  assert.equal(GGUF_DTYPE_BITS.Q8_K, 9.125);
  assert.equal(GGUF_DTYPE_BITS.IQ2_S, 2.5625);
  assert.equal(GGUF_DTYPE_BITS.IQ2_XXS, 2.0625);
});

test("GGUF validates row alignment, dimensions, and safe totals", () => {
  for (const tensor of [
    { type: 15, shape: [128, 2] }, // total is block-aligned but rows are not
    { type: 0, shape: [] }, { type: 0, shape: [0] }, { type: 0, shape: [1, 1, 1, 1, 1] },
    { type: 999, shape: [1] }, { type: 0, shape: [Number.MAX_SAFE_INTEGER, 2] },
    { type: 28, shape: [Number.MAX_SAFE_INTEGER] }, { type: 0, shape: [1], offset: Number.MAX_SAFE_INTEGER },
  ]) assert.throws(() => parseGguf(gguf([tensor])));
  assert.throws(() => parseGguf(gguf([{ type: 24, shape: [2 ** 52] }, { type: 24, shape: [2 ** 52] }])), RangeError);
});

test("GGUF metadata keys do not alter prototypes", () => {
  const parsed = parseGguf(gguf([], [["__proto__", 8, (writer) => writer.string("preserved")]]));
  assert.equal(parsed.metadata.__proto__, "preserved");
  assert.equal(Object.hasOwn(parsed.metadata, "__proto__"), true);
  assert.throws(() => parseGguf(gguf([], [
    ["key", 4, (writer) => writer.u32(1)], ["key", 4, (writer) => writer.u32(2)],
  ])));
});

test("GGUF rejects oversized and excessively nested metadata arrays", () => {
  assert.throws(() => parseGguf(gguf([], [["array", 9, (writer) => {
    writer.u32(0); writer.u64(2 ** 32);
  }]])), RangeError);
  assert.throws(() => parseGguf(gguf([], [["array", 9, (writer) => {
    for (let i = 0; i < 17; i++) { writer.u32(9); writer.u64(1); }
    writer.u32(0); writer.u64(0);
  }]])), RangeError);
});

test("GGUF respects small caps and legal EOF-shortened ranges", async () => {
  const file = gguf([{ type: 15, shape: [256] }]);
  const requests = [];
  const fetcher = async (_url, init) => { requests.push(new Headers(init.headers).get("range")); return ranged(file, init); };
  assert.equal((await fetchGgufMetadata(fetcher, "https://example.test/model", {}, 256)).bytes, 292);
  assert.deepEqual(requests, ["bytes=0-255"]);
  for (const cap of [0, -1, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1]) {
    await assert.rejects(fetchGgufMetadata(fetcher, "https://example.test/model", {}, cap), RangeError);
  }
  assert.equal(requests.length, 1);
});

test("GGUF retries actual truncation within the budget", async () => {
  const file = gguf([], [["description", 8, (writer) => writer.string("x".repeat(1_100_000))]]);
  const requests = [];
  const parsed = await fetchGgufMetadata(async (_url, init) => {
    requests.push(new Headers(init.headers).get("range")); return ranged(file, init);
  }, "https://example.test/model", {}, 1_500_000);
  assert.equal(parsed.metadata.description.length, 1_100_000);
  assert.deepEqual(requests, ["bytes=0-999999", "bytes=0-1499999"]);
});

test("GGUF does not retry permanent unsafe integers or truncation at EOF", async () => {
  const unsafe = gguf(); unsafe.writeBigUInt64LE(2n ** 53n, 8);
  const truncated = Buffer.alloc(1_000_000);
  const prefix = gguf([], [["description", 8, (writer) => writer.u64(1_100_000)]]);
  prefix.copy(truncated);
  for (const file of [unsafe, truncated]) {
    let requests = 0;
    await assert.rejects(fetchGgufMetadata(async (_url, init) => { requests++; return ranged(file, init); }, "https://example.test/model"), RangeError);
    assert.equal(requests, 1);
  }
});

test("Safetensors rejects inherited dtype names without prototype mutation", () => {
  const before = Object.getOwnPropertyDescriptors(Object.prototype);
  for (const dtype of ["__proto__", "constructor", "toString"]) {
    assert.throws(() => parseSafetensorsHeaders({ Transformer: { weight: { dtype, shape: [1] } } }));
  }
  assert.deepEqual(Object.getOwnPropertyDescriptors(Object.prototype), before);
  const result = parseSafetensorsHeaders(JSON.parse('{"__proto__":{"weight":{"dtype":"F16","shape":[2]}}}'));
  assert.equal(result.components.__proto__.bytes, 4);
});

test("Safetensors distinguishes scalars, empties, and malformed tensors", () => {
  const result = parseSafetensorsHeaders({ Transformer: {
    scalar: { dtype: "F32", shape: [], data_offsets: [0, 4] },
    empty: { dtype: "F64", shape: [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, 0], data_offsets: [4, 4] },
  } });
  assert.equal(result.parameters, 1);
  assert.equal(result.bytes, 4);
  for (const header of [null, [], "bad", { weight: null }, { weight: [] }, { __metadata__: { format: 1 } },
    { weight: { dtype: "F16", shape: [-1] } }, { weight: { dtype: "F16", shape: [1.5] } },
    { weight: { dtype: "F16", shape: [Number.MAX_SAFE_INTEGER] } },
    { weight: { dtype: "U8", shape: [Number.MAX_SAFE_INTEGER, 2] } },
  ]) assert.throws(() => parseSafetensorsHeaders({ Transformer: header }));
  assert.throws(() => parseSafetensorsHeaders({ A: { a: { dtype: "U8", shape: [2 ** 52] } }, B: { b: { dtype: "U8", shape: [2 ** 52] } } }), RangeError);
  assert.throws(() => parseSafetensorsHeaders({ Transformer: { w: { dtype: "F16", shape: new Array(1) } } }));
});

test("Safetensors validates offsets by dtype, length, and interval", () => {
  for (const data_offsets of [[0, 1], [2, 0], [-1, 1], [0, 2.5], [0, 2, 3], null, [0, Number.MAX_SAFE_INTEGER + 1]]) {
    assert.throws(() => parseSafetensorsHeaders({ Transformer: { w: { dtype: "F16", shape: [1], data_offsets } } }));
  }
  assert.throws(() => parseSafetensorsHeaders({ Transformer: {
    a: { dtype: "F16", shape: [2], data_offsets: [0, 4] },
    b: { dtype: "F16", shape: [1], data_offsets: [2, 4] },
  } }));
  const adjacent = parseSafetensorsHeaders({ Transformer: {
    b: { dtype: "F16", shape: [1], data_offsets: [4, 6] },
    empty: { dtype: "F16", shape: [0], data_offsets: [2, 2] },
    a: { dtype: "F32", shape: [1], data_offsets: [0, 4] },
  } });
  assert.equal(adjacent.bytes, 6);
});

test("Safetensors validates fetched headers and preserves every HeadersInit form", async () => {
  for (const headers of [{ Authorization: "Bearer token" }, [["Authorization", "Bearer token"]], new Headers({ Authorization: "Bearer token" })]) {
    const file = safetensors({ scalar: { dtype: "F16", shape: [] } });
    const header = await fetchSafetensorsHeader(async (_url, init) => {
      assert.equal(new Headers(init.headers).get("authorization"), "Bearer token"); return ranged(file, init);
    }, "https://example.test/model", headers);
    assert.deepEqual(header.scalar.shape, []);
  }
  for (const value of [null, [], { w: { dtype: "__proto__", shape: [1] } }]) {
    const file = safetensors(value);
    await assert.rejects(fetchSafetensorsHeader(async (_url, init) => ranged(file, init), "https://example.test/model"));
  }
});

test("HTTP rejects ignored or dishonest ranges before consuming their bodies", async () => {
  for (const [status, contentRange] of [[200, null], [206, null], [206, "bytes 1-8/100"],
    [206, "bytes 0-8/100"], [206, "bytes 0-3/100"], [206, "bytes 0-7/7"], [206, "bytes 0-3/*"]]) {
    let cancelled = false;
    let reads = 0;
    const body = new ReadableStream({ pull() { reads++; }, cancel() { cancelled = true; } }, { highWaterMark: 0 });
    await assert.rejects(fetchRange(async () => new Response(body, {
      status, headers: contentRange ? { "Content-Range": contentRange } : {},
    }), "https://example.test/model", 0, 7, {}));
    assert.equal(cancelled, true);
    assert.equal(reads, 0);
  }
});

test("HTTP cancels oversized streamed ranges without draining", async () => {
  let reads = 0;
  let cancelled = false;
  const body = new ReadableStream({
    pull(controller) { reads++; controller.enqueue(new Uint8Array(5)); },
    cancel() { cancelled = true; },
  }, { highWaterMark: 0 });
  await assert.rejects(fetchRange(async () => new Response(body, {
    status: 206, headers: { "Content-Range": "bytes 0-7/100" },
  }), "https://example.test/model", 0, 7, {}), RangeError);
  assert.equal(reads, 2);
  assert.equal(cancelled, true);
});

test("HTTP rejects mismatched range body sizes and oversized JSON metadata", async () => {
  await assert.rejects(fetchRange(async () => new Response(new Uint8Array(7), {
    status: 206, headers: { "Content-Range": "bytes 0-7/100" },
  }), "https://example.test/model", 0, 7, {}));
  await assert.rejects(fetchRange(async () => new Response(new Uint8Array(8), {
    status: 206, headers: { "Content-Range": "bytes 0-7/100", "Content-Length": "9" },
  }), "https://example.test/model", 0, 7, {}));
  let cancelled = false;
  let reads = 0;
  await assert.rejects(fetchJson(async () => new Response(new ReadableStream({
    pull() { reads++; }, cancel() { cancelled = true; },
  }, { highWaterMark: 0 }), { headers: { "Content-Length": "33554433" } }), "https://example.test/config", {}));
  assert.equal(cancelled, true);
  assert.equal(reads, 0);
});
