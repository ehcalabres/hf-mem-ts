import assert from "node:assert/strict";
import test from "node:test";
import { estimateModelMemory } from "../dist/index.js";

const SHA = "a".repeat(40);
const TP_KEYS = ["1", "2", "4", "8"];
const config = {
  num_hidden_layers: 2, num_attention_heads: 8, num_key_value_heads: 1,
  head_dim: 4, hidden_size: 32, max_position_embeddings: 16, dtype: "float16",
};
// 2 layers * 16 tokens * 2 K/V * 1 head * 4 elements * 2 bytes, replicated per rank.
const CACHE = { "1": 512, "2": 1024, "4": 2048, "8": 4096 };

function safetensors() {
  const header = Buffer.from(JSON.stringify({ weight: { dtype: "F16", shape: [2], data_offsets: [0, 4] } }));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(header.length));
  return Buffer.concat([prefix, header, Buffer.alloc(4)]);
}

function gguf({ layers = 2, kvHeads = 1, elements = 4, cache = true } = {}) {
  const chunks = [];
  const u32 = (value) => { const bytes = Buffer.alloc(4); bytes.writeUInt32LE(value); chunks.push(bytes); };
  const u64 = (value) => { const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(BigInt(value)); chunks.push(bytes); };
  const string = (value) => { const bytes = Buffer.from(value); u64(bytes.length); chunks.push(bytes); };
  const entries = cache ? {
    "llama.block_count": layers, "llama.attention.head_count_kv": kvHeads,
    "llama.attention.head_count": 8, "llama.embedding_length": 32, "llama.context_length": 16,
  } : {};
  chunks.push(Buffer.from("GGUF")); u32(3); u64(1); u64(Object.keys(entries).length);
  for (const [key, value] of Object.entries(entries)) { string(key); u32(4); u32(value); }
  string("blk.0.weight"); u32(1); u64(elements); u32(1); u64(0);
  return Buffer.concat(chunks);
}

function safetensorsRepository(cacheConfig = config) {
  return { "model.safetensors": safetensors(), "config.json": cacheConfig };
}

// One response per distinct metadata request. A repeated topology fetch fails the
// fixture instead of quietly returning the same metadata four times.
function repositories(models) {
  const requests = [];
  const seen = new Set();
  const fetcher = async (input, init) => {
    const url = new URL(input);
    const range = new Headers(init?.headers).get("range");
    const request = `${url.pathname} ${range ?? ""}`;
    assert.equal(seen.has(request), false, `duplicate metadata request: ${request}`);
    seen.add(request); requests.push({ path: url.pathname, range });
    const modelId = url.pathname.startsWith("/api/models/")
      ? url.pathname.split("/").slice(3, 5).join("/")
      : url.pathname.split("/").slice(1, 3).join("/");
    const files = models[modelId];
    assert.ok(files, `unexpected repository: ${modelId}`);
    if (url.pathname.includes("/revision/")) return Response.json({ sha: SHA });
    if (url.pathname.includes("/tree/")) return Response.json(Object.keys(files).map(path => ({ type: "file", path })));
    const filename = decodeURIComponent(url.pathname.split(`/resolve/${SHA}/`)[1]);
    assert.ok(Object.hasOwn(files, filename), `unexpected file: ${filename}`);
    if (filename.endsWith(".json")) return Response.json(files[filename]);
    assert.match(range, /^bytes=\d+-\d+$/);
    const [, startText, endText] = range.match(/^bytes=(\d+)-(\d+)$/);
    const bytes = files[filename];
    const start = Number(startText); const end = Math.min(Number(endText), bytes.length - 1);
    return new Response(bytes.subarray(start, end + 1), {
      status: 206, headers: { "Content-Range": `bytes ${start}-${end}/${bytes.length}` },
    });
  };
  return { fetch: fetcher, requests };
}

function assertFileComparison(file, expected) {
  assert.equal(file.kvCache, null);
  assert.deepEqual(Object.keys(file.kvCacheByTp), TP_KEYS);
  for (const tp of TP_KEYS) {
    const cache = file.kvCacheByTp[tp];
    assert.equal(cache.bytes, expected[tp]);
    assert.equal(cache.attentionBytes, expected[tp]);
    assert.equal(cache.stateBytes, 0);
    assert.equal(cache.tensorParallelSize, Number(tp));
    assert.equal(cache.maxModelLen, 16);
    assert.equal(cache.dtype, "F16");
  }
}

function assertSingle(result, filename, tp, bytes, total) {
  assert.equal(result.kvCacheBytes, bytes);
  assert.equal(result.totalBytes, total);
  assert.equal(result.kvCacheBytesByTp, null);
  assert.equal(result.totalBytesByTp, null);
  assert.equal(result.files[filename].kvCache.bytes, bytes);
  assert.equal(result.files[filename].kvCache.tensorParallelSize, tp);
  assert.equal(result.files[filename].kvCacheByTp, null);
}

test("default Safetensors comparison fetches one config and two header ranges for all four TP choices", async () => {
  const files = safetensorsRepository();
  const fixture = repositories({ "org/model": files });
  const result = await estimateModelMemory({ modelId: "org/model", kvCache: true, fetch: fixture.fetch });
  assert.equal(result.weightsBytes, 4);
  assert.equal(result.kvCacheBytes, null);
  assert.equal(result.totalBytes, null);
  assert.deepEqual(result.kvCacheBytesByTp, CACHE);
  assert.deepEqual(result.totalBytesByTp, { "1": 516, "2": 1028, "4": 2052, "8": 4100 });
  assertFileComparison(result.files.safetensors, CACHE);
  assert.equal(fixture.requests.length, 5);
  assert.deepEqual(fixture.requests.filter(request => request.range).map(request => request.range), [
    "bytes=0-7", `bytes=8-${files["model.safetensors"].length - 5}`,
  ]);
  assert.equal(fixture.requests.filter(request => request.path.endsWith("config.json")).length, 1);
});

test("explicit TP outside the comparison choices remains one Safetensors estimate", async () => {
  const fixture = repositories({ "org/model": safetensorsRepository() });
  const result = await estimateModelMemory({ modelId: "org/model", kvCache: true, tensorParallelSize: 3, fetch: fixture.fetch });
  assertSingle(result, "safetensors", 3, 1536, 1540);
});

test("disabled cache preserves weight totals and never reads even an unusable config", async () => {
  const fixture = repositories({ "org/model": safetensorsRepository(null) });
  const result = await estimateModelMemory({ modelId: "org/model", kvCache: false, fetch: fixture.fetch });
  assert.equal(result.weightsBytes, 4);
  assert.equal(result.totalBytes, 4);
  assert.equal(result.kvCacheBytes, null);
  assert.equal(result.kvCacheBytesByTp, null);
  assert.equal(result.totalBytesByTp, null);
  assert.equal(result.files.safetensors.kvCache, null);
  assert.equal(result.files.safetensors.kvCacheByTp, null);
  assert.equal(fixture.requests.some(request => request.path.endsWith("config.json")), false);
  assert.equal(fixture.requests.length, 4);
});

test("zero-valued cache alternatives retain every TP key", async () => {
  const fixture = repositories({ "org/model": safetensorsRepository({ ...config, layer_types: ["mlp", "feed_forward"] }) });
  const result = await estimateModelMemory({ modelId: "org/model", kvCache: true, fetch: fixture.fetch });
  assert.deepEqual(result.kvCacheBytesByTp, { "1": 0, "2": 0, "4": 0, "8": 0 });
  assert.deepEqual(result.totalBytesByTp, { "1": 4, "2": 4, "4": 4, "8": 4 });
  assertFileComparison(result.files.safetensors, { "1": 0, "2": 0, "4": 0, "8": 0 });
});

test("selected GGUF compares caches from a single metadata range", async () => {
  const fixture = repositories({ "org/model": { "model.gguf": gguf() } });
  const result = await estimateModelMemory({ modelId: "org/model", ggufFile: "model.gguf", kvCache: true, fetch: fixture.fetch });
  assert.equal(result.weightsBytes, 8);
  assert.equal(result.kvCacheBytes, null);
  assert.equal(result.totalBytes, null);
  assert.deepEqual(result.kvCacheBytesByTp, CACHE);
  assert.deepEqual(result.totalBytesByTp, { "1": 520, "2": 1032, "4": 2056, "8": 4104 });
  assertFileComparison(result.files["model.gguf"], CACHE);
  assert.equal(fixture.requests.length, 3);
  assert.equal(fixture.requests.filter(request => request.range).length, 1);
});

test("explicit TP1 retains the selected GGUF single-result contract", async () => {
  const fixture = repositories({ "org/model": { "model.gguf": gguf() } });
  const result = await estimateModelMemory({ modelId: "org/model", ggufFile: "model.gguf", kvCache: true, tensorParallelSize: 1, fetch: fixture.fetch });
  assertSingle(result, "model.gguf", 1, 512, 520);
});

test("padded GGUF shards share one cache per TP while projector weights are additive", async () => {
  const fixture = repositories({ "org/model": {
    "model-00002-of-00002.gguf": gguf({ cache: false }),
    "model-00001-of-00002.gguf": gguf(),
    "mmproj_f16.gguf": gguf({ cache: false, elements: 3 }),
  } });
  const result = await estimateModelMemory({ modelId: "org/model", ggufFile: "model-00002-of-00002.gguf", kvCache: true, fetch: fixture.fetch });
  assert.equal(result.filename, "model.gguf");
  assert.equal(result.weightsBytes, 16);
  assert.equal(result.mmproj.bytes, 6);
  assert.deepEqual(result.kvCacheBytesByTp, CACHE);
  assert.deepEqual(result.totalBytesByTp, { "1": 534, "2": 1046, "4": 2070, "8": 4118 });
  assertFileComparison(result.files["model.gguf"], CACHE);
  assert.equal(fixture.requests.length, 5);
  assert.equal(fixture.requests.filter(request => request.range).length, 3);
});

test("unselected GGUF keeps a TP by filename matrix without summing quantization alternatives", async () => {
  const fixture = repositories({ "org/model": {
    "model-F16.gguf": gguf(), "model-F32.gguf": gguf({ layers: 1, kvHeads: 2, elements: 8 }),
  } });
  const result = await estimateModelMemory({ modelId: "org/model", kvCache: true, fetch: fixture.fetch });
  assert.deepEqual(result.weightsBytes, { "model-F16.gguf": 8, "model-F32.gguf": 16 });
  assert.equal(result.kvCacheBytes, null);
  assert.equal(result.totalBytes, null);
  assert.deepEqual(result.kvCacheBytesByTp, {
    "1": { "model-F16.gguf": 512, "model-F32.gguf": 512 },
    "2": { "model-F16.gguf": 1024, "model-F32.gguf": 512 },
    "4": { "model-F16.gguf": 2048, "model-F32.gguf": 1024 },
    "8": { "model-F16.gguf": 4096, "model-F32.gguf": 2048 },
  });
  assert.deepEqual(result.totalBytesByTp, { "1": null, "2": null, "4": null, "8": null });
  assertFileComparison(result.files["model-F16.gguf"], CACHE);
  assertFileComparison(result.files["model-F32.gguf"], { "1": 512, "2": 512, "4": 1024, "8": 2048 });
  assert.equal(fixture.requests.length, 4);
});

test("explicit TP keeps unselected GGUF filename cache records rather than a TP matrix", async () => {
  const fixture = repositories({ "org/model": {
    "small.gguf": gguf(), "large.gguf": gguf({ layers: 1, kvHeads: 2, elements: 8 }),
  } });
  const result = await estimateModelMemory({ modelId: "org/model", kvCache: true, tensorParallelSize: 4, fetch: fixture.fetch });
  assert.deepEqual(result.weightsBytes, { "small.gguf": 8, "large.gguf": 16 });
  assert.deepEqual(result.kvCacheBytes, { "small.gguf": 2048, "large.gguf": 1024 });
  assert.equal(result.totalBytes, null);
  assert.equal(result.kvCacheBytesByTp, null);
  assert.equal(result.totalBytesByTp, null);
  assert.equal(result.files["small.gguf"].kvCache.bytes, 2048);
  assert.equal(result.files["large.gguf"].kvCache.bytes, 1024);
  assert.equal(result.files["small.gguf"].kvCacheByTp, null);
  assert.equal(result.files["large.gguf"].kvCacheByTp, null);
});

test("an unresolved GGUF draft prevents combined totals in every target TP scenario", async () => {
  const fixture = repositories({
    "org/model": safetensorsRepository(),
    "org/draft": { "small.gguf": gguf(), "large.gguf": gguf({ layers: 1, kvHeads: 2, elements: 8 }) },
  });
  const result = await estimateModelMemory({ modelId: "org/model", draftModel: "org/draft", kvCache: true, fetch: fixture.fetch });
  assert.deepEqual(result.kvCacheBytesByTp, CACHE);
  assert.equal(result.totalBytes, null);
  assert.deepEqual(result.totalBytesByTp, { "1": null, "2": null, "4": null, "8": null });
  assert.deepEqual(result.draft.kvCacheBytesByTp, {
    "1": { "small.gguf": 512, "large.gguf": 512 },
    "2": { "small.gguf": 1024, "large.gguf": 512 },
    "4": { "small.gguf": 2048, "large.gguf": 1024 },
    "8": { "small.gguf": 4096, "large.gguf": 2048 },
  });
  assert.deepEqual(result.draft.totalBytesByTp, { "1": null, "2": null, "4": null, "8": null });
  assert.equal(fixture.requests.length, 9);
});

function targetAndDraft() {
  return repositories({
    "org/target": { "model.gguf": gguf(), "mmproj_f16.gguf": gguf({ cache: false, elements: 3 }) },
    "org/draft": safetensorsRepository({ ...config, num_hidden_layers: 1, num_key_value_heads: 2 }),
  });
}

test("target, projector and a distinct draft are paired by TP rather than summing scenarios", async () => {
  const fixture = targetAndDraft();
  const result = await estimateModelMemory({ modelId: "org/target", ggufFile: "model.gguf", draftModel: "org/draft", kvCache: true, fetch: fixture.fetch });
  assert.deepEqual(result.kvCacheBytesByTp, CACHE);
  assert.deepEqual(result.draft.kvCacheBytesByTp, { "1": 512, "2": 512, "4": 1024, "8": 2048 });
  assert.deepEqual(result.draft.totalBytesByTp, { "1": 516, "2": 516, "4": 1028, "8": 2052 });
  assert.deepEqual(result.totalBytesByTp, { "1": 1042, "2": 1554, "4": 3090, "8": 6162 });
  assert.equal(result.totalBytes, null);
  assert.equal(result.draft.totalBytes, null);
  assert.equal(result.mmproj.bytes, 6);
  assert.equal(result.draft.mmproj, null);
  assert.equal(fixture.requests.length, 9);
});

test("explicit draft TP remains fixed across every target comparison scenario", async () => {
  const fixture = targetAndDraft();
  const result = await estimateModelMemory({
    modelId: "org/target", ggufFile: "model.gguf", kvCache: true,
    draftModel: { modelId: "org/draft", tensorParallelSize: 3 }, fetch: fixture.fetch,
  });
  assertSingle(result.draft, "safetensors", 3, 768, 772);
  assert.deepEqual(result.kvCacheBytesByTp, CACHE);
  assert.deepEqual(result.totalBytesByTp, { "1": 1298, "2": 1810, "4": 2834, "8": 4882 });
  assert.equal(fixture.requests.length, 9);
});

test("same-model default draft comparison reuses metadata for all paired scenarios", async () => {
  const fixture = repositories({ "org/model": safetensorsRepository() });
  const result = await estimateModelMemory({ modelId: "org/model", draftModel: "org/model", kvCache: true, fetch: fixture.fetch });
  assert.deepEqual(result.draft.kvCacheBytesByTp, CACHE);
  assert.deepEqual(result.totalBytesByTp, { "1": 1032, "2": 2056, "4": 4104, "8": 8200 });
  assert.deepEqual(result.draft.totalBytesByTp, { "1": 516, "2": 1028, "4": 2052, "8": 4100 });
  assert.equal(fixture.requests.length, 5);
});

test("same-model fixed draft TP projects an existing comparison without metadata refetch", async () => {
  const fixture = repositories({ "org/model": safetensorsRepository() });
  const result = await estimateModelMemory({
    modelId: "org/model", kvCache: true, draftModel: { modelId: "org/model", tensorParallelSize: 2 }, fetch: fixture.fetch,
  });
  assertSingle(result.draft, "safetensors", 2, 1024, 1028);
  assert.deepEqual(result.totalBytesByTp, { "1": 1544, "2": 2056, "4": 3080, "8": 5128 });
  assert.equal(fixture.requests.length, 5);
});

test("invalid target or draft TP is rejected before any metadata fetch", async () => {
  let requests = 0;
  const fetcher = async () => { requests++; throw new Error("must not fetch"); };
  await assert.rejects(estimateModelMemory({ modelId: "org/model", kvCache: true, tensorParallelSize: 0, fetch: fetcher }), RangeError);
  await assert.rejects(estimateModelMemory({ modelId: "org/model", kvCache: true, draftModel: { modelId: "org/draft", tensorParallelSize: 0 }, fetch: fetcher }), RangeError);
  assert.equal(requests, 0);
});
