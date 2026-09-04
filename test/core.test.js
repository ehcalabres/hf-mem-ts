import assert from "node:assert/strict";
import test from "node:test";

import {
  estimateGgufKvCache,
  estimateModelMemory,
  estimateSafetensorsKvCache,
  fetchSafetensorsHeader,
  parseGguf,
  parseSafetensorsHeaders,
} from "../dist/index.js";

function safetensorsFile(header) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const output = new Uint8Array(8 + json.length);
  new DataView(output.buffer).setBigUint64(0, BigInt(json.length), true);
  output.set(json, 8);
  return output;
}

class BinaryWriter {
  chunks = [];
  raw(value) { this.chunks.push(Uint8Array.from(value)); }
  u32(value) { const bytes = new Uint8Array(4); new DataView(bytes.buffer).setUint32(0, value, true); this.chunks.push(bytes); }
  u64(value) { const bytes = new Uint8Array(8); new DataView(bytes.buffer).setBigUint64(0, BigInt(value), true); this.chunks.push(bytes); }
  string(value) { const bytes = new TextEncoder().encode(value); this.u64(bytes.length); this.chunks.push(bytes); }
  finish() { const size = this.chunks.reduce((n, chunk) => n + chunk.length, 0); const out = new Uint8Array(size); let offset = 0; for (const chunk of this.chunks) { out.set(chunk, offset); offset += chunk.length; } return out; }
}

function ggufFile() {
  const writer = new BinaryWriter();
  writer.raw(new TextEncoder().encode("GGUF")); writer.u32(3); writer.u64(1); writer.u64(5);
  for (const [key, value] of [
    ["llama.block_count", 32], ["llama.attention.head_count_kv", 8], ["llama.attention.head_count", 32],
    ["llama.embedding_length", 4096], ["llama.context_length", 4096],
  ]) { writer.string(key); writer.u32(4); writer.u32(value); }
  writer.string("blk.0.attn.weight"); writer.u32(2); writer.u64(4096); writer.u64(4096); writer.u32(12); writer.u64(0);
  return writer.finish();
}

test("parses and totals Safetensors metadata", () => {
  const metadata = parseSafetensorsHeaders({ Transformer: {
    weight: { dtype: "BF16", shape: [4, 8], data_offsets: [0, 64] },
    bias: { dtype: "F32", shape: [8], data_offsets: [64, 96] },
    __metadata__: { format: "pt" },
  } });
  assert.equal(metadata.parameters, 40);
  assert.equal(metadata.bytes, 96);
  assert.deepEqual(metadata.components.Transformer.dtypes.BF16, { parameters: 32, bytes: 64 });
});

test("fetches only the two Safetensors metadata ranges", async () => {
  const file = safetensorsFile({ weight: { dtype: "F16", shape: [2, 3], data_offsets: [0, 12] } });
  const ranges = [];
  const fetcher = async (_url, init) => {
    const range = init.headers.Range;
    ranges.push(range);
    const [, start, end] = range.match(/bytes=(\d+)-(\d+)/).map(Number);
    return new Response(file.slice(start, end + 1), { status: 206 });
  };
  const header = await fetchSafetensorsHeader(fetcher, "https://example.test/model.safetensors");
  assert.equal(header.weight.dtype, "F16");
  assert.deepEqual(ranges, ["bytes=0-7", `bytes=8-${file.length - 1}`]);
});

test("computes GQA KV cache and parses GGUF tensor metadata", () => {
  const config = { hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, max_position_embeddings: 4096, torch_dtype: "bfloat16" };
  const cache = estimateSafetensorsKvCache(config);
  assert.equal(cache.bytes, 536_870_912);
  assert.equal(cache.attentionBytes, 536_870_912);
  assert.equal(cache.stateBytes, 0);
  assert.equal(cache.dtype, "BF16");
  assert.equal(cache.layout, "attention");

  const parsed = parseGguf(ggufFile());
  assert.equal(parsed.parameters, 4096 ** 2);
  assert.equal(parsed.bytes, Math.floor(4096 ** 2 * 4.5 / 8));
  assert.equal(estimateGgufKvCache(parsed.metadata).bytes, 536_870_912);
});

test("estimates a Hub Safetensors model through an injected fetch", async () => {
  const file = safetensorsFile({ weight: { dtype: "F16", shape: [10, 10], data_offsets: [0, 200] } });
  const fetcher = async (input, init = {}) => {
    const url = String(input);
    if (url.includes("/tree/")) return Response.json([
      { type: "file", path: "model.safetensors" }, { type: "file", path: "config.json" },
    ]);
    if (url.endsWith("config.json")) return Response.json({ hidden_size: 8, num_hidden_layers: 2, num_attention_heads: 2, max_position_embeddings: 16, torch_dtype: "float16" });
    const range = new Headers(init.headers).get("range");
    const match = range.match(/bytes=(\d+)-(\d+)/);
    return new Response(file.slice(Number(match[1]), Number(match[2]) + 1), { status: 206 });
  };
  const result = await estimateModelMemory({ modelId: "org/model", fetch: fetcher, kvCache: true });
  assert.equal(result.format, "safetensors");
  assert.equal(result.weightsBytes, 200);
  assert.equal(result.kvCacheBytes, 1024);
  assert.equal(result.totalBytes, 1224);
});

test("keeps embedded GGUF metadata internal to the parser", async () => {
  const file = ggufFile();
  const fetcher = async (input, init = {}) => {
    if (String(input).includes("/tree/")) return Response.json([{ type: "file", path: "model-Q4_K.gguf" }]);
    assert.match(new Headers(init.headers).get("range"), /^bytes=0-/);
    return new Response(file, { status: 206 });
  };
  const result = await estimateModelMemory({ modelId: "org/gguf", ggufFile: "model-Q4_K.gguf", fetch: fetcher });
  assert.equal(result.weightsBytes, Math.floor(4096 ** 2 * 4.5 / 8));
  assert.equal("metadata" in result.files["model-Q4_K.gguf"], false);
});

test("adds an auto-selected F16 mmproj and a separate draft model to total memory", async () => {
  const file = ggufFile();
  const fetcher = async (input) => {
    const url = String(input);
    if (url.includes("/api/models/org/main/tree/")) return Response.json([
      { type: "file", path: "main-Q4_K.gguf" },
      { type: "file", path: "mmproj-BF16.gguf" },
      { type: "file", path: "mmproj-F16.gguf" },
    ]);
    if (url.includes("/api/models/org/draft/tree/")) return Response.json([
      { type: "file", path: "draft-Q4_K.gguf" },
      { type: "file", path: "mmproj-F16.gguf" },
    ]);
    return new Response(file, { status: 206 });
  };
  const result = await estimateModelMemory({
    modelId: "org/main",
    ggufFile: "main-Q4_K.gguf",
    draftModel: { modelId: "org/draft", ggufFile: "draft-Q4_K.gguf" },
    fetch: fetcher,
  });
  const oneFile = Math.floor(4096 ** 2 * 4.5 / 8);
  assert.equal(result.mmproj.filename, "mmproj-F16.gguf");
  assert.equal(result.mmproj.bytes, oneFile);
  assert.equal(result.draft.weightsBytes, oneFile);
  assert.equal(result.draft.mmproj, null);
  assert.equal(result.totalBytes, oneFile * 3);
});

test("estimates target and draft models concurrently", async () => {
  const file = safetensorsFile({ weight: { dtype: "F16", shape: [2, 2], data_offsets: [0, 8] } });
  let active = 0;
  let maxActive = 0;
  const fetcher = async (input, init = {}) => {
    active++;
    maxActive = Math.max(maxActive, active);
    await new Promise((resolve) => setTimeout(resolve, 5));
    active--;
    const url = String(input);
    if (url.includes("/tree/")) return Response.json([{ type: "file", path: "model.safetensors" }]);
    const match = new Headers(init.headers).get("range").match(/bytes=(\d+)-(\d+)/);
    return new Response(file.slice(Number(match[1]), Number(match[2]) + 1), { status: 206 });
  };

  const result = await estimateModelMemory({
    modelId: "org/target",
    draftModel: "org/draft",
    concurrency: 1,
    fetch: fetcher,
  });

  assert.equal(result.totalBytes, 16);
  assert.ok(maxActive >= 2, `expected overlapping target and draft requests, saw ${maxActive}`);
});

test("fetches multiple Safetensors indexes concurrently", async () => {
  const file = safetensorsFile({ weight: { dtype: "F16", shape: [2, 2], data_offsets: [0, 8] } });
  let activeIndexes = 0;
  let maxActiveIndexes = 0;
  const fetcher = async (input, init = {}) => {
    const url = String(input);
    if (url.includes("/tree/")) return Response.json([
      { type: "file", path: "encoder/model.safetensors.index.json" },
      { type: "file", path: "decoder/model.safetensors.index.json" },
    ]);
    if (url.endsWith("model.safetensors.index.json")) {
      activeIndexes++;
      maxActiveIndexes = Math.max(maxActiveIndexes, activeIndexes);
      await new Promise((resolve) => setTimeout(resolve, 5));
      activeIndexes--;
      return Response.json({ weight_map: { weight: "weights.safetensors" } });
    }
    const match = new Headers(init.headers).get("range").match(/bytes=(\d+)-(\d+)/);
    return new Response(file.slice(Number(match[1]), Number(match[2]) + 1), { status: 206 });
  };

  const result = await estimateModelMemory({ modelId: "org/model", concurrency: 2, fetch: fetcher });
  assert.equal(result.weightsBytes, 16);
  assert.equal(maxActiveIndexes, 2);
});

test("discovers mixed sharded and unsharded Diffusers components from model_index.json", async () => {
  const binaries = {
    "text_encoder/model-00001-of-00002.safetensors": safetensorsFile({
      encoder_a: { dtype: "F16", shape: [10], data_offsets: [0, 20] },
    }),
    "text_encoder/model-00002-of-00002.safetensors": safetensorsFile({
      encoder_b: { dtype: "F16", shape: [20], data_offsets: [0, 40] },
    }),
    "transformer/diffusion_pytorch_model.safetensors": safetensorsFile({
      transformer: { dtype: "BF16", shape: [30], data_offsets: [0, 60] },
    }),
    "vae/diffusion_pytorch_model.safetensors": safetensorsFile({
      vae: { dtype: "F16", shape: [40], data_offsets: [0, 80] },
    }),
    "unused/model.safetensors": safetensorsFile({
      duplicate: { dtype: "F16", shape: [100], data_offsets: [0, 200] },
    }),
  };
  const fetcher = async (input, init = {}) => {
    const url = String(input);
    if (url.includes("/tree/")) return Response.json([
      { type: "file", path: "model_index.json" },
      { type: "file", path: "text_encoder/model.safetensors.index.json" },
      { type: "file", path: "text_encoder/model-00001-of-00002.safetensors" },
      { type: "file", path: "text_encoder/model-00002-of-00002.safetensors" },
      { type: "file", path: "transformer/diffusion_pytorch_model.safetensors" },
      { type: "file", path: "vae/diffusion_pytorch_model.safetensors" },
      { type: "file", path: "unused/model.safetensors" },
    ]);
    if (url.endsWith("model_index.json")) return Response.json({
      _class_name: "DiffusionPipeline",
      text_encoder: ["transformers", "Encoder"],
      transformer: ["diffusers", "Transformer"],
      vae: ["diffusers", "Autoencoder"],
      tokenizer: ["transformers", "Tokenizer"],
    });
    if (url.endsWith("text_encoder/model.safetensors.index.json")) return Response.json({
      weight_map: {
        encoder_a: "model-00001-of-00002.safetensors",
        encoder_b: "model-00002-of-00002.safetensors",
      },
    });
    const path = Object.keys(binaries).find((candidate) => url.endsWith(candidate));
    assert.ok(path, `unexpected URL: ${url}`);
    const range = new Headers(init.headers).get("range");
    const match = range.match(/bytes=(\d+)-(\d+)/);
    return new Response(binaries[path].slice(Number(match[1]), Number(match[2]) + 1), { status: 206 });
  };

  const result = await estimateModelMemory({ modelId: "org/diffusers", fetch: fetcher });

  assert.equal(result.weightsBytes, 200);
  assert.equal(result.totalBytes, 200);
  assert.deepEqual(Object.fromEntries(
    Object.entries(result.files.safetensors.components).map(([name, component]) => [name, component.bytes]),
  ), { text_encoder: 60, transformer: 60, vae: 80 });
});

test("reuses metadata when target and draft select the same model file", async () => {
  const file = ggufFile();
  let treeRequests = 0;
  let rangeRequests = 0;
  const fetcher = async (input) => {
    if (String(input).includes("/tree/")) {
      treeRequests++;
      return Response.json([{ type: "file", path: "model-Q4_K.gguf" }]);
    }
    rangeRequests++;
    return new Response(file, { status: 206 });
  };

  const result = await estimateModelMemory({
    modelId: "org/model",
    ggufFile: "model-Q4_K.gguf",
    draftModel: { modelId: "org/model", ggufFile: "model-Q4_K.gguf" },
    fetch: fetcher,
  });

  assert.equal(result.totalBytes, Math.floor(4096 ** 2 * 4.5 / 8) * 2);
  assert.equal(treeRequests, 1);
  assert.equal(rangeRequests, 1);
});
