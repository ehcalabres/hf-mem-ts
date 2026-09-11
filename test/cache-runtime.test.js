import assert from "node:assert/strict";
import test from "node:test";
import { estimateModelMemory, estimateSafetensorsKvCache, estimateGgufKvCache } from "../dist/index.js";

// Cache metadata from Qwen/Qwen3.8-Flash-Next. Expected payloads are derived
// from the vLLM 0.29 QSA cache shapes, NOT fitted to occupied-pool utilization.
const flashNext = {
  dtype: "bfloat16", hidden_size: 2560, num_hidden_layers: 48,
  num_attention_heads: 24, num_key_value_heads: 2, head_dim: 256,
  max_position_embeddings: 262144, full_attention_interval: 4,
  layer_types: Array.from({ length: 48 }, (_, index) => (index + 1) % 4 ? "linear_attention" : "full_attention"),
  linear_num_key_heads: 16, linear_num_value_heads: 48,
  linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
  mamba_ssm_dtype: "float32",
  indexer_head_dim: 128, indexer_kv_heads: 1, indexer_n_heads: 4, indexer_compress_ratio: 4,
};

test("long hybrid workload includes compressed index keys without equating payload to pool occupancy", () => {
  const cache = estimateSafetensorsKvCache(flashNext, { maxModelLen: 261956 });
  // Main K/V: 12 * 261956 * 2 K/V * 2 heads * 256 elements * 2 B.
  assert.equal(cache.attentionBytes - cache.indexerBytes, 6437830656);
  // 12 key-only caches: (65489 compressed rows + 4 raw rows) * 128 * 2 B.
  assert.equal(cache.indexerBytes, 201194496);
  // Full-kernel generic GDN history + one FP32 recurrent snapshot per sequence.
  assert.equal(cache.stateBytes, 116195328);
  assert.equal(cache.bytes, 6755220480);
  assert.equal(cache.approximate, true);
  assert.match(cache.assumptions.join(" "), /one state per sequence/);
  assert.match(cache.assumptions.join(" "), /pool blocks/);
});

test("tensor parallelism accounts for KV replication without multiplying sharded state", () => {
  const twoRanks = estimateSafetensorsKvCache(flashNext, { maxModelLen: 261956, tensorParallelSize: 2 });
  const fourRanks = estimateSafetensorsKvCache(flashNext, { maxModelLen: 261956, tensorParallelSize: 4 });
  assert.equal(twoRanks.attentionBytes - twoRanks.indexerBytes, 6437830656);
  assert.equal(twoRanks.indexerBytes, 402388992);
  assert.equal(fourRanks.attentionBytes - fourRanks.indexerBytes, 12875661312);
  assert.equal(fourRanks.indexerBytes, 804777984);
  assert.equal(fourRanks.stateBytes, 116195328);
  assert.equal(fourRanks.bytes, 13796634624);
  assert.throws(() => estimateSafetensorsKvCache(flashNext, { tensorParallelSize: 0 }), RangeError);
});

test("compressed index keys round groups up and never cache query-head multiplicity", () => {
  const config = {
    num_hidden_layers: 2, num_attention_heads: 2, num_key_value_heads: 1, head_dim: 4,
    dtype: "float16", indexer_head_dim: 3, indexer_kv_heads: 2, indexer_n_heads: 64, indexer_compress_ratio: 4,
  };
  assert.equal(estimateSafetensorsKvCache(config, { maxModelLen: 4 }).indexerBytes, 120);
  assert.equal(estimateSafetensorsKvCache(config, { maxModelLen: 5 }).indexerBytes, 144);
  assert.equal(estimateSafetensorsKvCache({ ...config, indexer_n_heads: 1 }, { maxModelLen: 5 }).indexerBytes, 144);
  assert.equal(estimateSafetensorsKvCache({ ...config, indexer_compress_ratio: 1 }, { maxModelLen: 5 }).indexerBytes, 144);
});

test("partial RoPE width does not replace a declared full key head dimension", () => {
  const cache = estimateSafetensorsKvCache({
    num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 1,
    head_dim: 16, qk_rope_head_dim: 4, dtype: "float16",
  }, { maxModelLen: 8 });
  assert.equal(cache.bytes, 1024);
});

test("compressed MLA payload is replicated per tensor-parallel rank", () => {
  const cache = estimateSafetensorsKvCache({ num_hidden_layers: 2, kv_lora_rank: 5, qk_rope_head_dim: 2 }, {
    maxModelLen: 8, tensorParallelSize: 4,
  });
  assert.equal(cache.bytes, 896);
});

test("GGUF indexer breakdown follows the selected cache precision", () => {
  const cache = estimateGgufKvCache({
    "general.architecture": "new_decoder", "new_decoder.block_count": 2,
    "new_decoder.embedding_length": 8, "new_decoder.attention.head_count": 2,
    "new_decoder.attention.head_count_kv": 1, "new_decoder.context_length": 8,
    "new_decoder.attention.indexer.key_length": 4, "new_decoder.attention.indexer.compress_ratio": 2,
  }, { dtype: "Q4_K" });
  assert.equal(cache.attentionBytes, 99);
  assert.equal(cache.indexerBytes, 27);
  assert.equal(cache.bytes, 99);
});

test("same-repository draft can select a different parallel topology", async () => {
  const json = new TextEncoder().encode(JSON.stringify({ weight: { dtype: "F16", shape: [2], data_offsets: [0, 4] } }));
  const bytes = new Uint8Array(8 + json.length + 4);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(json.length), true);
  bytes.set(json, 8);
  const fetcher = async (input, init) => {
    const url = String(input);
    if (url.includes("/revision/")) return Response.json({ sha: "a".repeat(40) });
    if (url.includes("/tree/")) return Response.json(["model.safetensors", "config.json"].map(path => ({ type: "file", path })));
    if (url.endsWith("config.json")) return Response.json({ num_hidden_layers: 2, num_attention_heads: 8, num_key_value_heads: 1, head_dim: 4, dtype: "float16", max_position_embeddings: 16 });
    const range = new Headers(init.headers).get("range").match(/bytes=(\d+)-(\d+)/);
    const start = Number(range[1]); const end = Number(range[2]);
    return new Response(bytes.slice(start, end + 1), { status: 206, headers: { "Content-Range": `bytes ${start}-${end}/${bytes.length}` } });
  };
  const result = await estimateModelMemory({
    modelId: "org/model", fetch: fetcher, kvCache: true, tensorParallelSize: 4,
    draftModel: { modelId: "org/model", tensorParallelSize: 1 },
  });
  assert.equal(result.kvCacheBytes, 2048);
  assert.equal(result.draft.kvCacheBytes, 512);
  assert.equal(result.totalBytes, 2568);
  assert.equal(result.kvCacheBytesByTp, null);
  assert.equal(result.totalBytesByTp, null);
  assert.equal(result.files.safetensors.kvCacheByTp, null);
  assert.equal(result.draft.kvCacheBytesByTp, null);
  assert.equal(result.draft.totalBytesByTp, null);
  assert.equal(result.draft.files.safetensors.kvCacheByTp, null);
});
