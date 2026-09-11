import assert from "node:assert/strict";
import test from "node:test";
import { estimateGgufKvCache } from "../dist/index.js";

const metadata = (fields = {}, namespace = "unseen_decoder") => ({
  "general.architecture": namespace,
  ...Object.fromEntries(Object.entries({
    block_count: 2, embedding_length: 8, context_length: 8,
    "attention.head_count": 2, "attention.head_count_kv": 1,
    ...fields,
  }).map(([key, value]) => [`${namespace}.${key}`, value])),
});

test("GGUF namespaces select dimensions without restricting unfamiliar layouts", () => {
  const input = metadata();
  const ordinary = estimateGgufKvCache(input);
  assert.equal(ordinary.bytes, 256);
  assert.equal(estimateGgufKvCache(metadata({}, "another_unpublished_namespace")).bytes, ordinary.bytes);
  assert.equal(estimateGgufKvCache(metadata({
    "attention.kv_lora_rank": 0, "rope.dimension_count": 0, "attention.sliding_window": 0,
  })).bytes, ordinary.bytes);
  assert.equal(estimateGgufKvCache({ ...input, "unseen_decoder.attention.causal": false,
    "unseen_decoder.attention.shared_kv_layers": 1, "unseen_decoder.encoder.block_count": 2 }).bytes, ordinary.bytes);
  const noncausal = estimateGgufKvCache({ ...input, "unseen_decoder.attention.causal": false });
  assert.equal(noncausal.approximate, true);
  assert.match(noncausal.assumptions.join(" "), /noncausal|encoder/i);
  const shared = estimateGgufKvCache({ ...input, "unseen_decoder.attention.shared_kv_layers": 1 });
  assert.equal(shared.approximate, true);
  assert.match(shared.assumptions.join(" "), /sharing|shared/i);
});

test("GGUF scalar and per-layer head and key/value dimensions retain every layer", () => {
  const cache = estimateGgufKvCache(metadata({
    block_count: 3, "attention.head_count": [2, 4, 2], "attention.head_count_kv": [1, 2, 1],
    "attention.key_length": [3, 5, 7], "attention.value_length": [2, 4, 6],
  }));
  assert.equal(cache.bytes, 576);
  assert.equal(cache.fullAttentionLayers, 3);
});

test("GGUF recurrent head slots retain length-independent state under quantized attention", () => {
  const input = metadata({
    block_count: 3, "attention.head_count": [0, 2, 0], "attention.head_count_kv": [0, 1, 0],
    "ssm.state_size": 3, "ssm.inner_size": 6, "ssm.conv_kernel": 4, "ssm.group_count": 2,
    "ssm.time_step_rank": 7,
  });
  const half = estimateGgufKvCache(input);
  const quantized = estimateGgufKvCache(input, { dtype: "Q4_K" });
  const longer = estimateGgufKvCache(input, { maxModelLen: 16, dtype: "Q4_K" });
  assert.equal(half.layout, "hybrid");
  assert.equal(half.recurrentLayers, 2);
  assert.equal(half.fullAttentionLayers, 1);
  assert.equal(half.attentionBytes, 128);
  assert.equal(half.stateBytes, 432);
  assert.equal(quantized.attentionBytes, 36);
  assert.equal(quantized.stateBytes, half.stateBytes);
  assert.equal(quantized.recurrentDtype, half.recurrentDtype);
  assert.equal(quantized.convolutionDtype, half.convolutionDtype);
  assert.equal(longer.attentionBytes, 72);
  assert.equal(longer.stateBytes, half.stateBytes);
  assert.equal(quantized.bytes, half.stateBytes + 36);
  assert.equal(quantized.approximate, true);
  // time_step_rank is overloaded in GGUF and must not become GatedDeltaNet heads.
  const otherRank = estimateGgufKvCache({ ...input, "unseen_decoder.ssm.time_step_rank": 19 });
  assert.equal(otherRank.stateBytes, half.stateBytes);
});

test("GGUF explicit recurrent schedules do not silently discard unknown state", () => {
  const input = metadata({ "attention.recurrent_layers": [true, false] });
  const cache = estimateGgufKvCache(input);
  assert.equal(cache.bytes, 256);
  assert.equal(cache.approximate, true);
  assert.match(cache.assumptions.join(" "), /proxy/i);
});

test("GGUF recurrent flags override inferred full-attention intervals", () => {
  const cache = estimateGgufKvCache(metadata({
    block_count: 3, full_attention_interval: 2,
    "attention.recurrent_layers": [false, true, false],
    "ssm.state_size": 3, "ssm.inner_size": 6, "ssm.conv_kernel": 4,
  }));
  assert.equal(cache.recurrentLayers, 1);
  assert.equal(cache.fullAttentionLayers, 2);
  assert.equal(cache.attentionBytes, 256);
});

test("GGUF MLA rank and RoPE use compressed storage while expanded dimensions include RoPE once", () => {
  const input = metadata({
    embedding_length: 16, "attention.head_count": 4,
    "attention.kv_lora_rank": 5, "rope.dimension_count": 2,
    "attention.key_length_mla": 7, "attention.value_length_mla": 3,
  });
  const compressed = estimateGgufKvCache(input);
  const expanded = estimateGgufKvCache(input, { mlaLayout: "expanded" });
  assert.equal(compressed.bytes, 224);
  assert.equal(compressed.layout, "mla-compressed");
  assert.equal(expanded.bytes, 1280);
  assert.equal(expanded.layout, "mla-expanded");
});

test("GGUF missing sliding width uses full context rather than rejecting the schedule", () => {
  const cache = estimateGgufKvCache(metadata({ block_count: 3, "attention.sliding_window_pattern": 2 }));
  assert.equal(cache.bytes, 384);
  assert.equal(cache.slidingAttentionLayers, 2);
  assert.equal(cache.fullAttentionLayers, 1);
  assert.equal(cache.approximate, true);
  assert.match(cache.assumptions.join(" "), /window|full.context/i);
});

test("GGUF sliding arrays apply SWA dimensions without replacing full-attention dimensions", () => {
  const input = metadata({
    block_count: 3, "attention.sliding_window_pattern": [true, false, true], "attention.sliding_window": 2,
    "attention.key_length": 6, "attention.value_length": 4,
    "attention.key_length_swa": 2, "attention.value_length_swa": 2,
  });
  assert.equal(estimateGgufKvCache(input).bytes, 192);
  assert.equal(estimateGgufKvCache(input, { slidingWindowPolicy: "full-context" }).bytes, 288);
});

test("GGUF quantized nonaligned dimensions use effective bits with whole-byte rounding", () => {
  const input = metadata({
    block_count: 1, context_length: 1, "attention.head_count": 1, "attention.head_count_kv": 1,
    "attention.key_length": 3, "attention.value_length": 2,
  });
  const cache = estimateGgufKvCache(input, { dtype: "Q4_K" });
  assert.equal(cache.bytes, 3);
  assert.equal(cache.dtype, "Q4_K");
  assert.equal(cache.approximate, true);
  assert.match(cache.assumptions.join(" "), /padding|packing/i);
  assert.throws(() => estimateGgufKvCache(input, { dtype: "not_a_dtype" }), /dtype/i);
});

test("GGUF auto precision honors cache metadata but explicit options override it", () => {
  const input = metadata({ "attention.kv_cache_dtype": "F32" });
  assert.equal(estimateGgufKvCache(input).bytes, 512);
  assert.equal(estimateGgufKvCache(input, { dtype: "F16" }).bytes, 256);
  assert.equal(estimateGgufKvCache(metadata({})).dtype, "F16");
  const unrecognized = estimateGgufKvCache(metadata({ "attention.kv_cache_dtype": "future_precision" }));
  assert.equal(unrecognized.dtype, "F16");
  assert.equal(unrecognized.approximate, true);
});

test("GGUF fallback namespace requires unambiguous cache dimensions", () => {
  const input = metadata();
  delete input["general.architecture"];
  assert.equal(estimateGgufKvCache({ ...input, "vision.block_count": 99 }).bytes, 256);
  const ambiguous = { ...input, "other.block_count": 3, "other.embedding_length": 8 };
  assert.throws(() => estimateGgufKvCache(ambiguous), /unambiguous/i);
  assert.equal(estimateGgufKvCache({ ...ambiguous, "general.architecture": "unseen_decoder" }).bytes, 256);
  assert.throws(() => estimateGgufKvCache({ "general.architecture": "empty" }), /dimensions|namespace/i);
});
