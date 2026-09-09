import assert from "node:assert/strict";
import test from "node:test";
import { estimateGgufKvCache, estimateSafetensorsKvCache, resolveSafetensorsKvDtype } from "../dist/index.js";
import { formatResult } from "../dist/report.js";

const gqa = {
  model_type: "llama", hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32,
  num_key_value_heads: 8, max_position_embeddings: 4096, torch_dtype: "bfloat16",
};
const gguf = {
  "general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
  "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096, "llama.context_length": 4096,
};

// Cache-relevant fields from Qwen/Qwen3.5-0.8B/raw/main/config.json.
const qwen35 = {
  model_type: "qwen3_5", text_config: {
    model_type: "qwen3_5_text", dtype: "bfloat16", hidden_size: 1024, head_dim: 256,
    num_hidden_layers: 24, num_attention_heads: 8, num_key_value_heads: 2,
    max_position_embeddings: 262144, full_attention_interval: 4,
    layer_types: [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
    linear_num_key_heads: 16, linear_num_value_heads: 16,
    linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
    mamba_ssm_dtype: "float32",
  },
};

// DeepSeek-V3 publishes BF16 compute precision despite FP8 weight quantization.
const deepseek = {
  model_type: "deepseek_v3", hidden_size: 7168, num_hidden_layers: 61,
  num_attention_heads: 128, num_key_value_heads: 128, kv_lora_rank: 512,
  qk_rope_head_dim: 64, qk_nope_head_dim: 128, v_head_dim: 128,
  max_position_embeddings: 163840, torch_dtype: "bfloat16",
  quantization_config: { quant_method: "fp8", fmt: "e4m3", weight_block_size: [128, 128] },
};

test("ordinary GQA retains its 512 MiB payload with a discoverable cache breakdown", () => {
  const cache = estimateSafetensorsKvCache(gqa);
  assert.equal(cache.bytes, 536870912);
  assert.equal(cache.attentionBytes, 536870912);
  assert.equal(cache.stateBytes, 0);
  assert.equal(cache.fullAttentionLayers, 32);
  assert.equal(cache.recurrentDtype, null);
  assert.equal(cache.layout, "attention");
});

test("Qwen3.5 counts six attention caches and eighteen length-independent Gated DeltaNet states", () => {
  const cache = estimateSafetensorsKvCache(qwen35, { maxModelLen: 32768 });
  assert.equal(cache.attentionBytes, 402653184);
  // Each linear layer has a [1,6144,4] BF16 convolution buffer (49152 B),
  // and a [1,16,128,128] FP32 recurrent buffer (1048576 B).
  assert.equal(cache.convolutionBytes, 884736);
  assert.equal(cache.recurrentBytes, 18874368);
  assert.equal(cache.stateBytes, 19759104);
  assert.equal(cache.bytes, 422412288);
  assert.equal(cache.fullAttentionLayers, 6);
  assert.equal(cache.recurrentLayers, 18);
  assert.equal(cache.slidingAttentionLayers, 0);
  assert.equal(cache.recurrentDtype, "F32");
  assert.equal(cache.convolutionDtype, "BF16");
  assert.equal(cache.layout, "hybrid");
  const longer = estimateSafetensorsKvCache(qwen35, { maxModelLen: 65536 });
  assert.equal(longer.attentionBytes, 805306368);
  assert.equal(longer.stateBytes, 19759104);
  const batched = estimateSafetensorsKvCache(qwen35, { maxModelLen: 32768, batchSize: 2, dtype: "fp8" });
  assert.equal(batched.attentionBytes, 402653184);
  assert.equal(batched.stateBytes, 39518208);
});

test("Qwen3.5 recurrent storage precision is independently selectable for backend differences", () => {
  const cache = estimateSafetensorsKvCache(qwen35, { maxModelLen: 32768, recurrentStateDtype: "bfloat16" });
  assert.equal(cache.recurrentBytes, 9437184);
  assert.equal(cache.convolutionBytes, 884736);
  assert.equal(cache.stateBytes, 10321920);
  assert.equal(cache.recurrentDtype, "BF16");
  assert.equal(cache.attentionBytes, 402653184);
});

test("DeepSeek-V3 compressed MLA uses a shared latent and RoPE key, not per-head expanded KV", () => {
  const cache = estimateSafetensorsKvCache(deepseek, { maxModelLen: 32768 });
  assert.equal(cache.bytes, 2302672896);
  assert.equal(cache.dtype, "BF16");
  assert.equal(cache.layout, "mla-compressed");
  assert.equal(cache.stateBytes, 0);
  // Expanded reference stores 128 heads, with 192 key and 128 value elements per head.
  const expanded = estimateSafetensorsKvCache(deepseek, { maxModelLen: 1, mlaLayout: "expanded" });
  assert.equal(expanded.bytes, 4997120);
  assert.equal(expanded.layout, "mla-expanded");
});

test("tiny Qwen3.5 BF16 recurrent policy matches measured CPU eager buffer sizes", () => {
  // Parent calibration: torch 2.13.0 / Transformers 5.13.0, no pretrained weights.
  // Three BF16 convolution tensors each [1,24,4] and recurrent tensors [1,2,4,4].
  const config = {
    model_type: "qwen3_5_text", dtype: "bfloat16", hidden_size: 16, head_dim: 8,
    num_hidden_layers: 4, num_attention_heads: 2, num_key_value_heads: 1,
    max_position_embeddings: 32, layer_types: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_num_key_heads: 2, linear_num_value_heads: 2, linear_key_head_dim: 4,
    linear_value_head_dim: 4, linear_conv_kernel_dim: 4, mamba_ssm_dtype: "float32",
  };
  const before = estimateSafetensorsKvCache(config, { maxModelLen: 7, recurrentStateDtype: "bfloat16" });
  const after = estimateSafetensorsKvCache(config, { maxModelLen: 8, recurrentStateDtype: "bfloat16" });
  assert.equal(before.convolutionBytes, 576);
  assert.equal(before.recurrentBytes, 192);
  assert.equal(before.attentionBytes, 224);
  assert.equal(before.bytes, 992);
  assert.equal(after.attentionBytes, 256);
  assert.equal(after.stateBytes, 768);
  assert.equal(after.bytes, 1024);
});

test("sliding allocation policy works for genuine all-sliding and hybrid attention", () => {
  const config = {
    hidden_size: 8, num_hidden_layers: 4, num_attention_heads: 2, num_key_value_heads: 1,
    max_position_embeddings: 16, sliding_window: 4, torch_dtype: "float16",
    layer_types: ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
  };
  assert.equal(estimateSafetensorsKvCache(config).bytes, 640);
  assert.equal(estimateSafetensorsKvCache(config, { slidingWindowPolicy: "full-context" }).bytes, 1024);
  assert.equal(estimateSafetensorsKvCache({ ...config, layer_types: Array(4).fill("sliding_attention") }).bytes, 256);
  assert.equal(estimateSafetensorsKvCache(config, { maxModelLen: 2 }).bytes, 128);
});

test("auto cache dtype ignores weight-only FP8 and honors only explicit cache precision", () => {
  assert.equal(resolveSafetensorsKvDtype({ torch_dtype: "bfloat16", quantization_config: { quant_method: "modelopt", format: "e5m2" } }), "BF16");
  assert.equal(resolveSafetensorsKvDtype({ torch_dtype: "bfloat16", quantization_config: { kv_cache_scheme: { num_bits: 8, type: "float" } } }), "F8_E4M3");
  assert.equal(resolveSafetensorsKvDtype({ torch_dtype: "float16", kv_cache_dtype: "fp8_e5m2" }), "F8_E5M2");
  assert.equal(resolveSafetensorsKvDtype({ quantization_config: { quant_method: "fp8" } }), "F16");
  assert.equal(resolveSafetensorsKvDtype({ dtype: "float16", torch_dtype: "float32" }), "F32");
  assert.equal(resolveSafetensorsKvDtype({ dtype: "float16", torch_dtype: "bfloat16" }, "bf16"), "BF16");
  assert.equal(resolveSafetensorsKvDtype({ kv_cache_dtype: "bf16", quantization_config: { kv_cache_dtype: "fp8" } }), "BF16");
});

test("GGUF uses the selected architecture namespace and separate explicit key/value dimensions", () => {
  const foreign = { "vision.block_count": 99, "vision.attention.head_count_kv": 99 };
  assert.equal(estimateGgufKvCache({ ...foreign, ...gguf }).bytes, 536870912);
  // 384 key + 128 value elements doubles the original 128+128 payload.
  assert.equal(estimateGgufKvCache({ ...gguf, "llama.attention.key_length": 384, "llama.attention.value_length": 128 }).bytes, 1073741824);
  const missingKvHeads = { ...gguf };
  delete missingKvHeads["llama.attention.head_count_kv"];
  assert.equal(estimateGgufKvCache(missingKvHeads).bytes, 2147483648);
  const ambiguous = { ...foreign, ...gguf };
  delete ambiguous["general.architecture"];
  assert.throws(() => estimateGgufKvCache(ambiguous), /unambiguous/);
});

test("GGUF cache quantization uses effective bits even without exact row packing", () => {
  assert.equal(estimateGgufKvCache(gguf, { dtype: "Q8_0" }).bytes, 285212672);
  const unaligned = estimateGgufKvCache({ ...gguf, "llama.attention.key_length": 129 }, { dtype: "Q8_0" });
  assert.equal(unaligned.bytes, 286326784);
  assert.equal(unaligned.approximate, true);
  assert.equal(estimateGgufKvCache(gguf, { dtype: "Q4_K" }).bytes, 150994944);
});

test("low-level estimators reject invalid lengths, dimensions and unsafe arithmetic", () => {
  for (const value of [-1, 0, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => estimateSafetensorsKvCache(gqa, { maxModelLen: value }), RangeError);
    assert.throws(() => estimateGgufKvCache(gguf, { maxModelLen: value }), RangeError);
    assert.throws(() => estimateSafetensorsKvCache(gqa, { batchSize: value }), RangeError);
    assert.throws(() => estimateGgufKvCache(gguf, { batchSize: value }), RangeError);
  }
  assert.throws(() => estimateSafetensorsKvCache({ ...gqa, head_dim: -1 }), RangeError);
  assert.equal(estimateSafetensorsKvCache({ ...gqa, hidden_size: 4097 }).bytes, 541065216);
  assert.equal(estimateGgufKvCache({ ...gguf, "llama.embedding_length": 4097 }).bytes, 541065216);
  assert.throws(() => estimateGgufKvCache({ ...gguf, "llama.attention.value_length": -1 }), RangeError);
  assert.throws(() => estimateSafetensorsKvCache(gqa, { maxModelLen: Number.MAX_SAFE_INTEGER }), RangeError);
  assert.throws(() => estimateGgufKvCache(gguf, { maxModelLen: Number.MAX_SAFE_INTEGER }), RangeError);
});

test("model names never gate cache support and unfamiliar layers receive explicit proxies", () => {
  assert.deepEqual(estimateSafetensorsKvCache({ ...gqa, model_type: "future_architecture" }), estimateSafetensorsKvCache(gqa));
  const unknown = estimateSafetensorsKvCache({ ...gqa, layer_types: Array(32).fill("unknown_attention") });
  assert.equal(unknown.bytes, 536870912);
  assert.equal(unknown.approximate, true);
  const recurrent = estimateSafetensorsKvCache({ ...gqa, layer_types: Array(32).fill("linear_attention") });
  assert.equal(recurrent.bytes, 536870912);
  assert.equal(recurrent.recurrentLayers, 32);
  assert.equal(recurrent.approximate, true);
  assert.match(recurrent.assumptions.join(" "), /proxy/);
  const shared = estimateSafetensorsKvCache({ ...gqa, num_kv_shared_layers: 2 });
  assert.equal(shared.bytes, 536870912);
  assert.equal(shared.approximate, true);
  assert.equal(estimateSafetensorsKvCache({ ...gqa, layer_types: ["full_attention"] }).bytes, 536870912);
});

test("GGUF noncausal, shared-cache and incomplete window metadata remain estimable", () => {
  const noncausal = estimateGgufKvCache({ ...gguf, "llama.attention.causal": false });
  assert.equal(noncausal.bytes, 536870912);
  assert.equal(noncausal.approximate, true);
  assert.equal(estimateGgufKvCache({ ...gguf, "llama.attention.shared_kv_layers": 4 }).bytes, 536870912);
  const missingWindow = estimateGgufKvCache({ ...gguf, "llama.attention.sliding_window_pattern": 2 });
  assert.equal(missingWindow.bytes, 536870912);
  assert.equal(missingWindow.approximate, true);
});

test("reports actual target and draft cache layouts, state sizes and independent storage policies", () => {
  const cache = estimateSafetensorsKvCache(qwen35, { maxModelLen: 32768 });
  const draftCache = estimateSafetensorsKvCache(qwen35, { maxModelLen: 32768, recurrentStateDtype: "bfloat16" });
  const file = { parameters: 0, bytes: 0, components: {}, kvCache: cache };
  const result = {
    modelId: "Qwen/Qwen3.5-0.8B", revision: "main", format: "safetensors", filename: null,
    weightsBytes: 0, kvCacheBytes: cache.bytes, totalBytes: cache.bytes + draftCache.bytes,
    files: { safetensors: file }, mmproj: null,
    draft: {
      modelId: "Qwen/Qwen3.5-0.8B", revision: "main", format: "safetensors", filename: null,
      weightsBytes: 0, kvCacheBytes: draftCache.bytes, totalBytes: draftCache.bytes,
      files: { safetensors: { ...file, kvCache: draftCache } }, mmproj: null, draft: null,
    },
  };
  const output = formatResult(result);
  assert.match(output, /Target safetensors: hybrid, BF16 attention/);
  assert.match(output, /Attention payload: [^\n]*402653184 bytes/);
  assert.match(output, /Convolution state: 884736 bytes \(BF16\)/);
  assert.match(output, /Recurrent state: 18874368 bytes \(F32\)/);
  assert.match(output, /Draft safetensors: hybrid, BF16 attention/);
  assert.match(output, /Recurrent state: 9437184 bytes \(BF16\)/);
});

test("common dimension aliases and MQA/GQA hints work without named architectures", () => {
  const config = { n_layer: 2, n_embd: 16, n_head: 4, n_positions: 8, torch_dtype: "float16" };
  assert.equal(estimateSafetensorsKvCache(config).bytes, 1024);
  assert.equal(estimateSafetensorsKvCache({ ...config, multi_query: true }).bytes, 256);
  assert.equal(estimateSafetensorsKvCache({ ...config, attn_config: { kv_n_heads: 2 } }).bytes, 512);
});


test("missing query counts preserve full projection width when head dimensions exist", () => {
  const config = { num_hidden_layers: 2, hidden_size: 16, head_dim: 4, max_position_embeddings: 8, dtype: "float16" };
  const mha = estimateSafetensorsKvCache(config);
  assert.equal(mha.bytes, 1024);
  assert.equal(mha.approximate, true);
  assert.equal(estimateSafetensorsKvCache({ ...config, num_key_value_heads: 2 }).bytes, 512);
});
test("missing context, heads and dtype produce a disclosed full-width estimate", () => {
  const config = { num_layers: 2, d_model: 8, model_type: "brand_new_decoder" };
  const cache = estimateSafetensorsKvCache(config);
  assert.equal(cache.bytes, 262144);
  assert.equal(cache.maxModelLen, 4096);
  assert.equal(cache.dtype, "F16");
  assert.equal(cache.approximate, true);
  assert.equal(estimateSafetensorsKvCache(config, { maxModelLen: 16, dtype: "float32" }).bytes, 2048);
  assert.throws(() => estimateSafetensorsKvCache({ num_layers: 2, n_head: 2 }), /Insufficient cache dimensions/);
});

test("nested decoder precision and global head dimensions override outer/local values", () => {
  const cache = estimateSafetensorsKvCache({
    torch_dtype: "float32",
    language_model_config: {
      dtype: "bfloat16", hidden_size: 8, num_hidden_layers: 2, num_attention_heads: 2,
      num_key_value_heads: 2, head_dim: 4, global_head_dim: 16, num_global_key_value_heads: 1,
      max_position_embeddings: 16, sliding_window: 4, layer_types: ["sliding_attention", "full_attention"],
    },
  });
  assert.equal(cache.dtype, "BF16");
  assert.equal(cache.bytes, 1152);
});

test("state dimensions determine recurrent storage independently of model names and context", () => {
  const config = {
    model_type: "future_ssm", d_model: 8, n_layer: 3, d_inner: 16, d_state: 4, d_conv: 3,
    dtype: "bfloat16", state_dtype: "float32", layer_types: ["ssm", "ssm", "ssm"],
  };
  const cache = estimateSafetensorsKvCache(config, { maxModelLen: 16 });
  assert.equal(cache.convolutionBytes, 288);
  assert.equal(cache.recurrentBytes, 768);
  assert.equal(cache.bytes, 1056);
  assert.equal(cache.attentionBytes, 0);
  assert.equal(cache.layout, "recurrent");
  assert.equal(estimateSafetensorsKvCache(config, { maxModelLen: 32768 }).bytes, 1056);
  assert.equal(estimateSafetensorsKvCache({ ...qwen35.text_config, model_type: "future_linear_decoder" }, { maxModelLen: 32768 }).bytes, 422412288);
});

test("MLA-only dimensions and indexer metadata improve estimates without a model registry", () => {
  assert.equal(estimateSafetensorsKvCache({ num_hidden_layers: 2, kv_lora_rank: 5, qk_rope_head_dim: 2 }, { maxModelLen: 8 }).bytes, 224);
  const indexed = estimateSafetensorsKvCache({ ...deepseek, index_head_dim: 128, index_n_heads: 32 }, { maxModelLen: 1 });
  assert.equal(indexed.bytes, 85888);
  assert.equal(indexed.approximate, true);
});

test("encoder-decoder aliases include an explicit cross-attention length approximation", () => {
  const cache = estimateSafetensorsKvCache({
    model_type: "new_encoder_decoder", decoder_layers: 2, decoder_attention_heads: 2,
    d_model: 8, max_position_embeddings: 16, dtype: "float16", is_encoder_decoder: true,
  });
  assert.equal(cache.bytes, 2048);
  assert.equal(cache.approximate, true);
});
