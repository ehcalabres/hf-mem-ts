import { test } from "node:test";
import assert from "node:assert/strict";

import * as config_ from "../dist/config.js";
import { cdiv } from "../dist/py.js";
import { Weights } from "../dist/weights.js";
import { kimiConfig, llamaConfig } from "./helpers.js";

function kv(config, dtype = "BF16", length = 1024, batch = 1) {
  return config_.kvCache(config, dtype, length, batch);
}

function some(value) {
  assert.notEqual(value, null);
  assert.notEqual(value, undefined);
  return value;
}

const repeat = (value, n) => Array.from({ length: n }, () => value);

test("gqa", () => {
  const [size, layers] = kv(llamaConfig({ head_dim: 64 }));
  assert.equal(size, 4 * 1024 * 2 * 2 * 64 * 2);
  assert.deepEqual(layers, { full: 4, sliding: 0, state: 0, none: 0 });
});

test("mla caches the latent", () => {
  const config = llamaConfig({ kv_lora_rank: 512, qk_rope_head_dim: 64 });
  assert.equal(kv(config)[0], 4 * 1024 * (512 + 64) * 2);
});

function slidingTokens(window, length = 100_000) {
  const tokens = Math.min(window - 1 + 2 * config_.MAX_NUM_BATCHED_TOKENS, length);
  return (cdiv(tokens, 16) + (tokens < length ? 1 : 0)) * 16;
}

test("sliding window layers reserve the window plus batched tokens", () => {
  const pure = llamaConfig({ sliding_window: 128, layer_types: repeat("sliding_attention", 4) });
  assert.equal(kv(pure, undefined, 100_000)[0], 4 * slidingTokens(128) * 2 * 2 * 32 * 2);

  const hybrid = llamaConfig({
    sliding_window: 128,
    layer_types: [...repeat("sliding_attention", 3), "full_attention"],
  });
  assert.equal(kv(hybrid, undefined, 100_000)[0], (100_000 + 3 * slidingTokens(128)) * 2 * 2 * 32 * 2);
  assert.equal(kv(hybrid, undefined, 1024)[0], 4 * 1024 * 2 * 2 * 32 * 2);
});

test("gemma3 defaults", () => {
  const config = {
    model_type: "gemma3_text",
    num_hidden_layers: 12,
    num_attention_heads: 8,
    hidden_size: 3840,
    sliding_window: 1024,
  };
  const [size, layers] = kv(config, undefined, 32768);
  assert.deepEqual(layers, { full: 2, sliding: 10, state: 0, none: 0 });
  assert.equal(size, (2 * 32768 + 10 * slidingTokens(1024, 32768)) * 2 * 4 * 256 * 2);
});

test("sliding window pattern", () => {
  const config = llamaConfig({ num_hidden_layers: 6, sliding_window: 64, sliding_window_pattern: 3 });
  assert.deepEqual(config_.layerKinds(config_.normalize(config)), [
    ...["sliding", "sliding", "full"],
    ...["sliding", "sliding", "full"],
  ]);
});

test("linear attention hybrid pads states to attention pages", () => {
  const config = llamaConfig({
    layer_types: [...repeat("linear_attention", 3), "full_attention"],
    linear_num_value_heads: 4,
    linear_num_key_heads: 2,
    linear_key_head_dim: 16,
    linear_value_head_dim: 16,
    linear_conv_kernel_dim: 4,
    mamba_ssm_dtype: "float32",
  });
  const state = (2 * 2 * 16 + 4 * 16) * 3 * 2 + 4 * 16 * 16 * 4;
  const token = 2 * 2 * 32 * 2;
  const block = cdiv(cdiv(state, token), 16) * 16;
  const [size, layers] = kv(config);
  assert.deepEqual(layers, { full: 1, sliding: 0, state: 3, none: 0 });
  assert.equal(size, (cdiv(1024, block) + 3 * 2) * block * token);
});

test("mamba hybrid pattern", () => {
  const config = llamaConfig({
    num_hidden_layers: null,
    hybrid_override_pattern: "M-M*",
    mamba_num_heads: 8,
    mamba_head_dim: 16,
    ssm_state_size: 32,
    n_groups: 1,
    conv_kernel: 4,
  });
  const [size, layers] = kv(config);
  assert.deepEqual(layers, { full: 1, sliding: 0, state: 2, none: 1 });
  const state = (8 * 16 + 2 * 32) * 3 * 2 + 8 * 16 * 32 * 2;
  const token = 2 * 2 * 32 * 2;
  const block = cdiv(cdiv(state, token), 16) * 16;
  assert.equal(size, (cdiv(1024, block) + 2 * 3) * block * token);
});

test("pure state model", () => {
  const config = llamaConfig({ layer_types: repeat("mamba", 4), ssm_state_size: 16, mamba_expand: 2, conv_kernel: 4 });
  const inner = 2 * 256;
  assert.equal(kv(config)[0], 4 * 3 * ((inner + 2 * 16) * 3 * 2 + inner * 16 * 2));
});

test("text config resolution", () => {
  const nested = {
    architectures: ["LlavaForConditionalGeneration"],
    torch_dtype: "float16",
    text_config: llamaConfig(),
  };
  assert.equal(some(config_.textConfig(nested)).torch_dtype, "bfloat16");
  delete nested.text_config.torch_dtype;
  assert.equal(some(config_.textConfig(nested)).torch_dtype, "float16");

  const aliases = {
    architectures: ["GPT2LMHeadModel"],
    n_layer: 2,
    n_head: 4,
    n_embd: 64,
    n_positions: 128,
  };
  const resolved = some(config_.textConfig(aliases));
  assert.deepEqual([resolved.num_hidden_layers, resolved.max_position_embeddings], [2, 128]);

  const vision = {
    architectures: ["X"],
    vision_config: llamaConfig({ hidden_size: 4096 }),
    decoder: llamaConfig(),
  };
  assert.equal(some(config_.textConfig(vision)).hidden_size, 256);

  assert.equal(config_.textConfig({ ...llamaConfig(), architectures: ["BertForMaskedLM"] }), null);
});

for (const modelType of ["llama", "qwen2", "gemma3_text"]) {
  for (const key of ["max_position_embeddings", "model_max_length", "n_positions", "max_seq_len", "seq_length"]) {
    test(`context aliases override architecture defaults [${key}-${modelType}]`, () => {
      const config = llamaConfig({ model_type: modelType });
      delete config.max_position_embeddings;
      config[key] = 65536;
      assert.equal(config_.normalize(config).max_position_embeddings, 65536);
      assert.ok(!("max_position_embeddings" in config) || key === "max_position_embeddings");
    });
  }
}

test("explicit legacy context takes precedence over aliases", () => {
  assert.equal(config_.normalize(llamaConfig({ model_max_length: 65536 })).max_position_embeddings, 1024);
  assert.equal(
    config_.normalize(llamaConfig({ max_position_embeddings: null, model_max_length: 65536 })).max_position_embeddings,
    65536,
  );
});

for (const parentKey of ["dtype", "torch_dtype"]) {
  for (const childKey of ["dtype", "torch_dtype"]) {
    test(`nested dtype precedes the parent alias [${childKey}-${parentKey}]`, () => {
      const child = llamaConfig();
      delete child.torch_dtype;
      child[childKey] = "bfloat16";
      const config = some(config_.textConfig({ [parentKey]: "float16", text_config: child }));
      assert.equal(config_.kvDtype(config, "auto", new Weights(), []), "BF16");
    });
  }
}

test("local layers use sliding attention dimensions", () => {
  const config = llamaConfig({
    head_dim: 64,
    swa_head_dim: 32,
    swa_num_key_value_heads: 4,
    local_layer_ids: [0, 2],
    sliding_window_size: 128,
  });
  const [size, layers] = kv(config, undefined, 100000);
  assert.deepEqual(layers, { full: 2, sliding: 2, state: 0, none: 0 });
  assert.equal(size, (2 * 100000 * 2 * 2 * 64 + 2 * slidingTokens(128) * 2 * 4 * 32) * 2);
  const explicit = { ...config, layer_types: repeat("full_attention", 4) };
  assert.equal(kv(explicit, undefined, 100000)[1].full, 4);
});

test("per layer attention overrides and zero padded indices", () => {
  const config = llamaConfig({
    layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    sliding_window: 128,
    head_dim: 32,
    per_layer_config: {
      "01": { head_dim: 64, num_key_value_heads: 1 },
      "02": { sliding_window: 256 },
      "03": { head_dim: null, num_attention_heads: 4 },
    },
  });
  const [size, layers] = kv(config, undefined, 100000, 2);
  let expected = (slidingTokens(128) + slidingTokens(256)) * 2 * 2 * 32;
  expected += 100000 * (2 * 1 * 64 + 2 * 2 * 64);
  assert.equal(size, expected * 2 * 2);
  assert.deepEqual(layers, { full: 2, sliding: 2, state: 0, none: 0 });
  assert.equal(config_.kvHeads(config), 2);
  // TP=4 replicates the one-head layer four times and the two-head layers twice.
  const [replicated] = config_.kvCache(config, "BF16", 100000, 2, undefined, 4);
  assert.equal(replicated, (expected * 2 + 100000 * 2 * 1 * 64 * 2) * 2 * 2);
});

test("per layer windows can define the layer kinds", () => {
  const config = llamaConfig({ per_layer_config: { 1: { sliding_window: 128 }, 3: { attention_chunk_size: 256 } } });
  assert.deepEqual(config_.layerKinds(config_.normalize(config)), ["full", "sliding", "full", "sliding"]);
});

for (const heterogeneous of [false, true]) {
  for (const sharing of [{ num_kv_shared_layers: 2 }, { kv_sharing_modes: ["store", null, "reuse", "reuse"] }]) {
    test(`shared layers do not allocate another cache [${JSON.stringify(sharing)}-${heterogeneous}]`, () => {
      const config = llamaConfig(sharing);
      if (heterogeneous) config.per_layer_config = { 1: { head_dim: 64 }, 2: { head_dim: 128 } };
      const [size, layers] = kv(config);
      assert.deepEqual(layers, { full: 2, sliding: 0, state: 0, none: 2 });
      assert.equal(size, 1024 * 2 * 2 * (32 + (heterogeneous ? 64 : 32)) * 2);
    });
  }
}

test("dummy layers do not allocate cache", () => {
  assert.equal(kv(llamaConfig({ layer_types: repeat("dummy", 4) }))[0], 0);
});

for (const explicitModes of [false, true]) {
  test(`shared mla retains independent indexer caches [${explicitModes}]`, () => {
    // Transformers #50305: a KV consumer may still own an indexer cache.
    const config = llamaConfig({
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      kv_source_layers: [0, 1, 0, 1],
      no_rope_layers: [1, 0, 1, 0],
      index_head_dim: 128,
      indexer_types: ["full", "full", "full", "shared"],
    });
    if (explicitModes) config.kv_sharing_modes = ["store", "store", "reuse_but_indexer", "reuse"];
    const [size, layers] = kv(config);
    assert.deepEqual(layers, { full: 2, sliding: 0, state: 0, none: 2 });
    assert.equal(size, 1024 * ((512 + 64 + 512) * 2 + 3 * (128 + 4)));
    assert.equal(config_.kvCache(config, "BF16", 1024, 1, undefined, 4)[0], size * 4);
  });
}

test("is generative", () => {
  assert.ok(config_.isGenerative({ architectures: ["Qwen3ForCausalLM"] }, new Map()));
  assert.ok(config_.isGenerative({ architectures: ["Custom"] }, new Map([["generation_config.json", 1]])));
  assert.ok(!config_.isGenerative({ architectures: ["BertModel"] }, new Map()));
});

for (const [architectures, files, expected] of [
  [["BertForMaskedLM"], {}, true],
  [["XLMRobertaForSequenceClassification"], {}, true],
  [["BertForTokenClassification"], {}, true],
  [["BertModel"], {}, true],
  [["Qwen3ForCausalLM"], { "config_sentence_transformers.json": 1, "generation_config.json": 1 }, true],
  [["LlamaForSequenceClassification"], { "generation_config.json": 1 }, true],
  [["LlamaModel"], { "generation_config.json": 1 }, false],
  [["LlamaForCausalLM"], {}, false],
  [["Dinov2Model"], { "preprocessor_config.json": 1 }, false],
  [["Wav2Vec2ForSequenceClassification"], { "preprocessor_config.json": 1 }, false],
]) {
  test(`is text encoder [${architectures}-${JSON.stringify(files)}-${expected}]`, () => {
    assert.equal(
      config_.isTextEncoder(llamaConfig({ architectures }), new Map(Object.entries(files))),
      expected,
    );
  });
}

test("is text encoder needs a top level text config", () => {
  const clip = { architectures: ["CLIPModel"], text_config: llamaConfig() };
  assert.ok(!config_.isTextEncoder(clip, new Map([["config_sentence_transformers.json", 1]])));
});

for (const [config, requested, expected] of [
  [{ torch_dtype: "bfloat16" }, "auto", "BF16"],
  [{ dtype: "float16" }, "auto", "F16"],
  [{ dtype: "float16", torch_dtype: "bfloat16" }, "auto", "F16"],
  [{ torch_dtype: "bfloat16", quantization_config: { quant_method: "fp8" } }, "auto", "BF16"],
  [{ quantization_config: { kv_cache_scheme: { num_bits: 8 } } }, "auto", "F8_E4M3"],
  [{ quantization_config: { kv_cache_quant_algo: "FP8" } }, "auto", "F8_E4M3"],
  [{ torch_dtype: "bfloat16" }, "fp8_e5m2", "F8_E5M2"],
  [{ torch_dtype: "bfloat16" }, "fp8", "F8_E4M3"],
  [{ torch_dtype: "bfloat16" }, "float16", "F16"],
]) {
  test(`kv dtype [${JSON.stringify(config)}-${requested}-${expected}]`, () => {
    assert.equal(config_.kvDtype(config, requested, new Weights(), []), expected);
  });
}

test("kv dtype falls back to the dominant weight dtype", () => {
  const weights = new Weights();
  const warnings = [];
  weights.add("F16", 10, 20);
  weights.add("F8_E4M3", 100, 100);
  assert.equal(config_.kvDtype({}, "auto", weights, warnings), "F16");
  assert.ok(warnings.length > 0);
});

test("global attention layers use their own kv heads and head dim", () => {
  const config = llamaConfig({
    num_key_value_heads: 16,
    head_dim: 256,
    num_global_key_value_heads: 4,
    global_head_dim: 512,
    sliding_window: 1024,
    layer_types: [...repeat("sliding_attention", 3), "full_attention"],
  });
  const [size] = kv(config, undefined, 131072);
  assert.equal(size, (131072 * 2 * 4 * 512 + 3 * slidingTokens(1024, 131072) * 2 * 16 * 256) * 2);
});

test("deepseek v41 compressed kv cache", () => {
  const config = {
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 1,
    hidden_size: 64,
    kv_lora_rank: 512,
    sliding_window: 128,
    compress_ratios: [0, 2, 1, 0],
    kv_source_layer_ids: [1, 2],
    index_source_layer_ids: [1, 2],
    index_head_dim: 128,
  };
  const [size, layers] = config_.kvCache(config, "F8_E4M3", 4096, 1, 1024);
  const sliding = 4 * cdiv(127 + 2048, 32) * 32 * config_.DSV4_RECORD;
  const perToken = (config_.DSV4_RECORD + config_.DSV4_INDEX_RECORD) * 1.5;
  assert.equal(size, Math.trunc(sliding + perToken * 4096));
  assert.ok(layers.compressed === 2 && layers.sliding === 4);
  assert.equal(config_.kvDtype(config, "auto", new Weights(), []), "F8_E4M3");
});

test("kimi linear attention config", () => {
  const config = {
    model_type: "kimi_linear",
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 8,
    hidden_size: 64,
    kv_lora_rank: 32,
    qk_rope_head_dim: 8,
    linear_attn_config: {
      full_attn_layers: [4],
      kda_layers: [1, 2, 3],
      num_heads: 4,
      head_dim: 16,
      short_conv_kernel_size: 4,
    },
  };
  assert.deepEqual(config_.layerKinds(config_.normalize(config)), ["state", "state", "state", "full"]);
  const [, layers] = config_.kvCache(config, "BF16", 1024, 1);
  assert.deepEqual([layers.full, layers.state], [1, 3]);
});

test("indexer cache only on indexer layers", () => {
  const base = {
    num_hidden_layers: 4,
    num_attention_heads: 8,
    hidden_size: 64,
    kv_lora_rank: 512,
    qk_rope_head_dim: 64,
  };
  const plain = config_.kvCache(base, "BF16", 1024, 1)[0];
  const every = config_.kvCache({ ...base, index_head_dim: 128 }, "BF16", 1024, 1)[0];
  const someLayers = config_.kvCache(
    { ...base, index_head_dim: 128, indexer_types: ["full", "shared", "shared", "full"] },
    "BF16",
    1024,
    1,
  )[0];
  assert.ok(every - plain === 4 * 1024 * 132 && someLayers - plain === 2 * 1024 * 132);
  const dsa = config_.kvCache({ ...base, dsa_config: { dsa_layers: [2, 4], index_head_dim: 128 } }, "BF16", 1024, 1)[0];
  assert.equal(dsa - plain, 2 * 1024 * 132);
  assert.equal(config_.kvDtype({ ...base, dsa_config: {} }, "auto", new Weights(), []), "FP8_DS_MLA");
  assert.equal(config_.kvCache(base, "FP8_DS_MLA", 1024, 1)[0], 4 * 1024 * 656);
});

// vLLM kimi-k31 (266e88c05): packed cache planner, with one KDA prefill checkpoint.
for (const [full, state, tp, dtype, length, expected] of [
  [3, 3, 1, "FP8_DS_MLA", 131072, 426725376],
  [3, 3, 1, "FP8_DS_MLA", 1048576, 2599145472],
  [3, 3, 8, "FP8_DS_MLA", 131072, 2970451968],
  [3, 3, 8, "FP8_DS_MLA", 1048576, 22832578560],
  [3, 9, 1, "FP8_DS_MLA", 16383, 387932160],
  [3, 9, 1, "FP8_DS_MLA", 16384, 387932160],
  [3, 9, 1, "FP8_DS_MLA", 16385, 426725376],
  [3, 9, 1, "FP8_DS_MLA", 131072, 659484672],
  [3, 9, 1, "FP8_DS_MLA", 1048576, 2831904768],
  [3, 9, 1, "BF16", 131072, 790732800],
  [3, 9, 1, "F8_E4M3", 131072, 592846848],
  [3, 9, 8, "FP8_DS_MLA", 131072, 3236462592],
  [3, 9, 8, "FP8_DS_MLA", 1048576, 23098589184],
  [3, 9, 8, "BF16", 131072, 4444323840],
  [3, 9, 8, "F8_E4M3", 131072, 2604957696],
  [3, 15, 1, "FP8_DS_MLA", 131072, 892243968],
  [3, 15, 1, "FP8_DS_MLA", 1048576, 3064664064],
  [3, 15, 8, "FP8_DS_MLA", 131072, 3502473216],
  [3, 15, 8, "FP8_DS_MLA", 1048576, 23364599808],
  [24, 69, 1, "FP8_DS_MLA", 131072, 3879321600],
  [24, 69, 1, "FP8_DS_MLA", 1048576, 21258682368],
  [24, 69, 8, "FP8_DS_MLA", 131072, 21258682368],
  [24, 69, 8, "FP8_DS_MLA", 1048576, 160293568512],
]) {
  test(`kimi sparse cache matches vllm rows [${full}-${state}-${tp}-${dtype}-${length}-${expected}]`, () => {
    const config = kimiConfig(full, state);
    const [size, layers] = config_.kvCache(config, dtype, length, 1, undefined, tp);
    assert.equal(size, expected);
    assert.deepEqual([layers.full, layers.state], [full, state]);
    assert.equal(config_.kvCache(config, dtype, length, 2, undefined, tp)[0], 2 * expected);
  });
}

test("kimi dense cache keeps the prefill checkpoint", () => {
  const config = kimiConfig(undefined, undefined, false);
  assert.equal(config_.kvCache(config, "BF16", 131072, 1)[0], 637009920);
  assert.equal(config_.kvCache(config, "BF16", 1048576, 1)[0], 3822059520);
});

for (const config of [{ dtype: "float32" }, { torch_dtype: "float32" }, {}]) {
  test(`fp32 auto cache uses a shared 16 bit reference [${JSON.stringify(config)}]`, () => {
    const weights = new Weights();
    weights.add("F32", 10, 40);
    assert.equal(config_.kvDtype(config, "auto", weights, []), "F16");
    assert.equal(config_.kvDtype(config, "bfloat16", weights, []), "BF16");
  });
}
