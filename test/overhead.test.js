import { test } from "node:test";
import assert from "node:assert/strict";

import * as overhead from "../dist/overhead.js";

const GIB = overhead.GIB;

test("vllm overhead follows the gpu class", () => {
  const config = { vocab_size: 128256, intermediate_size: 14336, hidden_size: 4096, num_hidden_layers: 32 };
  const small = overhead.vllm(config, false, false, 2048);
  const large = overhead.vllm(config, false, false, 8192);
  assert.ok(1.2 * GIB < small.activations && small.activations < 1.6 * GIB);
  assert.ok(2.6 * GIB < large.activations && large.activations < 3.0 * GIB);
  assert.ok(0.35 * GIB < small.graphs && small.graphs < 0.5 * GIB);
  assert.deepEqual(
    [small.runtime, large.runtime],
    [Math.trunc(0.27 * GIB) + small.context, Math.trunc(0.87 * GIB) + large.context],
  );
  const moe = overhead.vllm({ ...config, moe_intermediate_size: 768, num_experts_per_tok: 8 }, true, false, 2048);
  assert.ok(moe.graphs > small.graphs);
  assert.equal(overhead.vllm(config, false, false, 2048, GIB).bytes, small.bytes + GIB);
});

test("encoder patches", () => {
  const qwen = {
    vision_config: {
      hidden_size: 1152,
      intermediate_size: 4304,
      patch_size: 16,
      hidden_act: "gelu",
    },
  };
  assert.equal(overhead.encoder(qwen, { size: { longest_edge: 16777216 } }), 65536 * 2 * (4 * 1152 + 2 * 4304));
  const gated = { vision_config: { ...qwen.vision_config, patch_size: 14, hidden_act: "silu" } };
  assert.equal(overhead.encoder(gated, { max_pixels: 12845056 }), 65536 * 2 * (4 * 1152 + 3 * 4304));
  const siglip = { vision_config: { ...qwen.vision_config, patch_size: 14, image_size: 896 } };
  assert.equal(overhead.encoder(siglip, {}), 4096 * 2 * (4 * 1152 + 2 * 4304));
  const tiles = {
    ...siglip,
    max_dynamic_patch: 12,
    use_thumbnail: true,
    vision_config: { ...siglip.vision_config, image_size: 448 },
  };
  assert.equal(overhead.encoder(tiles, {}), 13 * 1024 * 2 * (4 * 1152 + 2 * 4304));
  const pooled = {
    vision_config: { ...qwen.vision_config, pooling_kernel_size: 3 },
    vision_soft_tokens_per_image: 280,
  };
  assert.equal(overhead.encoder(pooled, {}), 2520 * 2 * (4 * 1152 + 2 * 4304));
  const audio = { audio_config: { d_model: 1280, encoder_ffn_dim: 5120, max_source_positions: 1500 } };
  assert.equal(overhead.encoder(audio, {}), 1500 * 2 * (4 * 1280 + 2 * 5120));
  assert.equal(overhead.encoder({}, {}), 0);
});

test("batched tokens follow the gpu size", () => {
  assert.equal(overhead.batchedTokens(null), 8192);
  assert.equal(overhead.batchedTokens("8x80"), 8192);
  assert.equal(overhead.batchedTokens("24"), 2048);
  assert.equal(overhead.batchedTokens("180"), 16384);
});

for (const [value, expected] of [
  ["80", [null, 80 * GIB]],
  ["8x80", [8, 80 * GIB]],
  ["2x24GiB", [2, 24 * GIB]],
  ["0.5", [null, Math.floor(GIB / 2)]],
]) {
  test(`parse gpu memory [${value}]`, () => {
    assert.deepEqual(overhead.parseGpuMemory(value), expected);
  });
}

for (const value of ["", "x80", "0x80", "-2x80", "8x", "eighty", "-1", "inf", "1e400"]) {
  test(`parse gpu memory errors [${value}]`, () => {
    assert.throws(() => overhead.parseGpuMemory(value), Error);
  });
}

test("fit", () => {
  const args = { kvFixed: 0, kvPerToken: 1024, maxModelLen: 1000, contextLength: 10 ** 9, overhead: GIB };
  let fit = overhead.fit({ weights: 10 * GIB, gpuMemory: "20", utilization: 0.9, ...args });
  assert.deepEqual([fit.needed, fit.fits], [1, true]);
  assert.equal(fit.maxModelLen, Math.trunc((18 * GIB - GIB - 10 * GIB) / 1024));
  assert.deepEqual([fit.usable, fit.required], [Math.trunc(18 * GIB), 10 * GIB + 1024 * 1000 + GIB]);
  const replicated = overhead.fit({ weights: 10 * GIB, gpuMemory: "4x20", utilization: 0.9, kvHeads: 1, ...args });
  assert.equal(replicated.required, 10 * GIB + 4 * 1024 * 1000 + 4 * GIB);

  fit = overhead.fit({ weights: 30 * GIB, gpuMemory: "20", utilization: 0.9, ...args });
  assert.deepEqual([fit.needed, fit.fits, fit.maxModelLen], [2, false, 0]);
  assert.ok(overhead.fit({ weights: 30 * GIB, gpuMemory: "2x20", utilization: 0.9, ...args }).fits);
  const limited = { ...args, contextLength: 5 };
  assert.equal(overhead.fit({ weights: 30 * GIB, gpuMemory: "2x20", utilization: 0.9, ...limited }).maxModelLen, 5);
  assert.equal(overhead.fit({ weights: 10 ** 15, gpuMemory: "20", utilization: 0.9, ...args }).needed, null);
});

test("fit keeps the cuda context outside the utilization", () => {
  const args = { weights: 0, kvFixed: 0, kvPerToken: 1024, maxModelLen: 1000, contextLength: 10 ** 9, gpuMemory: "20" };
  const withContext = overhead.fit({ overhead: 2 * GIB, utilization: 0.9, context: GIB, ...args });
  const without = overhead.fit({ overhead: GIB, utilization: 0.9, ...args });
  assert.equal(withContext.maxModelLen, without.maxModelLen);
  assert.equal(
    overhead.fit({ overhead: 2 * GIB, utilization: 1.0, context: GIB, ...args }).maxModelLen,
    Math.trunc((18 * GIB) / 1024),
  );
});

test("fit replicates kv heads across gpus", () => {
  const args = { kvFixed: 0, kvPerToken: GIB / 1000, maxModelLen: 1000, contextLength: 10 ** 9, overhead: 0 };
  const split = overhead.fit({ weights: 0, gpuMemory: "8x10", utilization: 1.0, kvHeads: 8, ...args });
  const replicated = overhead.fit({ weights: 0, gpuMemory: "8x10", utilization: 1.0, kvHeads: 2, ...args });
  assert.equal(split.maxModelLen, 4 * replicated.maxModelLen);
  assert.equal(
    overhead.fit({ weights: 0, gpuMemory: "8x10", utilization: 1.0, kvHeads: null, ...args }).maxModelLen,
    split.maxModelLen,
  );
});

test("batched tokens override", () => {
  assert.equal(overhead.batchedTokens("8x80", 2048), 2048);
});

const V = "clip.vision.";

for (const [metadata, mib] of [
  // llama.cpp `server-cuda` on an A100, with the largest image: gemma-3-4b-it, Qwen2.5-VL-7B, LFM3-VL-2B,
  // LFM2.5-VL-3B and the ultravox-v0_5-llama-3_2-1b audio encoder
  [
    {
      [V + "embedding_length"]: 1152,
      [V + "feed_forward_length"]: 4304,
      [V + "image_size"]: 896,
      [V + "patch_size"]: 14,
    },
    121.25,
  ],
  [
    {
      [V + "embedding_length"]: 1280,
      [V + "feed_forward_length"]: 3420,
      [V + "patch_size"]: 14,
      [V + "n_wa_pattern"]: 8,
      "clip.use_silu": true,
    },
    1459.7,
  ],
  [
    {
      [V + "embedding_length"]: 1024,
      [V + "feed_forward_length"]: 2560,
      [V + "patch_size"]: 16,
      [V + "spatial_merge_size"]: 2,
      [V + "image_max_pixels"]: 16777216,
      "clip.use_silu": true,
    },
    718.1,
  ],
  [
    {
      [V + "embedding_length"]: 1152,
      [V + "feed_forward_length"]: 4304,
      [V + "image_size"]: 256,
      [V + "patch_size"]: 16,
      [V + "projector.scale_factor"]: 2,
      "clip.projector_type": "lfm2",
    },
    30.31,
  ],
  [{ "clip.audio.embedding_length": 1280, "clip.audio.feed_forward_length": 5120 }, 51.27],
]) {
  test(`mmproj encoder matches llama cpp [${mib}]`, () => {
    assert.ok(Math.abs(overhead.mmprojEncoder(metadata) / (1 << 20) / mib - 1) < 0.05);
    assert.equal(overhead.llamaCpp(metadata).encoder, overhead.mmprojEncoder(metadata));
    assert.equal(overhead.llamaCpp().encoder, 0);
  });
}

test("mxfp4 experts are padded to the vllm tiles", () => {
  // gpt-oss-120b on 1 GPU (2880 -> 2944 intermediate, 2880 -> 3072 hidden) and DeepSeek-V4.1 on 8 GPUs (288 -> 384)
  const gptOss = overhead.mxfp4Padding(100, 2880, 2880);
  assert.equal(gptOss(1), Math.trunc(100 * (((2944 / 2880) * 3072) / 2880 - 1)));
  const deepseek = overhead.mxfp4Padding(300, 2304, 5120);
  assert.ok(deepseek(8) >= 99 && deepseek(8) <= 100 && deepseek(1) === 0);
  const args = { kvFixed: 0, kvPerToken: 0, maxModelLen: 0, contextLength: 0, overhead: 0 };
  const plain = overhead.fit({ weights: 60 * GIB, gpuMemory: "64", utilization: 1.0, ...args });
  const padded = overhead.fit({
    weights: 60 * GIB,
    gpuMemory: "64",
    utilization: 1.0,
    padding: () => 8 * GIB,
    ...args,
  });
  assert.ok(plain.fits && !padded.fits && padded.required === 68 * GIB);
});

test("fit searches non linear cache and replicates fixed windows", () => {
  // A sliding cache saturates at 100 pages. Both its fixed and growing parts replicate when KV heads < TP.
  const cache = (length, n) => Math.min(Math.ceil(length / 16), 100) * 1024 * n;

  const args = {
    weights: 0,
    kvFixed: 100 * 1024,
    kvPerToken: 0,
    maxModelLen: 10000,
    contextLength: 10000,
    overhead: 0,
    kvHeads: 1,
    kvBytes: cache,
    utilization: 1.0,
  };
  const fit = overhead.fit({ gpuMemory: String((80 * 1024) / GIB), ...args });
  assert.ok(!fit.fits && fit.maxModelLen === 80 * 16);
  const replicated = overhead.fit({ gpuMemory: `4x${(80 * 1024) / GIB}`, ...args });
  assert.equal(replicated.required, 4 * 100 * 1024);
  assert.equal(replicated.maxModelLen, fit.maxModelLen);
  const roomy = overhead.fit({ gpuMemory: String((100 * 1024) / GIB), ...args });
  assert.ok(roomy.fits && roomy.maxModelLen === 10000);
});

test("fit checks the requested gpu count between power of two sizes", () => {
  const result = overhead.fit({
    weights: 25 * GIB,
    kvFixed: 0,
    kvPerToken: 0,
    maxModelLen: 0,
    contextLength: 0,
    overhead: 0,
    gpuMemory: "3x10",
    utilization: 1.0,
  });
  assert.ok(result.fits && result.needed === 3);
});

test("encoder attention workspace tracks sequence boundaries", () => {
  const config = { model_type: "mpnet", hidden_size: 768, intermediate_size: 3072, num_attention_heads: 12 };
  const files = new Map();
  const single = overhead.encoderWarmup(config, files, 384, 384, 4);
  const double = overhead.encoderWarmup(config, files, 768, 384, 4);
  const partial = overhead.encoderWarmup(config, files, 512, 384, 4);
  const short = overhead.encoderWarmup(config, files, 768, 128, 4);
  assert.equal(double.activations, 2 * single.activations);
  assert.ok(single.activations < partial.activations && partial.activations < double.activations);
  assert.ok(short.activations < double.activations);
  assert.ok(double.runtime === single.runtime && single.runtime === partial.runtime);
  const fused = overhead.encoderWarmup({ ...config, _attn_implementation: "sdpa" }, files, 768, 384, 2);
  assert.ok(fused.activations < overhead.encoderWarmup(config, files, 768, 128, 2).activations);
});

test("encoder attention runs eager in f32", () => {
  const gemma = {
    model_type: "gemma3_text",
    hidden_size: 768,
    intermediate_size: 1152,
    num_attention_heads: 3,
    head_dim: 256,
    num_key_value_heads: 1,
    rms_norm_eps: 1e-6,
  };
  const half = overhead.encoderWarmup(gemma, new Map(), 2048, 2048, 2);
  const full = overhead.encoderWarmup(gemma, new Map(), 2048, 2048, 4);
  assert.ok(!overhead.encoderEager(gemma, 2) && overhead.encoderEager(gemma, 4));
  assert.ok(full.activations > 2 * half.activations);
  assert.ok(overhead.encoderEager({ model_type: "bert", _attn_implementation: "sdpa" }, 4));
  assert.ok(!overhead.encoderEager({ model_type: "mpnet", _attn_implementation: "sdpa" }, 2));
});

const MODERNBERT = {
  model_type: "modernbert",
  hidden_size: 768,
  intermediate_size: 1152,
  num_attention_heads: 12,
  num_hidden_layers: 22,
  local_attention: 128,
  global_attn_every_n_layers: 3,
  max_position_embeddings: 8192,
};

for (const [config, single, batched] of [
  [{ model_type: "bert", num_hidden_layers: 12 }, 3, 4],
  [{ model_type: "mpnet", relative_attention_num_buckets: 32 }, 4, 4],
  [{ model_type: "bert", position_embedding_type: "alibi" }, 4, 4],
  [{ model_type: "deberta-v2", relative_attention: true, pos_att_type: ["p2c", "c2p"] }, 5, 5],
  [{ model_type: "deberta-v2", relative_attention: true, pos_att_type: "p2c|c2p" }, 5, 5],
  [{ model_type: "qwen3", rms_norm_eps: 1e-6, num_hidden_layers: 28 }, 4, 4],
  [MODERNBERT, 5, 5],
  [
    {
      model_type: "gemma3_text",
      rms_norm_eps: 1e-6,
      use_bidirectional_attention: true,
      num_hidden_layers: 2,
      layer_types: ["sliding_attention", "full_attention"],
    },
    5,
    5,
  ],
]) {
  test(`encoder attention tensors follow the architecture [${JSON.stringify(config)}]`, () => {
    assert.equal(overhead.encoderAttentionTensors(config, 1), single);
    assert.equal(overhead.encoderAttentionTensors(config, 4), batched);
  });
}

test("encoder eager warmup keeps the attention tensors of each sequence", () => {
  const perToken = 7 * 768 + 5 * 1152;
  const single = overhead.encoderWarmup(MODERNBERT, new Map(), 4096, 4096, 4);
  assert.equal(single.activations, (4096 * perToken + 5 * 12 * 4096 ** 2) * 4);
  const bert = { model_type: "bert", hidden_size: 768, intermediate_size: 3072, num_attention_heads: 12 };
  const batched = overhead.encoderWarmup(bert, new Map(), 1024, 512, 4);
  assert.equal(batched.activations, (1024 * (7 * 768 + 2 * 3072) + 4 * 12 * 2 * 512 ** 2) * 4);
});

test("eager alibi keeps the bias of every position pair", () => {
  const jina = {
    model_type: "bert",
    position_embedding_type: "alibi",
    hidden_size: 512,
    intermediate_size: 2048,
    num_attention_heads: 8,
    max_position_embeddings: 8192,
  };
  const full = overhead.encoderWarmup(jina, new Map(), 128, 128, 4);
  const half = overhead.encoderWarmup(jina, new Map(), 128, 128, 2);
  assert.equal(full.buffers, 8 * 8192 ** 2 * 4);
  assert.equal(half.buffers, 0);
  assert.equal(full.bytes, full.activations + full.buffers + full.runtime);
  assert.ok(full.toDict().buffers === full.buffers && full.toDict().loading === 0);
});
