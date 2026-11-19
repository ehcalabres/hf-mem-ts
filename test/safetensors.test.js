import { test } from "node:test";
import assert from "node:assert/strict";
import { openSync, closeSync, ftruncateSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { Estimate } from "../dist/index.js";
import { kvCache } from "../dist/config.js";
import { markdown, table } from "../dist/render.js";
import { LocalSource, nodeModules } from "../dist/source.js";
import { Weights } from "../dist/weights.js";
import { estimate, kimiConfig, llamaConfig, llamaTensors, tmpPath, writeJson, writeSafetensors } from "./helpers.js";

function some(value) {
  assert.notEqual(value, null);
  assert.notEqual(value, undefined);
  return value;
}

const keys = (map) => [...map.keys()].sort();
const bytesOf = (components) => Object.fromEntries([...components].map(([n, c]) => [n, c.bytes]));
const repeat = (value, n) => Array.from({ length: n }, () => value);

test("dense model with kv cache", async () => {
  const tmp = tmpPath();
  const nbytes = writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(tmp);

  assert.ok(result instanceof Estimate);
  assert.equal(result.weights.bytes, nbytes);
  assert.deepEqual(keys(result.weights.dtypes), ["BF16", "F32"]);
  assert.notEqual(result.kvCache, null);
  assert.equal(some(result.kvCache).bytes, 2 * 4 * 2 * 32 * 1024 * 2);
  assert.equal(result.total, nbytes + some(result.kvCache).bytes + some(result.overhead).bytes);
});

test("inkling2 config estimates the decoder and local layers", async () => {
  // https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/main/config.json
  const tmp = tmpPath();
  const config = JSON.parse(
    readFileSync(join(dirname(fileURLToPath(import.meta.url)), "fixtures", "inkling2.json"), "utf-8"),
  );
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), config);

  const result = await estimate(tmp);

  const cache = some(result.kvCache);
  assert.ok(cache.maxModelLen === 1048576 && cache.dtype === "BF16");
  assert.deepEqual(cache.layers, { full: 8, sliding: 40, state: 0, none: 0 });
  const windowTokens = (Math.ceil((1024 - 1 + 2 * 8192) / 16) + 1) * 16;
  assert.equal(cache.bytes, (8 * 1048576 + 40 * windowTokens) * 2 * 8 * 128 * 2);
  assert.ok(!result.warnings.some((warning) => warning.includes("KV cache estimation skipped")));

  const override = some((await estimate(tmp, { maxModelLen: 128, batchSize: 2 })).kvCache);
  assert.equal(override.bytes, 48 * 128 * 2 * 8 * 128 * 2 * 2);
});

test("kv cache options", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(tmp, { maxModelLen: 100, batchSize: 3, kvCacheDtype: "fp8" });

  assert.equal(some(result.kvCache).bytes, 2 * 4 * 2 * 32 * 112 * 3);
  assert.equal(some(result.kvCache).dtype, "F8_E4M3");
  await assert.rejects(estimate(tmp, { kvCacheDtype: "Q4_K" }), /not a vLLM KV cache dtype/);
});

test("sharded model uses the index", async () => {
  const tmp = tmpPath();
  const tensors = Object.entries(llamaTensors());
  const first = writeSafetensors(join(tmp, "model-00001-of-00002.safetensors"), tensors.slice(0, 5));
  const second = writeSafetensors(join(tmp, "model-00002-of-00002.safetensors"), tensors.slice(5));
  writeSafetensors(join(tmp, "consolidated.safetensors"), tensors);
  writeJson(join(tmp, "model.safetensors.index.json"), {
    weight_map: { a: "model-00001-of-00002.safetensors", b: "model-00002-of-00002.safetensors" },
  });
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(tmp);

  assert.equal(result.weights.bytes, first + second);
  assert.deepEqual([...result.components.keys()], ["Transformer"]);
});

test("complete shard listing skips the index", async () => {
  const tmp = tmpPath();
  const tensors = Object.entries(llamaTensors());
  const first = writeSafetensors(join(tmp, "model-00001-of-00002.safetensors"), tensors.slice(0, 5));
  const second = writeSafetensors(join(tmp, "model-00002-of-00002.safetensors"), tensors.slice(5));
  writeFileSync(join(tmp, "model.safetensors.index.json"), "not json");
  writeJson(join(tmp, "config.json"), llamaConfig());

  assert.equal((await estimate(tmp)).weights.bytes, first + second);

  writeSafetensors(join(tmp, "model-00001-of-00003.safetensors"), tensors);
  writeJson(join(tmp, "model.safetensors.index.json"), {
    weight_map: { a: "model-00001-of-00002.safetensors", b: "model-00002-of-00002.safetensors" },
  });
  assert.equal((await estimate(tmp)).weights.bytes, first + second);
});

test("many shards are parsed in a process pool", async () => {
  const tmp = tmpPath();
  for (let i = 0; i < 4; i++) {
    const tensors = {};
    for (let e = 0; e < 4; e++) tensors[`model.layers.${i}.mlp.experts.${e}.down_proj.weight`] = ["BF16", [64, 32]];
    tensors[`model.layers.${i}.self_attn.q_proj.weight`] = ["BF16", [64, 64]];
    writeSafetensors(join(tmp, `model-0000${i + 1}-of-00004.safetensors`), tensors);
  }
  writeJson(join(tmp, "config.json"), llamaConfig({ num_local_experts: 4, num_experts_per_tok: 2 }));

  const inline = await estimate(tmp);
  const pooled = await estimate(tmp);

  assert.deepEqual(pooled.toDict(), inline.toDict());
  assert.ok(pooled.moe !== null && some(pooled.moe).experts === 4);
  assert.equal(some(pooled.moe).expert.bytes, 4 * 64 * 32 * 2);
});

test("diffusers pipeline prefers the default variant", async () => {
  const tmp = tmpPath();
  const transformer = writeSafetensors(join(tmp, "transformer/diffusion_pytorch_model.safetensors"), {
    w: ["BF16", [128, 128]],
  });
  writeSafetensors(join(tmp, "transformer/diffusion_pytorch_model.fp16.safetensors"), { w: ["F16", [128, 128]] });
  const vae = writeSafetensors(join(tmp, "vae/diffusion_pytorch_model.fp16.safetensors"), { w: ["F32", [64]] });
  writeJson(join(tmp, "model_index.json"), {
    _class_name: "Pipeline",
    transformer: ["diffusers", "T"],
    vae: ["diffusers", "V"],
    scheduler: ["d", "S"],
  });

  const result = await estimate(tmp);

  assert.deepEqual(bytesOf(result.components), { transformer, vae });
  assert.ok(result.kvCache === null && result.overhead === null);
});

test("sentence transformers adds dense modules", async () => {
  const tmp = tmpPath();
  const main = writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [100, 64]] });
  const dense = writeSafetensors(join(tmp, "2_Dense/model.safetensors"), { "linear.weight": ["F32", [64, 32]] });
  writeJson(join(tmp, "config_sentence_transformers.json"), {});
  writeJson(join(tmp, "modules.json"), [{ path: "" }, { path: "1_Pooling" }, { path: "2_Dense" }]);
  writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["Gemma3TextModel"] }));

  const result = await estimate(tmp);

  assert.deepEqual(bytesOf(result.components), { "0_Transformer": main, "2_Dense": dense });
  assert.equal(result.kvCache, null);
});

test("unindexed files are separate components", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "flux-dev.safetensors"), { w: ["BF16", [64, 64]] });
  writeSafetensors(join(tmp, "flux-schnell-00001-of-00002.safetensors"), { a: ["BF16", [32]] });
  writeSafetensors(join(tmp, "flux-schnell-00002-of-00002.safetensors"), { b: ["BF16", [32]] });

  const result = await estimate(tmp);
  assert.deepEqual(keys(result.components), ["flux-dev", "flux-schnell"]);
  assert.ok(result.warnings.some((w) => w.includes("separate component")));

  const single = await estimate(tmp, { file: "flux-schnell-00001-of-00002.safetensors" });
  assert.equal(single.weights.bytes, 128);
  await assert.rejects(estimate(tmp, { file: "missing.safetensors" }), /No Safetensors file matching/);
});

for (const [active, expected] of [
  [0, null],
  [true, null],
  [1, 1],
  [2, 2],
]) {
  for (const activeKey of ["num_experts_per_tok", "num_experts_per_token", "top_k"]) {
    test(`moe breakdown [${activeKey}-${active}-${expected}]`, async () => {
      const tmp = tmpPath();
      const tensors = { "model.embed_tokens.weight": ["BF16", [100, 64]] };
      for (let layer = 0; layer < 2; layer++) {
        for (let expert = 0; expert < 4; expert++) {
          tensors[`model.layers.${layer}.mlp.experts.${expert}.up_proj.weight`] = ["BF16", [128, 64]];
        }
      }
      writeSafetensors(join(tmp, "model.safetensors"), tensors);
      const config = llamaConfig({ num_local_experts: 4 });
      config[activeKey] = active;
      writeJson(join(tmp, "config.json"), config);

      const result = await estimate(tmp);
      const moe = result.moe;

      assert.notEqual(moe, null);
      assert.ok(table(result).includes("EXPERTS (4 x 16.38K PARAMS, 0.00 GiB)"));
      assert.deepEqual([moe.experts, moe.active], [4, expected]);
      assert.equal(result.toDict().moe.active, expected);
      assert.equal(moe.expert.bytes, 2 * 128 * 64 * 2);
      assert.equal(moe.expertsTotal.bytes, 4 * moe.expert.bytes);
      assert.equal(moe.base.bytes, 100 * 64 * 2);
    });
  }
}

test("moe breakdown skipped for non uniform experts", async () => {
  const tmp = tmpPath();
  const tensors = {
    "model.layers.0.mlp.experts.0.w.weight": ["BF16", [128, 64]],
    "model.layers.0.mlp.experts.1.w.weight": ["BF16", [64, 64]],
  };
  writeSafetensors(join(tmp, "model.safetensors"), tensors);
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(tmp);

  assert.equal(result.moe, null);
  assert.ok(result.warnings.some((w) => w.includes("not uniform")));
});

for (const pooled of [false, true]) {
  for (const nested of [false, true]) {
    test(`mxfp4 parameters include both packed values [${nested}-${pooled}]`, async () => {
      const tmp = tmpPath();
      const quantization = { quant_method: "compressed-tensors", format: "mxfp4-pack-quantized" };
      const config = llamaConfig({ num_local_experts: 2, num_experts_per_tok: 1, quantization_config: quantization });
      writeJson(
        join(tmp, "config.json"),
        nested ? { architectures: ["KimiK31ForConditionalGeneration"], text_config: config } : config,
      );
      const firstTensors = { "model.embed_tokens.weight": ["BF16", [100, 64]], "model.buffer": ["U8", [8]] };
      for (let e = 0; e < 2; e++) firstTensors[`model.layers.0.mlp.experts.${e}.w1.weight_packed`] = ["U8", [32, 16]];
      const first = writeSafetensors(join(tmp, "model-00001-of-00002.safetensors"), firstTensors);
      const secondTensors = {};
      for (let e = 0; e < 2; e++) secondTensors[`model.layers.0.mlp.experts.${e}.w1.weight_scale`] = ["U8", [32, 1]];
      const second = writeSafetensors(join(tmp, "model-00002-of-00002.safetensors"), secondTensors);

      const result = await estimate(tmp);

      assert.ok(result.weights.bytes === first + second && first + second === 13896);
      assert.equal(result.weights.params, 8456);
      assert.equal(result.weights.dtypes.get("MXFP4").params, 2048);
      assert.equal(result.weights.dtypes.get("MXFP4").bytes, 1088);
      assert.ok(result.weights.dtypes.get("U8").params === result.weights.dtypes.get("U8").bytes);
      assert.equal(result.weights.dtypes.get("U8").bytes, 8);
      assert.equal(some(result.moe).expert.params, 1024);
      assert.equal(some(result.moe).expert.bytes, 544);
      assert.equal(some(result.moe).base.params, 6408);
      assert.ok(table(result).includes("MXFP4"));
    });
  }
}

for (const quantization of [{}, { format: "pack-quantized" }]) {
  test(`u8 tensors are not assumed to be mxfp4 [${JSON.stringify(quantization)}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), {
      "layer.weight_packed": ["U8", [32, 16]],
      "layer.weight_scale": ["U8", [32, 1]],
    });
    writeJson(join(tmp, "config.json"), llamaConfig({ quantization_config: quantization }));

    const result = await estimate(tmp);

    assert.ok(result.weights.params === result.weights.bytes && result.weights.bytes === 544);
    assert.deepEqual(keys(result.weights.dtypes), ["U8"]);
  });
}

test("gpt oss mxfp4 blocks count both values", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), {
    "model.embed_tokens.weight": ["BF16", [16, 32]],
    "model.layers.0.mlp.experts.gate_up_proj_blocks": ["U8", [2, 32, 1, 16]],
    "model.layers.0.mlp.experts.gate_up_proj_scales": ["U8", [2, 32, 1]],
    "model.layers.0.mlp.experts.gate_up_proj_bias": ["BF16", [2, 32]],
    "model.buffer": ["U8", [5]],
  });
  writeJson(join(tmp, "config.json"), llamaConfig({ quantization_config: { quant_method: "mxfp4" } }));

  const result = await estimate(tmp);

  assert.equal(result.weights.params, 2629);
  assert.equal(result.weights.bytes, 2245);
  assert.equal(result.weights.dtypes.get("MXFP4").params, 2048);
  assert.equal(result.weights.dtypes.get("MXFP4").bytes, 1088);
  assert.equal(result.weights.dtypes.get("BF16").params, 576);
  assert.equal(result.weights.dtypes.get("U8").params, 5);
});

for (const [expertDtype, params] of [
  ["fp4", 1024],
  ["fp8", 512],
]) {
  test(`deepseek fp4 experts keep the fp8 weights [${expertDtype}-${params}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), {
      "layers.0.ffn.experts.0.w1.weight": ["I8", [32, 16]],
      "layers.0.ffn.experts.0.w1.scale": ["F8_E8M0", [32, 1]],
      "layers.0.attn.wq.weight": ["I8", [16, 32]],
      "layers.0.attn.wq.scale": ["F8_E8M0", [1, 1]],
    });
    const quantization = { quant_method: "fp8", expert_dtype: expertDtype };
    writeJson(join(tmp, "config.json"), llamaConfig({ quantization_config: quantization }));

    const result = await estimate(tmp);

    assert.equal(result.weights.bytes, 1057);
    assert.equal(result.weights.params, params + 512);
    assert.equal(some(result.moe).expert.params, params);
    assert.equal(some(result.moe).expert.bytes, 544);
    assert.equal(some(result.moe).base.params, 512);
    assert.equal(some(result.moe).base.bytes, 513);
  });
}

test("fp8 scales are storage not parameters", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), {
    "model.layers.0.attn.weight": ["F8_E4M3", [32, 64]],
    "model.layers.0.attn.weight_scale_inv": ["F32", [1, 2]],
    "model.layers.0.norm.scale": ["F32", [32]],
    "model.layers.1.engram.embed.weight": ["F8_E4M3", [8, 8]],
    "model.layers.1.engram.embed.scale": ["F8_E8M0", [1, 1]],
  });
  writeJson(join(tmp, "config.json"), llamaConfig({ quantization_config: { quant_method: "fp8" } }));

  const result = await estimate(tmp);

  assert.equal(result.weights.params, 2080);
  assert.equal(result.weights.bytes, 2184);
  assert.equal(result.weights.dtypes.get("F32").params, 32);
  assert.equal(result.weights.dtypes.get("F32").bytes, 136);
  assert.equal(some(result.offloaded).params, 64);
  assert.equal(some(result.offloaded).bytes, 65);
});

test("pickle weights are not estimated", async () => {
  const tmp = tmpPath();
  for (const [name, size] of Object.entries({ "pytorch_model.bin": 1000, "optimizer.pt": 99, "README.md": 5 })) {
    const fd = openSync(join(tmp, name), "w");
    ftruncateSync(fd, size);
    closeSync(fd);
  }

  await assert.rejects(estimate(tmp), /only has PyTorch pickle weights \(e.g. `pytorch_model.bin`\)/);
});

test("no weights", async () => {
  const tmp = tmpPath();
  writeJson(join(tmp, "config.json"), llamaConfig());
  await assert.rejects(estimate(tmp), /No Safetensors, GGUF or ONNX weights/);
});

test("local file reads the sibling config", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(join(tmp, "model.safetensors"));

  assert.equal(result.file, "model.safetensors");
  assert.notEqual(result.kvCache, null);
});

test("mistral params json warns", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "consolidated.safetensors"), llamaTensors());
  writeJson(join(tmp, "params.json"), { dim: 64, n_layers: 4 });

  const result = await estimate(tmp);

  assert.equal(result.kvCache, null);
  assert.ok(result.warnings.some((w) => w.includes("params.json")));
});

test("max num batched tokens selects the vllm profile", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig({ intermediate_size: 128 }));

  const fallback = some((await estimate(tmp)).overhead);
  const small = some((await estimate(tmp, { maxNumBatchedTokens: 2048 })).overhead);

  assert.ok(fallback.activations > small.activations);
  assert.ok(fallback.runtime > small.runtime);
});

test("vision encoder overhead", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  const vision = { hidden_size: 64, intermediate_size: 128, patch_size: 14, image_size: 896 };
  const config = {
    architectures: ["LlavaForConditionalGeneration"],
    text_config: llamaConfig(),
    vision_config: vision,
  };
  writeJson(join(tmp, "config.json"), config);
  writeJson(join(tmp, "preprocessor_config.json"), { max_pixels: 14 * 14 * 1024 });

  const overhead = some((await estimate(tmp)).overhead);

  assert.equal(overhead.encoder, 1024 * 2 * (4 * 64 + 2 * 128));
});

test("video models warn about the vllm video profile", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  const vision = { hidden_size: 64, intermediate_size: 128, patch_size: 14, temporal_patch_size: 2 };
  const config = {
    architectures: ["Qwen2VLForConditionalGeneration"],
    text_config: llamaConfig(),
    vision_config: vision,
  };
  writeJson(join(tmp, "config.json"), config);

  assert.ok((await estimate(tmp)).warnings.some((w) => w.includes("largest video")));
});

test("index with missing files uses the uploaded ones", async () => {
  const tmp = tmpPath();
  const tensors = Object.entries(llamaTensors());
  const first = writeSafetensors(join(tmp, "model-00001-of-00003.safetensors"), tensors.slice(0, 5));
  const weightMap = Object.fromEntries([1, 2, 3].map((i) => [`t${i}`, `model-0000${i}-of-00003.safetensors`]));
  writeJson(join(tmp, "model.safetensors.index.json"), { weight_map: weightMap });
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(tmp);

  assert.equal(result.weights.bytes, first);
  assert.ok(result.warnings.some((w) => w.includes("2 files that are not in the repository")));
});

test("invalid files are skipped in multi file repositories", async () => {
  const tmp = tmpPath();
  const good = writeSafetensors(join(tmp, "lora-a.safetensors"), { w: ["BF16", [64, 64]] });
  writeFileSync(join(tmp, "lora-b.safetensors"), Buffer.from("\x10\0\0\0\0\0\0\0not json", "latin1"));

  const result = await estimate(tmp);

  assert.equal(result.weights.bytes, good);
  assert.ok(result.warnings.some((w) => w.includes("Skipping 1 invalid")));
  await assert.rejects(estimate(tmp, { file: "lora-b.safetensors" }), /not a valid Safetensors/);
});

test("80 gib gpus warn about a100", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig());

  assert.ok((await estimate(tmp, { gpuMemory: "80" })).warnings.some((w) => w.includes("A100")));
  assert.ok(!(await estimate(tmp, { gpuMemory: "141" })).warnings.some((w) => w.includes("A100")));
});

test("malformed and oversized headers are skipped", async () => {
  const tmp = tmpPath();
  const good = writeSafetensors(join(tmp, "lora-a.safetensors"), { w: ["BF16", [64, 64]] });
  writeFileSync(join(tmp, "lora-b.safetensors"), Buffer.from('\x0b\0\0\0\0\0\0\0{"w": {}}', "latin1"));
  writeFileSync(join(tmp, "lora-c.safetensors"), Buffer.concat([Buffer.alloc(8, 0xff), Buffer.from("{}")]));

  const result = await estimate(tmp);

  assert.equal(result.weights.bytes, good);
  assert.ok(result.warnings.some((w) => w.includes("Skipping 2 invalid")));
});

test("file prefers the exact path over the basename", async () => {
  const tmp = tmpPath();
  const root = writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeSafetensors(join(tmp, "other", "model.safetensors"), { w: ["BF16", [64, 64]] });
  writeJson(join(tmp, "config.json"), llamaConfig());

  assert.equal((await estimate(tmp, { file: "model.safetensors" })).weights.bytes, root);
  assert.equal((await estimate(join(tmp, "model.safetensors"))).weights.bytes, root);
});

test("local file lists only its folder", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeSafetensors(join(tmp, "nested", "deep", "model.safetensors"), llamaTensors());
  const source = LocalSource.open(some(await nodeModules()), tmp, null, null, false);
  assert.deepEqual([...source.files.keys()], ["model.safetensors"]);
});

test("fit is not capped below the requested length", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig({ max_position_embeddings: 1024 }));

  const result = await estimate(tmp, { maxModelLen: 4096, gpuMemory: "80" });

  assert.ok(some(result.fit).fits && some(result.fit).maxModelLen >= 4096);
});

test("audio tokenizers are added as components", async () => {
  const tmp = tmpPath();
  const model = writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  const tokenizer = writeSafetensors(join(tmp, "audio_tokenizer", "model.safetensors"), { w: ["BF16", [64, 64]] });
  writeSafetensors(join(tmp, "variant", "model.safetensors"), { w: ["BF16", [64, 64]] });
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(tmp);

  assert.deepEqual([...result.components.keys()], ["Transformer", "audio_tokenizer"]);
  assert.ok(result.weights.bytes === model + tokenizer && result.kvCache !== null);
});

test("engram and ngram tables are offloaded to cpu", async () => {
  const tmp = tmpPath();
  const tensors = {
    ...llamaTensors(),
    "layers.1.engram.embed.weight": ["F8_E4M3", [1000, 64]],
    "layers.1.engram.embed.scale": ["F8_E4M3", [1000, 2]],
    "layers.1.engram.wkv.weight": ["BF16", [64, 64]],
    "model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight": ["BF16", [500, 64]],
  };
  writeSafetensors(join(tmp, "model.safetensors"), tensors);
  writeJson(join(tmp, "config.json"), llamaConfig());

  const result = await estimate(tmp);

  const offloaded = some(result.offloaded);
  assert.equal(offloaded.bytes, 1000 * 64 + 1000 * 2 + 500 * 64 * 2);
  assert.ok(
    result.weights.dtypes.has("BF16") &&
      result.weights.bytes ===
        writeSafetensors(join(tmp, "gpu", "model.safetensors"), {
          ...llamaTensors(),
          "layers.1.engram.wkv.weight": ["BF16", [64, 64]],
        }),
  );
  assert.ok(result.warnings.some((w) => w.includes("pinned host memory")));
  assert.equal(result.toDict().offloaded.bytes, offloaded.bytes);
});

test("suggested context fits the actual cache pages", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig({ max_position_embeddings: 1000000 }));
  const result = await estimate(tmp, { gpuMemory: "2" });
  const length = some(result.fit).maxModelLen;
  assert.ok(0 < length && length < 1000000);
  const suggested = await estimate(tmp, { gpuMemory: "2", maxModelLen: length });
  assert.ok(some(suggested.fit).fits);
  assert.ok(some(suggested.fit).required <= some(suggested.fit).usable);
  const longer = await estimate(tmp, { gpuMemory: "2", maxModelLen: length + 1 });
  assert.ok(some(longer.fit).required > some(longer.fit).usable);
});

test("suggested context for saturated sliding cache", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(
    join(tmp, "config.json"),
    llamaConfig({
      max_position_embeddings: 100000,
      layer_types: repeat("sliding_attention", 4),
      sliding_window: 128,
    }),
  );
  const result = await estimate(tmp, { gpuMemory: "2" });
  assert.ok(some(result.fit).fits);
  assert.equal(some(result.fit).maxModelLen, 100000);
});

for (const gpuMemory of ["4", "2x4", "8x4"]) {
  test(`kimi fit uses the cache rows for each tensor parallel rank [${gpuMemory}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
    const config = kimiConfig();
    writeJson(join(tmp, "config.json"), config);
    const result = await estimate(tmp, { gpuMemory });
    const cache = some(result.kvCache);
    const fit = some(result.fit);
    const ranks = fit.gpus || 1;
    const expected = kvCache(config, cache.dtype, cache.maxModelLen, 1, undefined, ranks)[0];
    assert.equal(fit.required, result.weights.bytes + expected + ranks * some(result.overhead).bytes);
    assert.equal(cache.dtype, "FP8_DS_MLA");
    assert.ok(0 < fit.maxModelLen && fit.maxModelLen < cache.maxModelLen);
    const suggested = await estimate(tmp, { gpuMemory, maxModelLen: fit.maxModelLen });
    assert.ok(some(suggested.fit).fits);
    const longer = await estimate(tmp, { gpuMemory, maxModelLen: fit.maxModelLen + 1 });
    assert.ok(!some(longer.fit).fits);
    if (ranks === 8) {
      assert.equal(some(longer.fit).needed, 4);
      assert.ok(table(longer).includes("DOES NOT FIT"));
      assert.ok(markdown(longer).includes("fits on 4 GPUs"));
    }
  });
}

for (const [dtype, size] of [
  ["F32", 4],
  ["BF16", 2],
  ["F16", 2],
]) {
  test(`encoder warmup keeps weights and scales with tokens [${dtype}-${size}]`, async () => {
    const tmp = tmpPath();
    const stored = writeSafetensors(join(tmp, "model.safetensors"), {
      w: [dtype, [1000, 64]],
      "norm.weight": ["F32", [64]],
      position_ids: ["I64", [512]],
    });
    writeJson(
      join(tmp, "config.json"),
      llamaConfig({ architectures: ["BertModel"], model_type: "bert", torch_dtype: null }),
    );
    const small = await estimate(tmp, { maxNumBatchedTokens: 512 });
    const large = await estimate(tmp, { maxNumBatchedTokens: 2048 });
    assert.ok(small.weights.bytes === large.weights.bytes && large.weights.bytes === stored);
    assert.deepEqual(keys(small.weights.dtypes), [...new Set([dtype, "F32", "I64"])].sort());
    assert.ok(small.kvCache === null && large.kvCache === null);
    const perToken = 7 * 256 + 2 * 1024;
    if (size === 2) {
      assert.equal(some(large.overhead).activations, 4 * some(small.overhead).activations);
      assert.equal(Math.floor(some(small.overhead).activations / size), 512 * perToken);
    } else {
      assert.equal(Math.floor(some(small.overhead).activations / size), 512 * perToken + 3 * 8 * 512 ** 2);
      assert.equal(Math.floor(some(large.overhead).activations / size), 2048 * perToken + 4 * 8 * 2 * 1024 ** 2);
    }
    assert.equal(small.total, small.weights.bytes + some(small.overhead).bytes);
    assert.ok(small.warnings.some((w) => w.includes("Weights are estimated as stored")));
  });
}

test("encoder f32 activations add the attention tensors", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [1000, 64]] });
  const config = llamaConfig({ architectures: ["BertModel"], model_type: "bert", torch_dtype: "float32" });
  writeJson(join(tmp, "config.json"), config);
  const single = await estimate(tmp, { maxNumBatchedTokens: 512 });
  writeJson(join(tmp, "config.json"), { ...config, torch_dtype: "float16" });
  const half = await estimate(tmp, { maxNumBatchedTokens: 512 });
  assert.deepEqual(half.weights.toDict(), single.weights.toDict());
  const perToken = 7 * 256 + 2 * 1024;
  assert.equal(some(half.overhead).activations, 512 * perToken * 2);
  assert.equal(some(single.overhead).activations, (512 * perToken + 3 * 8 * 512 ** 2) * 4);
  assert.ok(single.warnings.some((w) => w.includes("eager attention") && w.includes("half precision")));
  assert.ok(half.warnings.some((w) => w.includes("memory-efficient attention")));
});

test("f32 modernbert keeps both attention masks and does not fit", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [149_014_272]] });
  const config = {
    architectures: ["ModernBertModel"],
    model_type: "modernbert",
    hidden_size: 768,
    intermediate_size: 1152,
    num_attention_heads: 12,
    num_hidden_layers: 22,
    local_attention: 128,
    global_attn_every_n_layers: 3,
    max_position_embeddings: 8192,
    torch_dtype: "float32",
  };
  writeJson(join(tmp, "config.json"), config);
  const options = { maxNumBatchedTokens: 4096, maxModelLen: 4096, gpuMemory: "4.6" };
  const full = await estimate(tmp, { dtype: "float32", ...options });
  const half = await estimate(tmp, { dtype: "float16", ...options });
  assert.equal(some(full.overhead).activations, (4096 * (7 * 768 + 5 * 1152) + 5 * 12 * 4096 ** 2) * 4);
  assert.ok(!some(full.fit).fits && some(half.fit).fits);
  assert.ok(full.warnings.some((w) => w.includes("eager attention")));
});

test("f16 override counts the loading peak of the largest tensor", async () => {
  const tmp = tmpPath();
  const embeddings = 250002 * 768;
  writeSafetensors(join(tmp, "model.safetensors"), {
    "embeddings.word_embeddings.weight": ["F32", [250002, 768]],
    "encoder.w": ["F32", [86_000_000]],
  });
  writeJson(
    join(tmp, "config.json"),
    llamaConfig({
      architectures: ["XLMRobertaModel"],
      model_type: "xlm-roberta",
      hidden_size: 768,
      intermediate_size: 3072,
      num_attention_heads: 12,
      max_position_embeddings: 514,
      pad_token_id: 1,
      torch_dtype: "float32",
    }),
  );
  writeJson(join(tmp, "config_sentence_transformers.json"), {});
  const cast = await estimate(tmp, { dtype: "float16", maxNumBatchedTokens: 128 });
  const stored = await estimate(tmp, { maxNumBatchedTokens: 128 });
  const overhead = some(cast.overhead);
  assert.ok(overhead.loading > 0);
  assert.equal(cast.total, overhead.runtime + cast.weights.bytes + embeddings * 4);
  assert.equal(some(stored.overhead).loading, 0);
  assert.ok(table(cast).includes("LOADING"));
  assert.ok(cast.warnings.some((w) => w.includes("while loading")));
});

test("f32 alibi keeps the bias table", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [1000, 64]] });
  writeJson(
    join(tmp, "config.json"),
    llamaConfig({
      architectures: ["JinaBertForMaskedLM"],
      model_type: "bert",
      position_embedding_type: "alibi",
      max_position_embeddings: 8192,
      torch_dtype: "float32",
    }),
  );
  writeJson(join(tmp, "config_sentence_transformers.json"), {});
  const full = await estimate(tmp, { maxNumBatchedTokens: 128 });
  const half = await estimate(tmp, { maxNumBatchedTokens: 128, dtype: "float16" });
  assert.equal(some(full.overhead).buffers, 8 * 8192 ** 2 * 4);
  assert.equal(some(half.overhead).buffers, 0);
  assert.equal(full.total, full.weights.bytes + some(full.overhead).bytes);
  assert.ok(table(full).includes("BUFFERS"));
  assert.ok(full.warnings.some((w) => w.includes("ALiBi")));
});

test("encoder eager warmup uses all tokens and sequence override", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [1000, 64]] });
  writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["MPNetModel"], model_type: "mpnet" }));
  writeJson(join(tmp, "config_sentence_transformers.json"), {});
  writeJson(join(tmp, "sentence_bert_config.json"), { max_seq_length: 384 });
  const original = await estimate(tmp, { maxNumBatchedTokens: 768 });
  const shorter = await estimate(tmp, { maxNumBatchedTokens: 768, maxModelLen: 128 });
  const bigger = await estimate(tmp, { maxNumBatchedTokens: 1536 });
  assert.ok(some(shorter.overhead).activations < some(original.overhead).activations);
  assert.equal(some(bigger.overhead).activations, 2 * some(original.overhead).activations);
  assert.ok(shorter.warnings.some((w) => w.includes("up to 128 tokens")));
  assert.ok(original.weights.bytes === shorter.weights.bytes && shorter.weights.bytes === bigger.weights.bytes);
});

test("encoder runtime and replica fit", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["BF16", [1024, 1024, 2048]] });
  writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["BertModel"], model_type: "bert" }));
  const result = await estimate(tmp, { gpuMemory: "2x3" });
  assert.deepEqual([some(result.fit).gpus, some(result.fit).needed, some(result.fit).fits], [1, null, false]);
  assert.ok(result.warnings.some((w) => w.includes("one replica per GPU")));
  assert.ok(some(result.overhead).runtime > 0);
  assert.equal(some(result.fit).required, result.total);
});

test("sentence transformers decoder uses forward overhead", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig({ hidden_act: "silu", rms_norm_eps: 1e-6 }));
  writeJson(join(tmp, "config_sentence_transformers.json"), {});
  writeJson(join(tmp, "generation_config.json"), {});
  const result = await estimate(tmp, { maxNumBatchedTokens: 512 });
  assert.ok(result.kvCache === null && some(result.overhead).activations > 0);
});

test("masked language model counts logits unless embedding pooling", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [1000, 64]] });
  writeJson(
    join(tmp, "config.json"),
    llamaConfig({ architectures: ["BertForMaskedLM"], model_type: "bert", vocab_size: 30522 }),
  );
  const masked = await estimate(tmp, { maxNumBatchedTokens: 512 });
  writeJson(join(tmp, "config_sentence_transformers.json"), {});
  const pooled = await estimate(tmp, { maxNumBatchedTokens: 512 });
  writeJson(join(tmp, "document_1_SpladePooling/config.json"), {});
  const sparse = await estimate(tmp, { maxNumBatchedTokens: 512 });
  assert.ok(
    some(sparse.overhead).activations > some(masked.overhead).activations &&
      some(masked.overhead).activations > some(pooled.overhead).activations,
  );
});

for (const extra of [
  { is_encoder_decoder: true },
  { vision_config: { hidden_size: 64 } },
  { audio_config: { hidden_size: 64 } },
]) {
  test(`other models do not take the encoder path [${JSON.stringify(extra)}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
    writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["T5Model"], ...extra }));
    writeJson(join(tmp, "config_sentence_transformers.json"), {});
    const result = await estimate(tmp);
    assert.equal(result.overhead, null);
    assert.ok(!result.warnings.some((w) => w.includes("Encoder warmup")));
  });
}

test("unknown encoder length skips warmup until overridden", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["BF16", [1000, 64]] });
  writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["BertModel"], max_position_embeddings: null }));
  const result = await estimate(tmp);
  assert.ok(result.overhead === null && result.warnings.some((w) => w.includes("sequence length is unknown")));
  assert.notEqual((await estimate(tmp, { maxModelLen: 512 })).overhead, null);
});

for (const extra of [{ num_experts: 8 }, { layer_types: repeat("mamba", 4) }, { kv_lora_rank: 64 }]) {
  test(`encoder warmup skips unmodeled architectures [${JSON.stringify(extra)}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
    writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["LlamaForSequenceClassification"], ...extra }));
    const result = await estimate(tmp);
    assert.ok(
      result.overhead === null && result.warnings.some((w) => w.includes("Encoder warmup estimation skipped")),
    );
  });
}

for (const [dtype, target, size] of [
  ["float16", "F16", 2],
  ["float32", "F32", 4],
  ["bfloat16", "BF16", 2],
]) {
  test(`encoder precision override keeps components and buffer storage [${dtype}-${target}-${size}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), {
      w: ["F32", [1000, 64]],
      "norm.weight": ["F32", [64]],
      position_ids: ["I64", [512]],
    });
    writeSafetensors(join(tmp, "2_Dense/model.safetensors"), { "linear.weight": ["BF16", [64, 32]] });
    writeJson(
      join(tmp, "config.json"),
      llamaConfig({ architectures: ["BertModel"], model_type: "bert", torch_dtype: "float32" }),
    );
    writeJson(join(tmp, "config_sentence_transformers.json"), {});
    writeJson(join(tmp, "modules.json"), [{ path: "" }, { path: "1_Pooling" }, { path: "2_Dense" }]);
    const before = await estimate(tmp, { maxNumBatchedTokens: 768, maxModelLen: 384 });
    const result = await estimate(tmp, { maxNumBatchedTokens: 768, maxModelLen: 384, dtype });
    assert.equal(result.weights.params, before.weights.params);
    assert.equal(result.weights.bytes, (64000 + 64 + 2048) * size + 512 * 8);
    assert.deepEqual(keys(result.weights.dtypes), [target, "I64"].sort());
    assert.deepEqual(result.weights.dtypes.get("I64").toDict(), before.weights.dtypes.get("I64").toDict());
    assert.deepEqual(bytesOf(result.components), {
      "0_Transformer": (64000 + 64) * size + 512 * 8,
      "2_Dense": 2048 * size,
    });
    const perToken = 7 * 256 + 2 * 1024;
    const attention = 4 * 8 * 2 * 384 ** 2;
    assert.equal(some(before.overhead).activations, (768 * perToken + attention) * 4);
    assert.equal(some(result.overhead).activations, (768 * perToken + (size === 4 ? attention : 0)) * size);
    assert.equal(result.total, result.weights.bytes + some(result.overhead).bytes);
    assert.deepEqual(
      (await estimate(tmp, { maxNumBatchedTokens: 768, maxModelLen: 384 })).weights.toDict(),
      before.weights.toDict(),
    );
  });
}

for (const modelType of ["bert", "mpnet"]) {
  test(`encoder dtype override keeps attention and warmup shape [${modelType}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [1000, 64]] });
    writeJson(
      join(tmp, "config.json"),
      llamaConfig({ architectures: ["BertModel"], model_type: modelType, torch_dtype: "float32" }),
    );
    const half = await estimate(tmp, { maxNumBatchedTokens: 768, maxModelLen: 384, dtype: "float16" });
    const full = await estimate(tmp, { maxNumBatchedTokens: 768, maxModelLen: 384, dtype: "float32" });
    if (modelType === "mpnet") {
      assert.equal(some(full.overhead).activations, 2 * some(half.overhead).activations);
    } else {
      assert.ok(some(full.overhead).activations > 2 * some(half.overhead).activations);
    }
    assert.equal(some(full.overhead).runtime, some(half.overhead).runtime);
    assert.equal(half.warnings.join(" ").includes("eager attention"), modelType === "mpnet");
    assert.ok(full.warnings.join(" ").includes("eager attention"));
  });
}

for (const config of [
  llamaConfig(),
  llamaConfig({ architectures: ["BertModel"], quantization_config: { quant_method: "fp8" } }),
  {
    architectures: ["LlavaForConditionalGeneration"],
    text_config: llamaConfig(),
    vision_config: { hidden_size: 64 },
  },
]) {
  test(`dtype override rejects other workloads without changing defaults [${config.architectures}]`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
    writeJson(join(tmp, "config.json"), config);
    const before = (await estimate(tmp)).toDict();
    await assert.rejects(estimate(tmp, { dtype: "float16" }), /only supports unquantized Safetensors text encoders/);
    assert.deepEqual((await estimate(tmp)).toDict(), before);
  });
}

test("precision cast preserves non parameter bytes without mutating weights", () => {
  const weights = new Weights();
  weights.add("F32", 0, 4096);
  weights.add("BF16", 1024, 2048);
  weights.add("U8", 512, 512);
  const before = weights.toDict();
  const result = weights.cast("float16");
  assert.ok(result.bytes === weights.bytes && result.params === weights.params);
  assert.ok(result.dtypes.get("F32").bytes === 4096 && result.dtypes.get("F32").params === 0);
  assert.ok(result.dtypes.get("F16").bytes === 2048 && result.dtypes.get("U8").bytes === 512);
  assert.deepEqual(weights.toDict(), before);
});

test("fp32 decoder reference casts weights and preserves integer buffers", async () => {
  const tmp = tmpPath();
  const stored = writeSafetensors(join(tmp, "model.safetensors"), {
    "model.embed_tokens.weight": ["F32", [100, 64]],
    position_ids: ["I64", [3]],
  });
  writeJson(join(tmp, "config.json"), llamaConfig({ torch_dtype: "float32" }));
  const result = await estimate(tmp);
  assert.equal(result.weights.bytes, Math.floor((stored - 24) / 2) + 24);
  assert.deepEqual(keys(result.weights.dtypes), ["F16", "I64"]);
  assert.equal(some(result.kvCache).dtype, "F16");
  assert.ok(result.warnings.some((warning) => warning.includes("16-bit CUDA serving reference")));
});

test("fp32 expert storage stays consistent with the decoder reference", async () => {
  const tmp = tmpPath();
  const tensors = { "model.embed_tokens.weight": ["F32", [100, 64]] };
  for (let index = 0; index < 4; index++) {
    tensors[`model.layers.0.mlp.experts.${index}.up_proj.weight`] = ["F32", [128, 64]];
  }
  const stored = writeSafetensors(join(tmp, "model.safetensors"), tensors);
  writeJson(
    join(tmp, "config.json"),
    llamaConfig({ torch_dtype: "float32", num_local_experts: 4, num_experts_per_tok: 1 }),
  );
  const result = await estimate(tmp);
  const moe = some(result.moe);
  assert.equal(result.weights.bytes, Math.floor(stored / 2));
  assert.equal(moe.base.bytes + moe.expertsTotal.bytes, result.weights.bytes);
  assert.equal(moe.active, 1);
});
