import { test } from "node:test";
import assert from "node:assert/strict";
import { appendFileSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  estimate,
  listing,
  llamaConfig,
  llamaTensors,
  tmpPath,
  writeGguf,
  writeJson,
  writeOnnx,
  writeSafetensors,
} from "./helpers.js";
import * as onnx from "../dist/onnx.js";

const ARCH = {
  "general.architecture": "llama",
  "llama.block_count": 4,
  "llama.context_length": 2048,
  "llama.embedding_length": 256,
  "llama.attention.head_count": 8,
  "llama.attention.head_count_kv": 2,
};
const TENSORS = [
  ["token_embd.weight", 12, [256, 1000]],
  ["output_norm.weight", 0, [256]],
  ["blk.0.attn_q.weight", 8, [256, 256]],
];
const TOKENIZER = { "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": Array(1000).fill(["a", "b", "c"]).flat() };

const WEIGHTS = [
  ["embed", 10, [1000, 64]],
  ["proj", 22, [64, 64]],
  ["scale", 1, [64]],
];
const PAST = [0, 1].flatMap((i) =>
  ["key", "value"].map((kind) => [`past_key_values.${i}.${kind}`, 10, ["batch", 2, "past_sequence_length", 16]]),
);
const PER_TOKEN = 4 * 2 * 16 * 2;

function some(value) {
  assert.ok(value !== null && value !== undefined);
  return value;
}

async function withSmallFile(value, fn) {
  const previous = onnx.SMALL_FILE;
  onnx.setSmallFile(value);
  try {
    return await fn();
  } finally {
    onnx.setSmallFile(previous);
  }
}

test("single file reads dtypes and the kv cache from the graph", async () => {
  const tmp = tmpPath();
  writeOnnx(join(tmp, "model.onnx"), WEIGHTS, [["input_ids", 7, ["batch", "sequence"]], ...PAST]);
  writeJson(join(tmp, "config.json"), llamaConfig({ max_position_embeddings: 1024 }));

  const result = await estimate(tmp, { batchSize: 2 });

  assert.ok(result.format === "onnx" && result.weights.params === 64000 + 4096 + 64);
  assert.deepEqual(Object.fromEntries([...result.weights.dtypes].map(([d, w]) => [d, w.bytes])), {
    F16: 128000,
    I4: 2048,
    F32: 256,
  });
  const kv = some(result.kvCache);
  assert.deepEqual([kv.dtype, kv.maxModelLen, kv.bytes], ["F16", 1024, PER_TOKEN * 1024 * 2]);
  assert.ok(result.warnings.some((w) => w.includes("ONNX Runtime overhead")));
  await assert.rejects(estimate(tmp, { dtype: "float16" }), /only supports unquantized Safetensors text encoders/);
  await assert.rejects(estimate(tmp, { kvCacheDtype: "fp8" }), /only `auto`/);
});

test("external data and large inline files", async () => {
  const tmp = tmpPath();
  const size = writeOnnx(join(tmp, "model.onnx"), WEIGHTS, PAST, "model.onnx_data");
  writeOnnx(join(tmp, "model_fp16.onnx"), WEIGHTS, PAST);
  writeOnnx(join(tmp, "model_q4.onnx"), WEIGHTS, PAST, "model_q4.onnx_data");
  appendFileSync(join(tmp, "model_q4.onnx"), Buffer.alloc(2000));

  const result = await withSmallFile(1500, () => listing(tmp, { maxModelLen: 100 }));

  const files = new Map(result.files.map((f) => [some(f.file), f]));
  assert.ok(files.get("model.onnx").weights.bytes === size && files.get("model.onnx").weights.dtypes.size);
  const large = files.get("model_fp16.onnx");
  assert.ok(large.weights.bytes === statSync(join(tmp, "model_fp16.onnx")).size && !large.weights.dtypes.size);
  assert.equal(some(large.kvCache).bytes, PER_TOKEN * 100);
  const q4 = files.get("model_q4.onnx");
  assert.equal(
    q4.weights.bytes,
    ["model_q4.onnx", "model_q4.onnx_data"].reduce((t, f) => t + statSync(join(tmp, f)).size, 0),
  );
  assert.ok(result.warnings.some((w) => w.includes("estimated from their file sizes")));
});

for (const large of [false, true]) {
  for (const prefix of ["past_key_values", "cache", "cache_params", "input.cache"]) {
    test(`transformers cache input names [${large}-${prefix}]`, async () => {
      const tmp = tmpPath();
      const inputs = [0, 1].flatMap((i) =>
        ["keys", "values"].map((kind) => [`${prefix}.layers.${i}.${kind}`, 10, ["batch", 2, "past_sequence_length", 16]]),
      );
      inputs.push(["cache_position", 7, ["sequence"]], [`${prefix}.layers.0.cumulative_length`, 7, []]);
      writeOnnx(join(tmp, "model.onnx"), WEIGHTS, inputs);
      writeJson(join(tmp, "config.json"), llamaConfig({ max_position_embeddings: null, model_max_length: 2048 }));

      const result = some(
        (await withSmallFile(large ? 1 : onnx.SMALL_FILE, () => estimate(tmp, { batchSize: 2 }))).kvCache,
      );

      assert.deepEqual([result.maxModelLen, result.dtype, result.bytes], [2048, "F16", PER_TOKEN * 2048 * 2]);
    });
  }
}

for (const large of [false, true]) {
  test(`cache position alone is not a kv cache [${large}]`, async () => {
    const tmp = tmpPath();
    writeOnnx(join(tmp, "model.onnx"), WEIGHTS, [["cache_position", 7, ["sequence"]]]);
    const result = await withSmallFile(large ? 1 : onnx.SMALL_FILE, () => estimate(tmp, { maxModelLen: 1024 }));
    assert.equal(result.kvCache, null);
  });
}

test("variants and encoder decoder parts are grouped", () => {
  const files = new Map(
    [
      "onnx/encoder_model.onnx",
      "onnx/decoder_model.onnx",
      "onnx/decoder_model_merged.onnx",
      "onnx/decoder_with_past_model.onnx",
      "onnx/encoder_model_fp16.onnx",
      "onnx/decoder_model_merged_fp16.onnx",
      "onnx/decoder_model_merged_fp16.onnx_data",
      "cuda/model.onnx",
      "cuda/model.onnx.data",
      "model_O1.onnx",
      "model.onnx",
    ].map((f) => [f, 1]),
  );
  assert.deepEqual(
    onnx.groupFiles(files),
    new Map([
      ["cuda/model.onnx", ["cuda/model.onnx", "cuda/model.onnx.data"]],
      ["model.onnx", ["model.onnx"]],
      ["model_O1.onnx", ["model_O1.onnx"]],
      ["onnx/encoder_model+decoder_model_merged", ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"]],
      [
        "onnx/encoder_model+decoder_model_merged_fp16",
        [
          "onnx/encoder_model_fp16.onnx",
          "onnx/decoder_model_merged_fp16.onnx",
          "onnx/decoder_model_merged_fp16.onnx_data",
        ],
      ],
    ]),
  );
  const groups = onnx.groupFiles(files);
  assert.equal(onnx.selectFile(groups, "cuda")[0], "cuda/model.onnx");
  assert.equal(onnx.selectFile(groups, "model.onnx")[0], "model.onnx");
  assert.throws(() => onnx.selectFile(groups, "missing.onnx"), /No ONNX file/);
});

test("onnx comes after safetensors and file sizes", async () => {
  const tmp = tmpPath();
  writeOnnx(join(tmp, "onnx", "model.onnx"), WEIGHTS, PAST);
  assert.equal((await estimate(tmp)).format, "onnx");

  writeFileSync(join(tmp, "pytorch_model.bin"), Buffer.alloc(10));
  assert.equal((await estimate(tmp)).format, "onnx");
  assert.equal((await estimate(tmp, { format: "onnx" })).format, "onnx");
  assert.equal((await estimate(tmp, { file: "onnx/model.onnx" })).format, "onnx");

  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig());
  const result = await estimate(tmp);
  assert.ok(result.format === "safetensors" && result.warnings.some((w) => w.includes("--format onnx")));
});

test("invalid onnx files", async () => {
  const tmp = tmpPath();
  writeOnnx(join(tmp, "model.onnx"), WEIGHTS);
  writeFileSync(join(tmp, "model_fp16.onnx"), Buffer.alloc(16, 0xff));

  const result = await listing(tmp);

  assert.deepEqual(
    result.files.map((f) => f.file),
    ["model.onnx"],
  );
  assert.ok(result.warnings.some((w) => w.includes("Skipping `model_fp16.onnx`")));
  await assert.rejects(estimate(tmp, { file: "model_fp16.onnx" }), /not a valid ONNX file/);
});

test("review fixes", async () => {
  const tmp = tmpPath();
  writeOnnx(
    join(tmp, "onnx", "model_q4.onnx"),
    [
      ["w_quant", 2, [64, 2, 16]],
      ["emb_quant", 2, [100, 32]],
    ],
    [],
    null,
    [
      ["MatMulNBits", ["x", "w_quant", "scales"], 4],
      ["GatherBlockQuantized", ["emb_quant", "ids"], 4],
    ],
  );
  writeOnnx(join(tmp, "onnx", "model.onnx"), WEIGHTS, PAST);
  writeJson(join(tmp, "config.json"), llamaConfig({ max_position_embeddings: 64 }));

  const q4 = await estimate(tmp, { file: "onnx/model_q4.onnx" });
  assert.ok(q4.weights.params === (64 * 2 * 16 + 100 * 32) * 2 && q4.weights.bytes === 64 * 2 * 16 + 100 * 32);
  const result = await listing(tmp);
  assert.equal(result.toDict().format, "onnx");
  await assert.rejects(estimate(tmp, { file: "onnx" }), /Multiple ONNX files/);

  writeGguf(join(tmp, "model-Q4_K_M.gguf"), { ...ARCH, ...TOKENIZER }, TENSORS);
  assert.equal((await estimate(tmp, { file: "onnx/model.onnx" })).format, "onnx");
  assert.equal((await estimate(tmp)).format, "gguf");

  unlinkSync(join(tmp, "config.json"));
  const noLength = await estimate(tmp, { file: "onnx/model.onnx" });
  assert.ok(noLength.kvCache === null && noLength.warnings.some((w) => w.includes("--max-model-len")));
});

for (const large of [false, true]) {
  for (const prefix of ["past_key_values", "cache"]) {
    test(`decoder with past and encoder caches [${large}-${prefix}]`, async () => {
      const tmp = tmpPath();
      const cross = ["decoder", "encoder"].flatMap((side) =>
        ["key", "value"].map((kind) => [
          prefix === "past_key_values"
            ? `past_key_values.0.${side}.${kind}`
            : `input.cache.${side === "encoder" ? "cross" : "self"}_attention_cache.layers.0.${kind}s`,
          1,
          ["batch", 2, `${side}_sequence_length`, 16],
        ]),
      );
      writeOnnx(join(tmp, "onnx", "encoder_model.onnx"), WEIGHTS);
      writeOnnx(join(tmp, "onnx", "decoder_model.onnx"), WEIGHTS);
      writeOnnx(join(tmp, "onnx", "decoder_with_past_model.onnx"), WEIGHTS, cross);
      writeJson(join(tmp, "config.json"), { ...llamaConfig({ max_position_embeddings: 100 }), max_source_positions: 10 });

      const result = await withSmallFile(large ? 1 : onnx.SMALL_FILE, () => estimate(tmp));

      assert.equal(result.file, "onnx/encoder_model+decoder_model+decoder_with_past_model");
      const perToken = 2 * 2 * 16 * 4;
      assert.equal(some(result.kvCache).bytes, perToken * 100 + perToken * 10);
    });
  }
}
