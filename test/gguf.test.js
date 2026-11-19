import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { estimate, listing, llamaConfig, llamaTensors, tmpPath, writeGguf, writeJson, writeSafetensors } from "./helpers.js";
import { Listing, run } from "../dist/index.js";
import * as gguf from "../dist/gguf.js";
import { LocalSource } from "../dist/source.js";

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
const TOKENIZER = {
  "tokenizer.ggml.model": "gpt2",
  "tokenizer.ggml.tokens": Array.from({ length: 1000 }, () => ["a", "b", "c"]).flat(),
};

function some(value) {
  assert.ok(value !== null && value !== undefined);
  return value;
}

test("single file", async () => {
  const tmp = tmpPath();
  writeGguf(join(tmp, "model-Q4_K_M.gguf"), { ...ARCH, ...TOKENIZER }, TENSORS);

  const result = await estimate(tmp);

  assert.equal(result.file, "model-Q4_K_M.gguf");
  assert.equal(result.weights.dtypes.get("Q4_K").bytes, ((256 * 1000) / 256) * 144);
  assert.equal(result.weights.dtypes.get("F32").bytes, 256 * 4);
  assert.equal(result.weights.dtypes.get("Q8_0").bytes, ((256 * 256) / 32) * 34);
  assert.equal(some(result.kvCache).bytes, 4 * 2 * (32 + 32) * 2048 * 2);
  assert.equal(some(result.kvCache).dtype, "F16");
  await assert.rejects(estimate(tmp, { dtype: "float16" }), /only supports unquantized Safetensors text encoders/);
});

test("listing uses file sizes and per file metadata", async () => {
  const tmp = tmpPath();
  for (const quant of ["Q4_K_M", "Q8_0"]) writeGguf(join(tmp, `model-${quant}.gguf`), { ...ARCH, ...TOKENIZER }, TENSORS);
  writeGguf(join(tmp, "mtp-model-Q4_K_M.gguf"), { ...ARCH, "llama.block_count": 5, ...TOKENIZER }, TENSORS);
  for (const i of [1, 2]) {
    writeGguf(join(tmp, `BF16/model-BF16-0000${i}-of-00002.gguf`), { ...ARCH, ...TOKENIZER }, TENSORS.slice(0, 2));
  }
  writeGguf(join(tmp, "mmproj-model-F16.gguf"), ARCH, TENSORS);

  const result = await listing(tmp, { maxModelLen: 512 });

  const files = new Map(result.files.map((f) => [some(f.file), f]));
  assert.deepEqual([...files.keys()].sort(), [
    "BF16/model-BF16.gguf",
    "model-Q4_K_M.gguf",
    "model-Q8_0.gguf",
    "mtp-model-Q4_K_M.gguf",
  ]);
  const single = await estimate(tmp, { file: "model-Q4_K_M.gguf" });
  assert.ok(Math.abs(files.get("model-Q4_K_M.gguf").weights.bytes - single.weights.bytes) < 1024);
  assert.equal(
    some(files.get("mtp-model-Q4_K_M.gguf").kvCache).bytes,
    (5 / 4) * some(files.get("model-Q4_K_M.gguf").kvCache).bytes,
  );
  const bf16 = await estimate(tmp, { file: "BF16" });
  assert.equal(bf16.components.get("Model").bytes, 2 * (((256 * 1000) / 256) * 144 + 256 * 4));
  assert.equal(bf16.weights.bytes, bf16.components.get("Model").bytes + bf16.components.get("mmproj").bytes);
});

test("kv cache variants", () => {
  assert.equal(
    some(gguf.kvCache({ ...ARCH, "llama.attention.head_count_kv": [2, 0, 2, 0] }, 100, 1, "F16"))[0],
    2 * 2 * 64 * 100 * 2,
  );

  const mla = { ...ARCH, "llama.attention.kv_lora_rank": 512, "llama.rope.dimension_count": 64 };
  assert.equal(some(gguf.kvCache(mla, 100, 1, "F16"))[0], 4 * 576 * 100 * 2);

  const hybrid = { ...ARCH, "llama.full_attention_interval": 2, "llama.nextn_predict_layers": 1 };
  assert.deepEqual(some(gguf.kvCache(hybrid, 100, 1, "Q8_0"))[1], { full: 1, sliding: 0, state: 0, none: 2 });

  const ssm = {
    ...ARCH,
    "llama.full_attention_interval": 4,
    "llama.ssm.state_size": 128,
    "llama.ssm.inner_size": 6144,
    "llama.ssm.group_count": 16,
    "llama.ssm.conv_kernel": 4,
  };
  let [size, layers] = some(gguf.kvCache(ssm, 100, 2, "F16"));
  assert.deepEqual(layers, { full: 1, sliding: 0, state: 3, none: 0 });
  assert.equal(size, 2 * (2 * 64 * 100 * 2 + 3 * 4 * (3 * (6144 + 2 * 16 * 128) + 6144 * 128)));

  const swa = {
    ...ARCH,
    "llama.attention.sliding_window": 64,
    "llama.attention.sliding_window_pattern": [true, true, true, false],
  };
  [size, layers] = some(gguf.kvCache(swa, 4096, 1, "F16"));
  assert.deepEqual(layers, { full: 1, sliding: 3, state: 0, none: 0 });
  assert.equal(size, (4096 + 3 * (64 + 512)) * 2 * 64 * 2);

  assert.equal(gguf.kvCache({ "general.architecture": "clip" }, 100, 1, "F16"), null);
});

test("unknown tensor types use data offsets", async () => {
  const path = join(tmpPath(), "model.gguf");
  writeGguf(path, ARCH, [
    ["a", 0, [64]],
    ["b", 142, [64]],
    ["c", 0, [64]],
  ]);
  const result = await estimate(path);
  assert.equal(result.weights.dtypes.get("TYPE_142").bytes, 64);
});

test("large headers are read in chunks", async (t) => {
  const saved = { ...gguf.reads };
  t.after(() => Object.assign(gguf.reads, saved));
  gguf.reads.FIRST_READ = 1024;
  gguf.reads.PREFIX_READ = 256;
  const tmp = tmpPath();
  writeGguf(join(tmp, "a-Q4_0.gguf"), { ...ARCH, ...TOKENIZER }, TENSORS);
  writeGguf(join(tmp, "b-Q4_0.gguf"), { ...TOKENIZER, ...ARCH }, TENSORS);
  const result = await listing(tmp);
  assert.deepEqual(
    result.files.map((f) => some(f.kvCache).bytes),
    [4 * 2 * 64 * 2048 * 2, 4 * 2 * 64 * 2048 * 2],
  );
});

test("invalid files", async () => {
  const tmp = tmpPath();
  writeFileSync(join(tmp, "broken-Q4_0.gguf"), Buffer.alloc(64));
  writeGguf(join(tmp, "ok-Q4_0.gguf"), ARCH, TENSORS);
  const result = await listing(tmp);
  assert.deepEqual(
    result.files.map((f) => f.file),
    ["ok-Q4_0.gguf"],
  );
  assert.ok(result.warnings.some((w) => w.includes("broken-Q4_0.gguf")));
  await assert.rejects(run(tmp, { file: "broken-Q4_0.gguf" }), /not a valid GGUF/);
  writeGguf(join(tmp, "Q8_0/a.gguf"), ARCH, TENSORS);
  writeGguf(join(tmp, "Q8_0/b.gguf"), ARCH, TENSORS);
  await assert.rejects(run(tmp, { file: "Q8_0" }), /Multiple GGUF files/);
  await assert.rejects(run(tmp, { file: "ok-Q4_0.gguf", kvCacheDtype: "bfloat16" }), /not a llama.cpp KV cache type/);
});

test("listing stops at large embedded metadata", async (t) => {
  const tmp = tmpPath();
  const embedded = {
    "general.architecture": "audio",
    "audio.embedded_files.data": Array.from({ length: 1 << 20 }, (_, i) => i),
    ...ARCH,
  };
  for (const quant of ["F16", "Q8_0"]) writeGguf(join(tmp, `audio-${quant}.gguf`), embedded, TENSORS);
  const reads = [];
  const read = LocalSource.prototype.read;
  t.after(() => {
    LocalSource.prototype.read = read;
  });
  LocalSource.prototype.read = function (path, start, end) {
    reads.push(end - start);
    return read.call(this, path, start, end);
  };
  const result = await run(tmp);

  assert.ok(result instanceof Listing);
  assert.equal(result.files.length, 2);
  assert.ok(result.files.every((f) => f.kvCache === null));
  assert.ok(reads.reduce((a, b) => a + b, 0) < 8 << 20);
});

test("listing warns that the gpu fit needs a file", async () => {
  const tmp = tmpPath();
  for (const quant of ["Q4_K_M", "Q8_0"]) writeGguf(join(tmp, `model-${quant}.gguf`), { ...ARCH, ...TOKENIZER }, TENSORS);
  const result = await listing(tmp, { gpuMemory: "24" });
  assert.ok(result.warnings.some((w) => w.includes("`--gpu-memory` is only checked")));
  await assert.rejects(listing(tmp, { gpuMemory: "lots" }), /--gpu-memory/);
});

test("format picks the weights", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig());
  writeGguf(join(tmp, "model-Q4_K_M.gguf"), { ...ARCH, ...TOKENIZER }, TENSORS);

  const def = await estimate(tmp);
  assert.ok(def.format === "safetensors" && def.warnings.some((w) => w.includes("--format gguf")));
  assert.equal((await estimate(tmp, { format: "gguf" })).format, "gguf");
  assert.equal((await run(tmp, { format: "safetensors" })).warnings.length, 0);

  unlinkSync(join(tmp, "model.safetensors"));
  assert.equal((await estimate(tmp)).format, "gguf");
  await assert.rejects(run(tmp, { format: "safetensors" }), /No Safetensors weights/);
  await assert.rejects(
    run(tmp, { format: "onnx" }),
    /No ONNX weights found .* which has GGUF weights, use `--format gguf`/,
  );
  await assert.rejects(run(tmp, { format: "pt" }), /not valid/);

  writeFileSync(join(tmp, "pytorch_model.bin"), Buffer.alloc(10));
  unlinkSync(join(tmp, "model-Q4_K_M.gguf"));
  await assert.rejects(run(tmp, { format: "safetensors" }), /only has PyTorch pickle weights/);
  await assert.rejects(run(tmp, { format: "gguf" }), /only has PyTorch pickle weights/);
});

test("quantization tags and mmproj", async () => {
  const tmp = tmpPath();
  for (const name of ["m-Q4_K_M", "m-UD-Q4_K_M", "m-BF16", "mmproj-m-BF16", "mmproj-m-Q8_0"]) {
    writeGguf(join(tmp, `${name}.gguf`), { ...ARCH, ...TOKENIZER }, TENSORS);
  }

  writeGguf(
    join(tmp, "mmproj-m-Q8_0.gguf"),
    {
      "general.architecture": "clip",
      "clip.vision.embedding_length": 64,
      "clip.vision.image_size": 224,
      "clip.vision.patch_size": 14,
    },
    TENSORS,
  );
  const q4 = await estimate(tmp, { file: "q4_k_m" });
  assert.ok(q4.file === "m-Q4_K_M.gguf");
  assert.deepEqual([...q4.components.keys()], ["Model", "mmproj"]);
  assert.equal(some(q4.overhead).encoder, Math.trunc(7.2 * 16 ** 2 * 4 * 64));
  assert.equal((await estimate(tmp, { file: "UD-Q4_K_M" })).file, "m-UD-Q4_K_M.gguf");
  const keys = (names) => new Map(names.map((n) => [n, 1]));
  assert.equal(
    gguf.mmproj(keys(["m-Q4_K_M.gguf", "mmproj-m-BF16.gguf", "mmproj-m-Q8_0.gguf"]), "m-Q4_K_M.gguf"),
    "mmproj-m-Q8_0.gguf",
  );
  assert.equal(
    gguf.mmproj(keys(["m-BF16.gguf", "mmproj-m-BF16.gguf", "mmproj-m-Q8_0.gguf"]), "m-BF16.gguf"),
    "mmproj-m-BF16.gguf",
  );
  assert.equal(gguf.mmproj(keys(["m-BF16.gguf"]), "m-BF16.gguf"), null);
  const files = new Map((await listing(tmp)).files.map((f) => [some(f.file), f]));
  assert.ok(
    files.get("m-Q4_K_M.gguf").weights.bytes >
      (await estimate(tmp, { file: "m-Q4_K_M.gguf" })).components.get("Model").bytes,
  );
  await assert.rejects(estimate(tmp, { file: "Q3_K_S" }), /No GGUF file matching/);
});

test("listing reads architecture keys after the tokenizer", async () => {
  const tmp = tmpPath();
  const entries = Object.entries(ARCH);
  const interleaved = {
    ...Object.fromEntries(entries.slice(0, 3)),
    ...TOKENIZER,
    ...Object.fromEntries(entries.slice(3)),
  };
  for (const quant of ["Q4_K_M", "Q8_0"]) writeGguf(join(tmp, `model-${quant}.gguf`), interleaved, TENSORS);

  const files = (await listing(tmp, { maxModelLen: 512 })).files;

  assert.ok(files.every((f) => f.kvCache));
  const single = await estimate(tmp, { file: "Q4_K_M", maxModelLen: 512 });
  assert.equal(some(files[0].kvCache).bytes, some(single.kvCache).bytes);
});

test("llama.cpp keeps the token embeddings on the cpu", async () => {
  const tmp = tmpPath();
  const embd = ["token_embd.weight", 1, [256, 1000]];
  const q = ["blk.0.attn_q.weight", 8, [256, 256]];
  mkdirSync(join(tmp, "untied"), { recursive: true });
  writeGguf(join(tmp, "untied", "m.gguf"), { ...ARCH, ...TOKENIZER }, [embd, ["output.weight", 1, [256, 2000]], q]);
  writeGguf(join(tmp, "tied", "m.gguf"), { ...ARCH, ...TOKENIZER }, [embd, q]);
  const [embdBytes, outputBytes, qBytes] = [256 * 1000 * 2, 256 * 2000 * 2, ((256 * 256) / 32) * 34];

  const separate = await estimate(join(tmp, "untied"));
  const tied = await estimate(join(tmp, "tied"));

  assert.equal(some(separate.offloaded).bytes, embdBytes);
  assert.equal(some(tied.offloaded).bytes, embdBytes);
  assert.equal(separate.weights.bytes, outputBytes + qBytes);
  assert.equal(tied.weights.bytes, embdBytes + qBytes);
});
