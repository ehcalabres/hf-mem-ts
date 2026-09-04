import assert from "node:assert/strict";
import test from "node:test";

import { formatResult } from "../dist/report.js";
import { estimateGgufKvCache } from "../dist/index.js";


test("shows Diffusers components as model sub-rows", () => {
  const result = {
    modelId: "org/diffusers",
    revision: "main",
    resolvedRevision: "c".repeat(40),
    format: "safetensors",
    filename: null,
    weightsBytes: 15 * 2 ** 30,
    kvCacheBytes: null,
    totalBytes: 15 * 2 ** 30,
    files: {
      safetensors: {
        parameters: 0,
        bytes: 15 * 2 ** 30,
        components: {
          vae: { parameters: 0, bytes: 1 * 2 ** 30, dtypes: {} },
          text_encoder: { parameters: 0, bytes: 8 * 2 ** 30, dtypes: {} },
          transformer: { parameters: 0, bytes: 6 * 2 ** 30, dtypes: {} },
        },
        kvCache: null,
      },
    },
    mmproj: null,
    draft: null,
  };

  const output = formatResult(result);
  assert.match(output, /Model:\s+15\.00 GiB/);
  assert.match(output, /  VAE:\s+1\.00 GiB/);
  assert.match(output, /  TEXT ENCODER:\s+8\.00 GiB/);
  assert.match(output, /  TRANSFORMER:\s+6\.00 GiB/);
});

test("identifies unresolved draft quantizations and their selection flag", () => {
  const output = formatResult({
    modelId: "org/target", revision: "main", format: "safetensors", filename: null,
    weightsBytes: 2 ** 30, kvCacheBytes: null, totalBytes: null, files: {}, mmproj: null,
    draft: {
      modelId: "org/draft", revision: "v2", format: "gguf", filename: null,
      weightsBytes: { "draft-Q4.gguf": 2 ** 30, "draft-Q8.gguf": 2 * 2 ** 30 },
      kvCacheBytes: { "draft-Q4.gguf": 0.5 * 2 ** 30, "draft-Q8.gguf": 0.5 * 2 ** 30 },
      totalBytes: null, files: {}, mmproj: null, draft: null,
    },
  });
  assert.match(output, /Draft model:\s+org\/draft@v2/);
  assert.match(output, /draft-Q4\.gguf:\s+1\.50 GiB/);
  assert.match(output, /draft-Q8\.gguf:\s+2\.50 GiB/);
  assert.match(output, /Total:.*--draft-gguf-file for the draft/);
  assert.doesNotMatch(output, /select --gguf-file for the target/);
});

test("keeps alternatives separate and identifies both unresolved selections", () => {
  const alternative = {
    modelId: "org/target", revision: "main", format: "gguf", filename: null,
    weightsBytes: { "Q4.gguf": 2 ** 30, "Q8.gguf": 2 * 2 ** 30 },
    kvCacheBytes: null, totalBytes: null, files: {}, mmproj: null, draft: null,
  };
  const output = formatResult({ ...alternative, draft: { ...alternative, modelId: "org/draft" } });
  assert.match(output, /Total:.*--gguf-file for the target.*--draft-gguf-file for the draft/);
  assert.doesNotMatch(output, /Total:\s+\d/);
});

test("shows cache assumptions and preserves separate accessory costs", () => {
  const output = formatResult({
    modelId: "org/target", revision: "main", format: "gguf", filename: "model.gguf",
    weightsBytes: 2 ** 30, kvCacheBytes: 0.5 * 2 ** 30, totalBytes: 3.75 * 2 ** 30,
    files: { "model.gguf": { components: {}, kvCache: estimateGgufKvCache({
      "llama.block_count": 16, "llama.attention.head_count_kv": 8,
      "llama.attention.head_count": 32, "llama.embedding_length": 4096, "llama.context_length": 4096,
    }, { batchSize: 2 }) } },
    mmproj: { filename: "mmproj.gguf", bytes: 0.25 * 2 ** 30 },
    draft: {
      modelId: "org/draft", revision: "main", format: "safetensors", filename: null,
      weightsBytes: 2 ** 30, kvCacheBytes: 2 ** 30, totalBytes: 2 * 2 ** 30,
      files: {}, mmproj: null, draft: null,
    },
  });
  assert.match(output, /Multimodal projector:\s+0\.25 GiB/);
  assert.match(output, /Draft model:\s+1\.00 GiB/);
  assert.match(output, /Draft model KV cache:\s+1\.00 GiB/);
  assert.match(output, /Total:\s+3\.75 GiB/);
  assert.match(output, /Target cache: F16, 4096 tokens per sequence, 2 sequence\(s\)/);
});
