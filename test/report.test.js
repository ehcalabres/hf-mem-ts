import assert from "node:assert/strict";
import test from "node:test";

import { formatResult } from "../dist/report.js";

test("formats a global report with target and draft memory rows", () => {
  const draft = {
    modelId: "org/draft",
    revision: "main",
    resolvedRevision: "a".repeat(40),
    format: "safetensors",
    filename: null,
    weightsBytes: 1 * 2 ** 30,
    kvCacheBytes: 0.5 * 2 ** 30,
    totalBytes: 1.5 * 2 ** 30,
    files: {},
    mmproj: null,
    draft: null,
  };
  const result = {
    modelId: "org/model",
    revision: "main",
    resolvedRevision: "b".repeat(40),
    format: "safetensors",
    filename: null,
    weightsBytes: 4 * 2 ** 30,
    kvCacheBytes: 2 * 2 ** 30,
    totalBytes: 7.5 * 2 ** 30,
    files: {},
    mmproj: null,
    draft,
  };

  const output = formatResult(result);
  assert.match(output, /Resolved revision:\s+b{40}/);
  assert.match(output, /Model:\s+4\.00 GiB/);
  assert.match(output, /KV cache:\s+2\.00 GiB/);
  assert.match(output, /Draft model:\s+1\.00 GiB \(org\/draft@main\)/);
  assert.match(output, /Draft model KV cache:\s+0\.50 GiB/);
  assert.match(output, /Total:\s+7\.50 GiB/);
});

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
