import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { llamaConfig, llamaTensors, tmpPath, writeGguf, writeJson, writeSafetensors } from "./helpers.js";
import { VERSION } from "../dist/index.js";
import { bar, where } from "../dist/render.js";
import { main } from "../dist/cli.js";

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

function model() {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeJson(join(tmp, "config.json"), llamaConfig());
  return tmp;
}

async function cli(...args) {
  let out = "";
  let err = "";
  const code = await main(args, {
    stdout: { write: (chunk) => (out += chunk) },
    stderr: { write: (chunk) => (err += chunk) },
  });
  return [code, out, err];
}

const widths = (out) => new Set(out.split("\n").filter(Boolean).map((line) => Array.from(line).length));

test("table", async () => {
  const [code, out, err] = await cli(model());
  assert.equal(code, 0);
  assert.ok(out.includes("TOTAL MEMORY") && out.includes("KV CACHE (1024 TOKENS") && out.includes("MODEL ("));
  assert.ok(out.includes(`hf-mem-ts v${VERSION}`));
  assert.ok(!out.includes("\x1b[") && err === "");
  assert.equal(widths(out).size, 1);
});

test("table without context length", async () => {
  const path = model();
  const config = llamaConfig();
  delete config.max_position_embeddings;
  writeJson(join(path, "config.json"), config);

  const [code, out, err] = await cli(path);

  assert.equal(code, 0);
  assert.ok(out.includes("MODEL (") && out.includes("BF16") && out.includes("F32"));
  assert.ok(!out.includes("KV CACHE ("));
  assert.ok(err.includes("KV cache estimation skipped, as the context length is not in `config.json`"));
  assert.equal(widths(out).size, 1);
});

for (const [tty, color, noColor, expected] of [
  [true, false, false, false],
  [true, true, false, true],
  [true, true, true, false],
  [false, true, false, false],
]) {
  test(`terminal color is opt in (tty=${tty}, color=${color}, no_color=${noColor})`, async () => {
    const isTTY = process.stdout.isTTY;
    const env = process.env.NO_COLOR;
    process.stdout.isTTY = tty;
    delete process.env.NO_COLOR;
    if (noColor) process.env.NO_COLOR = "1";
    try {
      const [code, out] = await cli(model(), ...(color ? ["--color"] : []));
      assert.equal(code, 0);
      assert.equal(out.includes("\x1b["), expected);
      assert.ok(out.includes("TOTAL MEMORY"));
    } finally {
      process.stdout.isTTY = isTTY;
      if (env === undefined) delete process.env.NO_COLOR;
      else process.env.NO_COLOR = env;
    }
  });
}

test("json", async () => {
  const [code, out] = await cli(model(), "-o", "json", "--max-model-len", "10", "--batch-size", "2");
  const data = JSON.parse(out);
  assert.equal(code, 0);
  const keys = [
    "version",
    "model",
    "revision",
    "file",
    "format",
    "total",
    "weights",
    "offloaded",
    "components",
    "kv_cache",
    "overhead",
    "moe",
    "fit",
    "warnings",
  ];
  assert.deepEqual(Object.keys(data), keys);
  assert.equal(data.version, `hf-mem-ts@${VERSION}`);
  assert.ok(data.kv_cache.max_model_len === 10 && data.kv_cache.batch_size === 2);
  assert.equal(data.total, data.weights.bytes + data.kv_cache.bytes + data.overhead.bytes);
});

test("gpu fit", async () => {
  const path = model();
  let [code, out, err] = await cli(path, "--gpu-memory", "2x4", "-o", "json");
  const fit = JSON.parse(out).fit;
  assert.ok(code === 0 && fit.gpus === 2 && fit.needed === 1 && fit.fits);
  assert.equal(fit.max_model_len, 1024);
  [code, out] = await cli(path, "--gpu-memory", "4");
  assert.ok(code === 0 && out.includes("GPU MEMORY") && out.includes("/ 4.00 GiB [") && out.includes("│ FITS "));
  [code, out] = await cli(path, "--gpu-memory", "2x4");
  assert.ok(out.includes("FITS ON 1 GPU") && out.includes("/ 8.00 GiB ["));
  [code, out] = await cli(path, "--gpu-memory", "2x4", "--output", "markdown");
  assert.ok(out.includes("| Overhead (activations, graphs, runtime) |"));
  assert.ok(out.includes("Fits on 2 x 4.00 GiB, needs 1 GPU, up to 1024 tokens."));
  [code, out] = await cli(path, "--gpu-memory", "0.5");
  assert.ok(out.includes("⚠ NEEDS 2 GPUS") || out.includes("⚠ DOES NOT FIT"));
  [code, , err] = await cli(path, "--gpu-memory", "lots");
  assert.ok(code === 1 && err.includes("--gpu-memory"));
});

test("markdown", async () => {
  const [code, out] = await cli(model(), "--output", "markdown");
  assert.equal(code, 0);
  assert.ok(out.includes("| **Total** |") && out.includes("| KV cache (BF16, 1024 tokens x 1) |"));
});

test("removed flags", async () => {
  const path = model();
  for (const flag of ["--model-id", "--experimental", "--json", "--markdown", "--json-output", "--details", "--gguf-file"]) {
    const [code, , err] = await cli(path, flag, "x");
    assert.ok(code === 2 && err.includes("unrecognized arguments"), flag);
  }
});

test("gguf listing table", async () => {
  const tmp = tmpPath();
  for (const quant of ["Q4_K_M", "Q8_0"]) writeGguf(join(tmp, `m-${quant}.gguf`), ARCH, TENSORS);
  let [code, out] = await cli(tmp, "--file", "m-Q8_0.gguf", "-o", "json");
  assert.ok(code === 0 && JSON.parse(out).file === "m-Q8_0.gguf");
  [code, out] = await cli(tmp);
  assert.ok(code === 0 && out.includes("FILES") && out.includes("m-Q4_K_M.gguf"));
  assert.ok(out.includes("GiB WEIGHTS +") && out.includes("GiB OVERHEAD)"));
});

test("errors", async () => {
  const tmp = tmpPath();
  let [code, , err] = await cli(join(tmp, "missing"));
  assert.ok(code === 1 && err.startsWith("hf-mem-ts: error:"));
  [code, , err] = await cli();
  assert.ok(code === 2 && err.includes("required"));
  [code] = await cli(tmp, "--batch-size", "0");
  assert.equal(code, 2);
});

test("version", async () => {
  const [code, out] = await cli("--version");
  assert.ok(code === 0 && out.trim() === `hf-mem-ts ${VERSION}`);
});

test("titles shorten paths and commits", () => {
  const tmp = realpathSync(tmpPath());
  mkdirSync(join(tmp, "cwd"));
  const cwd = process.cwd();
  const home = process.env.HOME;
  process.chdir(join(tmp, "cwd"));
  process.env.HOME = join(tmp, "home");
  try {
    assert.equal(where(join(tmp, "cwd", "m"), "a".repeat(40)), "m @ aaaaaaa");
    assert.equal(where(join(tmp, "home", "m"), null), "~/m");
    assert.equal(where("https://hf.co/org/model", "main"), "https://hf.co/org/model @ main");
    const long = where("https://hf.co/org/" + "x".repeat(200), null);
    assert.ok(long.length === 72 && long.startsWith("...") && long.endsWith("x"));
  } finally {
    process.chdir(cwd);
    if (home === undefined) delete process.env.HOME;
    else process.env.HOME = home;
  }
});

test("version matches package.json", () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf-8"));
  assert.equal(pkg.version, VERSION);
});

test("errors are actionable", async () => {
  const tmp = tmpPath();
  writeFileSync(join(tmp, "pytorch_model.bin"), Buffer.alloc(1000));
  writeFileSync(join(tmp, "optimizer.pt"), Buffer.alloc(10));
  let [code, , err] = await cli(tmp);
  assert.ok(code === 1 && err.includes("only has PyTorch pickle weights (e.g. `pytorch_model.bin`)"));
  [code, , err] = await cli(join(tmp, "missing"));
  assert.ok(code === 1 && err.includes("does not exist"));
  [code, , err] = await cli("not-a-model");
  assert.ok(code === 1 && err.includes("is not a Hub model ID (`org/name`)"));
});

test("bars show small fractions", () => {
  assert.equal(bar(0.0, 4), "░░░░");
  assert.equal(bar(0.001, 4), "▏░░░");
  assert.equal(bar(0.5, 4), "██░░");
  assert.equal(bar(0.3, 4), "█▎░░");
  assert.equal(bar(1.0, 4), "████");
  for (let f = 0; f <= 100; f++) assert.equal(Array.from(bar(f / 100, 7)).length, 7);
});

test("encoder table", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [1000, 64]] });
  writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["BertModel"] }));
  const [code, out, err] = await cli(tmp, "--max-num-batched-tokens", "512");
  assert.ok(code === 0 && err.includes("Encoder warmup"));
  assert.ok(out.includes("MODEL (") && out.includes("ACTIVATIONS") && !out.includes("KV CACHE"));
});

for (const [dtype, target, size] of [
  ["float16", "F16", 2],
  ["bfloat16", "BF16", 2],
  ["float32", "F32", 4],
]) {
  test(`encoder dtype option (${dtype})`, async () => {
    const tmp = tmpPath();
    writeSafetensors(join(tmp, "model.safetensors"), { w: ["F32", [1000, 64]] });
    writeJson(join(tmp, "config.json"), llamaConfig({ architectures: ["BertModel"] }));
    const [code, out, err] = await cli(tmp, "--dtype", dtype, "--output", "json");
    const result = JSON.parse(out);
    assert.ok(code === 0 && err === "");
    assert.deepEqual(result.weights.dtypes, { [target]: { bytes: 64000 * size, params: 64000 } });
  });
}
