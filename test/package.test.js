import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = fileURLToPath(new URL("../", import.meta.url));

test("packed package works through its executable, ESM subpaths, and TypeScript declarations", async () => {
  const temp = await mkdtemp(join(tmpdir(), "hf-mem-consumer-"));
  const npm = process.platform === "win32" ? "npm.cmd" : "npm";
  try {
    const packed = JSON.parse(execFileSync(npm, ["pack", "--ignore-scripts", "--json", "--pack-destination", temp], {
      cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"],
    }));
    const consumer = join(temp, "consumer");
    await mkdir(consumer);
    await writeFile(join(consumer, "package.json"), JSON.stringify({ private: true, type: "module" }));
    execFileSync(npm, ["install", "--offline", "--ignore-scripts", "--no-audit", "--no-fund", "--package-lock=false", join(temp, packed[0].filename)], {
      cwd: consumer, stdio: "pipe",
    });
    const manifest = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
    const version = execFileSync(join(consumer, "node_modules", ".bin", "hf-mem-ts"), ["--version"], {
      cwd: consumer, encoding: "utf8",
    }).trim();
    assert.equal(version, manifest.version);
    const source = `
      import { parseSafetensorsHeaders, type EstimateOptions } from 'hf-mem-ts';
      import { parseSafetensorsHeaders as parse } from 'hf-mem-ts/safetensors';
      import { estimateGgufKvCache } from 'hf-mem-ts/gguf';
      import { estimateSafetensorsKvCache } from 'hf-mem-ts/kv-cache';
      const options: EstimateOptions = { modelId: 'org/model' };
      const header = { Transformer: { weight: { dtype: 'F16', shape: [2] } } };
      const config = { hidden_size: 8, num_hidden_layers: 2, num_attention_heads: 2, max_position_embeddings: 16, torch_dtype: 'float16' };
      const metadata = { 'llama.block_count': 2, 'llama.attention.head_count_kv': 2, 'llama.attention.head_count': 2, 'llama.embedding_length': 8, 'llama.context_length': 16 };
      console.log(JSON.stringify([options.modelId, parseSafetensorsHeaders(header).bytes, parse(header).bytes, estimateSafetensorsKvCache(config).bytes, estimateGgufKvCache(metadata).bytes]));
    `;
    await writeFile(join(consumer, "consumer.ts"), source);
    execFileSync(process.execPath, [join(root, "node_modules", "typescript", "bin", "tsc"), "--strict", "--module", "NodeNext", "--target", "ES2022", "--outDir", "out", "consumer.ts"], {
      cwd: consumer, stdio: "pipe",
    });
    const output = execFileSync(process.execPath, ["out/consumer.js"], { cwd: consumer, encoding: "utf8" });
    assert.deepEqual(JSON.parse(output), ["org/model", 4, 4, 1024, 1024]);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
