import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import test from "node:test";

const cli = new URL("../dist/cli.js", import.meta.url).href;

function invoke(args, options = {}, timeout = 10_000) {
  const setup = `
    const args = JSON.parse(process.env.TEST_ARGS);
    process.argv = [process.execPath, ${JSON.stringify(cli)}, ...args];
    const header = Buffer.from(JSON.stringify({ weight: { dtype: 'F16', shape: [2], data_offsets: [0, 4] } }));
    const file = Buffer.alloc(8 + header.length + 4);
    file.writeBigUInt64LE(BigInt(header.length)); header.copy(file, 8);
    let failures = Number(process.env.TEST_TRANSIENT_FAILURES ?? 0);
    globalThis.fetch = async (input, init) => {
      if (process.env.TEST_DENY === 'true') return new Response('denied', { status: 401 });
      if (new Headers(init?.headers).get('authorization') !== process.env.TEST_EXPECT_TOKEN) throw Error('Incorrect authorization');
      if (process.env.TEST_STALL_BODY === 'true') return new Response(new ReadableStream({}));
      if (failures > 0) {
        failures--;
        return new Response('temporarily unavailable', { status: 503, headers: { 'Retry-After': '0' } });
      }
      const url = String(input);
      if (url.includes('/tree/')) return Response.json([
        { type: 'file', path: 'model.safetensors' },
        ...(process.env.TEST_CACHE_CONFIG ? [{ type: 'file', path: 'config.json' }] : []),
      ]);
      if (url.includes('/api/models/')) return Response.json({ sha: 'a'.repeat(40) });
      if (url.endsWith('config.json')) return Response.json(JSON.parse(process.env.TEST_CACHE_CONFIG));
      const range = new Headers(init.headers).get('range').match(/bytes=(\\d+)-(\\d+)/);
      if (!range) throw Error('Expected a range request');
      const start = Number(range[1]); const end = Math.min(Number(range[2]), file.length - 1);
      return new Response(file.subarray(start, end + 1), { status: 206, headers: { 'Content-Range': 'bytes ' + start + '-' + end + '/' + file.length } });
    };
    await import(${JSON.stringify(cli)});
  `;
  return spawnSync(process.execPath, ["--input-type=module", "--eval", setup], {
    encoding: "utf8",
    timeout,
    env: { ...process.env, TEST_ARGS: JSON.stringify(args), TEST_EXPECT_TOKEN: "Bearer environment-token", HF_TOKEN: "environment-token", ...options },
  });
}

test("CLI prints valid JSON and honors explicit authentication over HF_TOKEN", () => {
  for (const [args, expected] of [
    [["org/model", "--json"], "Bearer environment-token"],
    [["org/model", "--json", "--token", "explicit-token"], "Bearer explicit-token"],
  ]) {
    const child = invoke(args, { TEST_EXPECT_TOKEN: expected });
    assert.equal(child.status, 0, child.stderr);
    assert.equal(child.stderr, "");
    const result = JSON.parse(child.stdout);
    assert.equal(result.weightsBytes, 4);
    assert.equal(result.totalBytes, 4);
    assert.equal(result.kvCacheBytes, null);
    assert.equal(result.kvCacheBytesByTp, null);
    assert.equal(result.totalBytesByTp, null);
  }
});

test("CLI keeps request failures on stderr and does not emit partial JSON", () => {
  const child = invoke(["org/model", "--json"], { TEST_DENY: "true" });
  assert.equal(child.status, 1);
  assert.equal(child.stdout, "");
  assert.match(child.stderr, /401/);
  assert.doesNotMatch(child.stderr, /environment-token/);
});

test("CLI rejects an invalid context before making network requests", () => {
  const child = invoke(["org/model", "--max-model-len", "0.5"], { TEST_DENY: "true" });
  assert.equal(child.status, 1);
  assert.equal(child.stdout, "");
  assert.match(child.stderr, /--max-model-len.*positive integer/);
  assert.doesNotMatch(child.stderr, /401/);
});

test("CLI help does not require a model or network access", () => {
  const output = execFileSync(process.execPath, [new URL("../dist/cli.js", import.meta.url).pathname, "--help"], { encoding: "utf8" });
  assert.match(output, /--draft-gguf-file/);
  assert.match(output, /--kv-cache/);
});

test("CLI honors zero retries and can recover with a larger explicit retry budget", () => {
  const disabled = invoke(["org/model", "--json", "--max-retries", "0"], { TEST_TRANSIENT_FAILURES: "1" });
  assert.equal(disabled.status, 1, disabled.stderr);
  assert.equal(disabled.stdout, "");
  assert.match(disabled.stderr, /503/);

  const recovered = invoke(["org/model", "--json", "--max-retries", "3"], { TEST_TRANSIENT_FAILURES: "3" });
  assert.equal(recovered.status, 0, recovered.stderr);
  assert.equal(recovered.stderr, "");
  assert.equal(JSON.parse(recovered.stdout).totalBytes, 4);
});

test("CLI request deadline interrupts a stalled metadata response body", () => {
  const child = invoke(["org/model", "--json", "--request-timeout-ms", "50"], { TEST_STALL_BODY: "true" }, 2_000);
  assert.equal(child.status, 1, child.stderr || child.error?.message);
  assert.equal(child.stdout, "");
  assert.match(child.stderr, /Metadata request timed out/);
});

test("CLI rejects out-of-range request policies before making network requests", () => {
  for (const [flag, value, error] of [
    ["--request-timeout-ms", "0", /--request-timeout-ms.*positive integer/],
    ["--request-timeout-ms", "2147483648", /requestTimeoutMs.*between 1 and 2147483647/],
    ["--max-retries", "0.5", /--max-retries.*non-negative integer/],
    ["--max-retries", "11", /maxRetries.*between 0 and 10/],
  ]) {
    const child = invoke(["org/model", flag, value], { TEST_DENY: "true" });
    assert.equal(child.status, 1);
    assert.equal(child.stdout, "");
    assert.match(child.stderr, error);
    assert.doesNotMatch(child.stderr, /401/);
  }
});

test("CLI tensor parallelism changes aggregate KV payload and is inherited by the draft", () => {
  const config = JSON.stringify({
    hidden_size: 32, num_hidden_layers: 2, num_attention_heads: 8, num_key_value_heads: 1,
    max_position_embeddings: 16, dtype: "float16",
  });
  const child = invoke(["org/model", "--kv-cache", "--tensor-parallel-size", "4", "--draft-model", "org/draft", "--json"], { TEST_CACHE_CONFIG: config });
  assert.equal(child.status, 0, child.stderr);
  const result = JSON.parse(child.stdout);
  assert.equal(result.kvCacheBytes, 2048);
  assert.equal(result.draft.kvCacheBytes, 2048);
  assert.equal(result.totalBytes, 4104);
  assert.equal(result.kvCacheBytesByTp, null);
  assert.equal(result.totalBytesByTp, null);
  assert.equal(result.files.safetensors.kvCacheByTp, null);
  assert.equal(result.draft.kvCacheBytesByTp, null);
  assert.equal(result.draft.totalBytesByTp, null);
});

test("CLI default cache JSON exposes all TP alternatives without choosing a scalar total", () => {
  const child = invoke(["org/model", "--kv-cache", "--json"], {
    TEST_CACHE_CONFIG: JSON.stringify({
      hidden_size: 32, num_hidden_layers: 2, num_attention_heads: 8, num_key_value_heads: 1,
      max_position_embeddings: 16, dtype: "float16",
    }),
  });
  assert.equal(child.status, 0, child.stderr);
  assert.equal(child.stderr, "");
  const result = JSON.parse(child.stdout);
  assert.equal(result.weightsBytes, 4);
  assert.equal(result.kvCacheBytes, null);
  assert.equal(result.totalBytes, null);
  assert.deepEqual(result.kvCacheBytesByTp, { "1": 512, "2": 1024, "4": 2048, "8": 4096 });
  assert.deepEqual(result.totalBytesByTp, { "1": 516, "2": 1028, "4": 2052, "8": 4100 });
  assert.equal(result.files.safetensors.kvCache, null);
  assert.deepEqual(Object.keys(result.files.safetensors.kvCacheByTp), ["1", "2", "4", "8"]);
  for (const tp of ["1", "2", "4", "8"]) {
    assert.equal(result.files.safetensors.kvCacheByTp[tp].tensorParallelSize, Number(tp));
    assert.equal(result.files.safetensors.kvCacheByTp[tp].bytes, result.kvCacheBytesByTp[tp]);
  }
});

test("CLI text shows each default TP cache and its own total, but explicit TP remains single", () => {
  const options = { TEST_CACHE_CONFIG: JSON.stringify({
    hidden_size: 32, num_hidden_layers: 2, num_attention_heads: 8, num_key_value_heads: 1,
    max_position_embeddings: 16, dtype: "float16",
  }) };
  const comparison = invoke(["org/model", "--kv-cache"], options);
  assert.equal(comparison.status, 0, comparison.stderr);
  for (const [tp, cache, total] of [[1, 512, 516], [2, 1024, 1028], [4, 2048, 2052], [8, 4096, 4100]]) {
    assert.match(comparison.stdout, new RegExp(`TP\\s+${tp}\\b[^\\n]*\\b${cache} bytes\\b[^\\n]*\\b${total} bytes\\b`));
  }
  const explicit = invoke(["org/model", "--kv-cache", "--tensor-parallel-size", "3"], options);
  assert.equal(explicit.status, 0, explicit.stderr);
  assert.match(explicit.stdout, /\b1536 bytes\b/);
  assert.doesNotMatch(explicit.stdout, /^\s*TP\s+[1248]\b/m);
});

test("CLI rejects TP zero before requesting metadata", () => {
  const child = invoke(["org/model", "--kv-cache", "--tensor-parallel-size", "0", "--json"], { TEST_DENY: "true" });
  assert.equal(child.status, 1);
  assert.equal(child.stdout, "");
  assert.match(child.stderr, /--tensor-parallel-size.*positive integer/);
  assert.doesNotMatch(child.stderr, /401/);
});
