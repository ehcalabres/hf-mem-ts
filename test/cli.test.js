import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import test from "node:test";

const cli = new URL("../dist/cli.js", import.meta.url).href;

function invoke(args, options = {}) {
  const setup = `
    const args = JSON.parse(process.env.TEST_ARGS);
    process.argv = [process.execPath, ${JSON.stringify(cli)}, ...args];
    const header = Buffer.from(JSON.stringify({ weight: { dtype: 'F16', shape: [2], data_offsets: [0, 4] } }));
    const file = Buffer.alloc(8 + header.length + 4);
    file.writeBigUInt64LE(BigInt(header.length)); header.copy(file, 8);
    globalThis.fetch = async (input, init) => {
      if (process.env.TEST_DENY === 'true') return new Response('denied', { status: 401 });
      if (new Headers(init?.headers).get('authorization') !== process.env.TEST_EXPECT_TOKEN) throw Error('Incorrect authorization');
      const url = String(input);
      if (url.includes('/tree/')) return Response.json([{ type: 'file', path: 'model.safetensors' }]);
      if (url.includes('/api/models/')) return Response.json({ sha: 'a'.repeat(40) });
      const range = new Headers(init.headers).get('range').match(/bytes=(\\d+)-(\\d+)/);
      if (!range) throw Error('Expected a range request');
      const start = Number(range[1]); const end = Math.min(Number(range[2]), file.length - 1);
      return new Response(file.subarray(start, end + 1), { status: 206, headers: { 'Content-Range': 'bytes ' + start + '-' + end + '/' + file.length } });
    };
    await import(${JSON.stringify(cli)});
  `;
  return spawnSync(process.execPath, ["--input-type=module", "--eval", setup], {
    encoding: "utf8",
    timeout: 10_000,
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
