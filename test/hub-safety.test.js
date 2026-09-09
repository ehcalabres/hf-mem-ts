import assert from "node:assert/strict";
import test from "node:test";
import { estimateModelMemory } from "../dist/index.js";

const SHA = "a".repeat(40);

function gguf(config = false) {
  const chunks = [];
  const u32 = (value) => { const bytes = new Uint8Array(4); new DataView(bytes.buffer).setUint32(0, value, true); chunks.push(bytes); };
  const u64 = (value) => { const bytes = new Uint8Array(8); new DataView(bytes.buffer).setBigUint64(0, BigInt(value), true); chunks.push(bytes); };
  const string = (value) => { const bytes = new TextEncoder().encode(value); u64(bytes.length); chunks.push(bytes); };
  chunks.push(new TextEncoder().encode("GGUF")); u32(3); u64(1); u64(config ? 5 : 0);
  if (config) {
    for (const [key, value] of Object.entries({
      "llama.block_count": 2, "llama.attention.head_count_kv": 1,
      "llama.attention.head_count": 2, "llama.embedding_length": 8, "llama.context_length": 16,
    })) { string(key); u32(4); u32(value); }
  }
  string("blk.0.weight"); u32(1); u64(4); u32(1); u64(0);
  const size = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const output = new Uint8Array(Math.ceil(size / 32) * 32 + 8);
  let offset = 0;
  for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; }
  return output;
}

function repository(files, onResolve) {
  return async (input, init) => {
    const url = new URL(input);
    if (url.pathname.includes("/revision/")) return Response.json({ sha: SHA });
    if (url.pathname.includes("/tree/")) return Response.json(files.map((path) => ({ type: "file", path })));
    if (onResolve) return onResolve(url, init);
    const bytes = gguf(url.pathname.includes("00001"));
    return new Response(bytes, { status: 206, headers: { "Content-Range": `bytes 0-${bytes.length - 1}/${bytes.length}` } });
  };
}

test("padded first shard supplies the sole KV cache and projector bytes are additive", async () => {
  const result = await estimateModelMemory({
    modelId: "org/model", ggufFile: "model-00002-of-00002.gguf", kvCache: true,
    fetch: repository(["model-00002-of-00002.gguf", "model-00001-of-00002.gguf", "mmproj_f16.gguf"]),
  });
  assert.equal(result.filename, "model.gguf");
  assert.equal(result.weightsBytes, 16);
  assert.equal(result.kvCacheBytes, 512);
  assert.equal(result.mmproj.bytes, 8);
  assert.equal(result.totalBytes, 536);
  assert.equal(result.files["model.gguf"].parameters, 8);
  assert.equal(result.revision, "main");
  assert.equal(result.resolvedRevision, SHA);
  assert.equal(result.mmproj.resolvedRevision, SHA);
});

test("unselected complete shards yield one alternative without a partial total", async () => {
  const result = await estimateModelMemory({ modelId: "org/model", fetch: repository(["model-1-of-2.gguf", "model-2-of-2.gguf"]) });
  assert.deepEqual(result.weightsBytes, { "model.gguf": 16 });
  assert.equal(result.totalBytes, null);
});

for (const [name, files, selected, error] of [
  ["missing shard", ["model-1-of-2.gguf"], "model-1-of-2.gguf", /Incomplete/],
  ["inconsistent counts", ["model-1-of-2.gguf", "model-2-of-3.gguf"], undefined, /Inconsistent/],
  ["duplicate numeric indices", ["model-01-of-2.gguf", "model-1-of-2.gguf", "model-2-of-2.gguf"], undefined, /Duplicate/],
  ["zero count", ["model-1-of-0.gguf"], undefined, /Invalid/],
  ["out of range index", ["model-3-of-2.gguf"], undefined, /Invalid/],
  ["unsafe count", ["model-1-of-9007199254740992.gguf"], undefined, /Invalid/],
  ["absent requested shard", ["model-1-of-2.gguf", "model-2-of-2.gguf"], "model-3-of-2.gguf", /No GGUF file/],
  ["ambiguous directories", ["a/model-1-of-1.gguf", "b/model-1-of-1.gguf"], "model-1-of-1.gguf", /Multiple/],
  ["colliding group", ["model.gguf", "model-1-of-1.gguf"], undefined, /Ambiguous/],
]) {
  test(`rejects ${name} before fetching weights`, async () => {
    let reads = 0;
    await assert.rejects(estimateModelMemory({
      modelId: "org/model", ...(selected ? { ggufFile: selected } : {}),
      fetch: repository(files, () => { reads++; throw new Error("weights must not be fetched"); }),
    }), error);
    assert.equal(reads, 0);
  });
}

test("full paths select a complete set without mixing same-basename directories", async () => {
  const result = await estimateModelMemory({
    modelId: "org/model", ggufFile: "b/model-1-of-1.gguf",
    fetch: repository(["a/model-1-of-1.gguf", "b/model-1-of-1.gguf"]),
  });
  assert.equal(result.filename, "b/model.gguf");
  assert.equal(result.weightsBytes, 8);
});

test("all projector basename forms remain excluded when disabled", async () => {
  const result = await estimateModelMemory({
    modelId: "org/model", mmprojFile: false,
    fetch: repository(["model.gguf", "mmproj.gguf", "nested/mmproj_f16.gguf", "mmproj-F32.gguf"]),
  });
  assert.deepEqual(result.weightsBytes, { "model.gguf": 8 });
  assert.equal(result.mmproj, null);
});

test("ambiguous projector selection cannot launch a failing base promise", async () => {
  let reads = 0;
  await assert.rejects(estimateModelMemory({
    modelId: "org/model", ggufFile: "model.gguf", maxRetries: 0,
    fetch: repository(["model.gguf", "mmproj-Q8.gguf", "mmproj-Q4.gguf"], async () => {
      reads++; throw new TypeError("base fetch failure");
    }),
  }), /Multiple mmproj/);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(reads, 0);
});

test("simultaneously failing base and projector requests are both observed", async () => {
  await assert.rejects(estimateModelMemory({
    modelId: "org/model", ggufFile: "model.gguf", maxRetries: 0,
    fetch: repository(["model.gguf", "mmproj.gguf"], async () => { throw new TypeError("metadata failure"); }),
  }), /metadata failure/);
  // node:test fails the test on an unhandled rejection, including after its promise settles.
  await new Promise((resolve) => setImmediate(resolve));
});

test("one per-model bound includes projector response-body consumption", async () => {
  let active = 0;
  let peak = 0;
  const bytes = gguf();
  const result = await estimateModelMemory({
    modelId: "org/model", concurrency: 1,
    fetch: repository(["a.gguf", "b.gguf", "mmproj.gguf"], () => {
      active++; peak = Math.max(peak, active);
      return new Response(new ReadableStream({
        start(controller) {
          setTimeout(() => { active--; controller.enqueue(bytes); controller.close(); }, 5);
        },
      }), { status: 206, headers: { "Content-Range": `bytes 0-${bytes.length - 1}/${bytes.length}` } });
    }),
  });
  assert.equal(peak, 1);
  assert.deepEqual(result.weightsBytes, { "a.gguf": 8, "b.gguf": 8 });
  assert.equal(result.mmproj.bytes, 8);
});

test("pins a mutable revision across paginated tree, base, and projector requests", async () => {
  let lookups = 0;
  const seen = [];
  const delegate = repository([]);
  const result = await estimateModelMemory({
    modelId: "org/model", revision: "refs/pr/7", ggufFile: "model.gguf",
    fetch: async (input, init) => {
      const url = new URL(input); seen.push(url.pathname);
      if (url.pathname.includes("/revision/")) { lookups++; return Response.json({ sha: SHA }); }
      assert.ok(url.pathname.includes(`/${SHA}`));
      if (url.pathname.includes("/tree/")) {
        if (url.searchParams.has("cursor")) return Response.json([{ type: "file", path: "mmproj.gguf" }]);
        return Response.json([{ type: "file", path: "model.gguf" }], { headers: { Link: '<?cursor=second>; rel="next"' } });
      }
      return delegate(input, init);
    },
  });
  assert.equal(lookups, 1);
  assert.equal(result.revision, "refs/pr/7");
  assert.equal(result.resolvedRevision, SHA);
  assert.equal(result.totalBytes, 16);
  assert.equal(seen.filter((path) => path.includes("/tree/")).length, 2);
});

for (const [name, next, error] of [
  ["cross-origin link", "https://attacker.example/tree?page=2", /Unsafe/],
  ["changed snapshot", `/api/models/org/model/tree/${"b".repeat(40)}?page=2`, /Unsafe/],
  ["loop", `?expand=false&limit=1000&recursive=true`, /loop/],
]) {
  test(`rejects pagination ${name} before forwarding credentials`, async () => {
    let requests = 0;
    await assert.rejects(estimateModelMemory({
      modelId: "org/model", token: "secret",
      fetch: async (input, init) => {
        requests++;
        assert.equal(new URL(input).origin, "https://huggingface.co");
        assert.equal(new Headers(init.headers).get("authorization"), "Bearer secret");
        if (String(input).includes("/revision/")) return Response.json({ sha: SHA });
        return Response.json([], { headers: { Link: `<${next}>; rel="next"` } });
      },
    }), error);
    assert.equal(requests, 2);
  });
}

test("invalid revision metadata cannot silently use the moving branch", async () => {
  let requests = 0;
  await assert.rejects(estimateModelMemory({ modelId: "org/model", fetch: async () => {
    requests++; return Response.json({ sha: "main" });
  } }), /immutable commit/);
  assert.equal(requests, 1);
});
