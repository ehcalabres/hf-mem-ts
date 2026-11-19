import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, symlinkSync, writeFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";

import { LocalSource, nodeModules, openSource, parseModel } from "../dist/source.js";
import { estimate, llamaConfig, llamaTensors, tmpPath, writeJson, writeSafetensors } from "./helpers.js";

function some(value) {
  assert.notEqual(value, null);
  assert.notEqual(value, undefined);
  return value;
}

function setEnv(values) {
  const saved = Object.fromEntries(Object.keys(values).map((k) => [k, process.env[k]]));
  const apply = (entries) => {
    for (const [k, v] of Object.entries(entries)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  };
  apply(values);
  return () => apply(saved);
}

for (const [model, expected] of [
  ["Qwen/Qwen3-8B", ["Qwen/Qwen3-8B", null, null]],
  ["Qwen/Qwen3-8B@refs/pr/1", ["Qwen/Qwen3-8B", "refs/pr/1", null]],
  ["https://huggingface.co/Qwen/Qwen3-8B", ["Qwen/Qwen3-8B", null, null]],
  ["https://hf.co/Qwen/Qwen3-8B/tree/dev", ["Qwen/Qwen3-8B", "dev", null]],
  ["https://hf.co/org/m/tree/refs%2Fpr%2F1", ["org/m", "refs/pr/1", null]],
  ["org/m-GGUF:Q4_K_M", ["org/m-GGUF", null, "Q4_K_M"]],
  ["org/m-GGUF@dev:UD-Q4_K_XL", ["org/m-GGUF", "dev", "UD-Q4_K_XL"]],
  ["https://hf.co/org/m/blob/main/a%20b.gguf", ["org/m", "main", "a b.gguf"]],
  ["https://huggingface.co/org/m-GGUF/blob/main/q/m-Q4_K_M.gguf", ["org/m-GGUF", "main", "q/m-Q4_K_M.gguf"]],
]) {
  test(`parse model [${model}]`, async () => {
    assert.deepEqual(await parseModel(model), expected);
  });
}

test("parse model rejects other urls", async () => {
  await assert.rejects(parseModel("https://huggingface.co/datasets"), /not a valid/);
});

test("token resolution", async (t) => {
  const tmp = tmpPath();
  t.after(setEnv({ HF_TOKEN: undefined, HF_TOKEN_PATH: undefined, HF_HOME: tmp, HF_HUB_OFFLINE: undefined }));
  const token = async (arg) => {
    let authorization;
    const fetch = async (url, init) => {
      authorization = init.headers.Authorization ?? null;
      return new Response("[]", { headers: { "x-repo-commit": "a".repeat(40) } });
    };
    await openSource("org/model", null, arg, null, { fetch });
    return authorization === null ? null : authorization.replace(/^Bearer /, "");
  };
  assert.equal(await token(null), null);
  writeFileSync(join(tmp, "token"), "hf_file\n");
  assert.equal(await token(null), "hf_file");
  process.env.HF_TOKEN = "hf_env";
  assert.equal(await token(null), "hf_env");
  assert.equal(await token("hf_arg"), "hf_arg");
});

function cache(tmp, modelId = "org/model", sha = "a".repeat(40)) {
  const repo = join(tmp, `models--${modelId.replaceAll("/", "--")}`);
  const snapshot = join(repo, "snapshots", sha);
  writeSafetensors(join(snapshot, "model.safetensors"), llamaTensors());
  writeJson(join(snapshot, "config.json"), llamaConfig());
  mkdirSync(join(repo, "refs"), { recursive: true });
  writeFileSync(join(repo, "refs", "main"), sha);
  return snapshot;
}

test("offline mode reads the cache", async (t) => {
  const tmp = tmpPath();
  cache(tmp);
  t.after(setEnv({ HF_HUB_CACHE: tmp, HF_HUB_OFFLINE: "1" }));

  const result = await estimate("org/model");

  assert.deepEqual([result.model, result.revision, result.url], ["org/model", "main", "https://hf.co/org/model"]);
  assert.notEqual(result.kvCache, null);
  assert.equal((await estimate("org/model", { revision: "a".repeat(40) })).total, result.total);
  await assert.rejects(estimate("org/other"), /not in the local cache/);
});

test("cache snapshot resolution", async (t) => {
  const tmp = tmpPath();
  const snapshot = cache(tmp);
  t.after(setEnv({ HF_HUB_CACHE: tmp, HF_HUB_OFFLINE: "1" }));
  const [source] = await openSource("org/model", "main");
  assert.equal(source.root, snapshot);
  await assert.rejects(openSource("org/model", "dev"), /not in the local cache/);
  await assert.rejects(openSource("org/missing", "main"), /not in the local cache/);
});

test("local listing skips hidden files", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  writeSafetensors(join(tmp, ".cache", "model.safetensors"), llamaTensors());
  writeFileSync(join(tmp, ".hidden"), "x");
  const source = LocalSource.open(some(await nodeModules()), tmp);
  assert.deepEqual([...source.files.keys()], ["model.safetensors"]);
  assert.ok(isAbsolute(source.name));
  assert.equal(source.revision, null);
});

test("local dir revision from download metadata", async () => {
  const tmp = tmpPath();
  writeSafetensors(join(tmp, "model.safetensors"), llamaTensors());
  const metadata = join(tmp, ".cache", "huggingface", "download");
  mkdirSync(metadata, { recursive: true });
  writeFileSync(join(metadata, "model.safetensors.metadata"), "c".repeat(40) + "\netag\n1.0\n");
  writeFileSync(join(metadata, "config.json.metadata"), "d".repeat(40) + "\netag\n1.0\n");
  assert.equal(LocalSource.open(some(await nodeModules()), tmp).revision, "d".repeat(40));
});

test("invalid model", async () => {
  await assert.rejects(estimate(join(tmpPath(), "missing")), /does not exist/);
});

function hubCache(tmp, sha) {
  const staging = join(tmp, "staging");
  const repo = join(tmp, "models--org--model");
  writeSafetensors(join(staging, "model.safetensors"), llamaTensors());
  writeJson(join(staging, "config.json"), llamaConfig());
  const oids = {};
  for (const path of ["model.safetensors", "config.json"]) {
    const oid = createHash("sha256").update(readFileSync(join(staging, path))).digest("hex");
    mkdirSync(join(repo, "blobs"), { recursive: true });
    renameSync(join(staging, path), join(repo, "blobs", oid));
    mkdirSync(join(repo, "snapshots", sha), { recursive: true });
    symlinkSync(join("..", "..", "blobs", oid), join(repo, "snapshots", sha, path));
    oids[path] = oid;
  }
  return oids;
}

test("online mode reads validated files from the cache", async (t) => {
  const tmp = tmpPath();
  const sha = "b".repeat(40);
  const oids = hubCache(tmp, sha);
  t.after(setEnv({ HF_HUB_CACHE: tmp, HF_HUB_OFFLINE: undefined }));
  const tree = [
    {
      type: "file",
      path: "model.safetensors",
      size: 1,
      lfs: { oid: oids["model.safetensors"], size: 1 << 30 },
    },
    { type: "file", path: "config.json", size: 1, oid: oids["config.json"] },
  ];
  const requested = [];

  const fetch = async (url) => {
    requested.push(url);
    if (!url.includes("/api/models/")) {
      return new Response(null, { status: 400, headers: { "x-error-message": `unexpected request to ${url}` } });
    }
    return new Response(JSON.stringify(tree), { headers: { "x-repo-commit": sha } });
  };

  const result = await estimate("org/model", { fetch });
  assert.ok(result.kvCache !== null && requested.length === 1);

  tree[1].oid = "stale";
  requested.length = 0;
  assert.equal((await estimate("org/model", { fetch })).kvCache, null);
  assert.deepEqual(
    requested.slice(1).map((url) => url.slice(url.lastIndexOf("/") + 1)),
    ["config.json"],
  );
});
