import assert from "node:assert/strict";
import test from "node:test";
import { estimateModelMemory } from "../dist/index.js";
import { requestPolicy, transportFetch } from "../dist/transport.js";

function transport(fetcher, options = {}, concurrency = 1) {
  return transportFetch(fetcher, concurrency, requestPolicy({ modelId: "org/model", ...options }));
}

for (const [field, value] of [
  ["requestTimeoutMs", 0], ["requestTimeoutMs", Infinity], ["requestTimeoutMs", 2 ** 31],
  ["requestTimeoutMs", 1.5], ["maxRetries", -1], ["maxRetries", NaN], ["maxRetries", 11], ["maxRetries", 1.5],
]) {
  test(`validates ${field}=${value} before making requests`, async () => {
    let called = false;
    await assert.rejects(estimateModelMemory({ modelId: "org/model", [field]: value, fetch: async () => {
      called = true; throw new Error("must not request");
    } }), RangeError);
    assert.equal(called, false);
  });
}

test("retries transient HTTP failures and returns the final successful body", async () => {
  const statuses = [429, 502, 503, 504, 200];
  const fetcher = transport(async () => new Response("payload", {
    status: statuses.shift(), headers: { "Retry-After": "0" },
  }), { maxRetries: 4 });
  assert.equal(await (await fetcher("https://example.com")).text(), "payload");
  assert.deepEqual(statuses, []);
});

test("default retry budget stops after three attempts", async () => {
  let attempts = 0;
  const fetcher = transport(async () => {
    attempts++; return new Response(null, { status: 503, headers: { "Retry-After": "0" } });
  });
  assert.equal((await fetcher("https://example.com")).status, 503);
  assert.equal(attempts, 3);
});

test("retries a fetch network error without retrying arbitrary application errors", async () => {
  let attempts = 0;
  const fetcher = transport(async () => {
    if (++attempts === 1) throw new TypeError("network disconnected");
    return Response.json({ recovered: true });
  });
  assert.deepEqual(await (await fetcher("https://example.com")).json(), { recovered: true });
  assert.equal(attempts, 2);
  let permanentAttempts = 0;
  await assert.rejects(transport(async () => { permanentAttempts++; throw new Error("bad adapter"); })("https://example.com"), /bad adapter/);
  assert.equal(permanentAttempts, 1);
});

for (const status of [400, 401, 403, 404, 500]) {
  test(`does not retry permanent HTTP ${status}`, async () => {
    let attempts = 0;
    const response = await transport(async () => { attempts++; return new Response(null, { status }); })("https://example.com");
    assert.equal(response.status, status);
    assert.equal(attempts, 1);
  });
}

test("does not retry non-idempotent requests", async () => {
  let attempts = 0;
  const response = await transport(async () => { attempts++; return new Response(null, { status: 503 }); })("https://example.com", { method: "POST" });
  assert.equal(response.status, 503);
  assert.equal(attempts, 1);
});

test("deadline rejects injected fetch that ignores its abort signal", async () => {
  let signal;
  await assert.rejects(transport((_input, init) => {
    signal = init.signal;
    return new Promise(() => {});
  }, { requestTimeoutMs: 20 })("https://example.com"), { name: "TimeoutError" });
  assert.equal(signal.aborted, true);
});

test("deadline includes a stalled response body and cancels its source", async () => {
  let cancelled = false;
  const response = await transport(async () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode("partial")); },
    cancel() { cancelled = true; },
  })), { requestTimeoutMs: 20 })("https://example.com");
  await assert.rejects(response.text(), { name: "TimeoutError" });
  assert.equal(cancelled, true);
});

test("body errors are propagated without replaying already delivered bytes", async () => {
  let attempts = 0;
  const response = await transport(async () => {
    attempts++;
    return new Response(new ReadableStream({ start(controller) { controller.error(new TypeError("body disconnected")); } }));
  })("https://example.com");
  await assert.rejects(response.text(), /body disconnected/);
  assert.equal(attempts, 1);
});

test("Retry-After backoff uses the same deadline as fetch and body", async () => {
  let attempts = 0;
  await assert.rejects(transport(async () => {
    attempts++; return new Response(null, { status: 429, headers: { "Retry-After": "999999" } });
  }, { requestTimeoutMs: 20 })("https://example.com"), { name: "TimeoutError" });
  assert.equal(attempts, 1);
});

test("past date Retry-After allows an immediate retry", async () => {
  let attempts = 0;
  const response = await transport(async () => ++attempts === 1
    ? new Response(null, { status: 503, headers: { "Retry-After": "Thu, 01 Jan 1970 00:00:00 GMT" } })
    : new Response("ok"))("https://example.com");
  assert.equal(await response.text(), "ok");
  assert.equal(attempts, 2);
});

test("cancellation interrupts fetch, body reads, and retry backoff with the supplied reason", async () => {
  for (const phase of ["fetch", "body", "backoff"]) {
    const controller = new AbortController();
    const reason = new Error(`cancel ${phase}`);
    let attempts = 0;
    const fetcher = transport(async () => {
      attempts++;
      setImmediate(() => controller.abort(reason));
      if (phase === "fetch") return new Promise(() => {});
      if (phase === "backoff") return new Response(null, { status: 503, headers: { "Retry-After": "5" } });
      return new Response(new ReadableStream({}));
    }, { signal: controller.signal });
    await assert.rejects((async () => { const response = await fetcher("https://example.com"); await response.text(); })(), (error) => error === reason);
    assert.equal(attempts, 1);
  }
});

test("aborting queued work never starts its fetch and does not leak a rejection", async () => {
  const controller = new AbortController();
  let attempts = 0;
  const fetcher = transport(async () => {
    attempts++; return new Response(new ReadableStream({}));
  }, { signal: controller.signal });
  const first = await fetcher("https://example.com/first");
  const body = first.text();
  const second = fetcher("https://example.com/second");
  const settled = Promise.allSettled([body, second]);
  controller.abort();
  const results = await settled;
  assert.deepEqual(results.map((result) => result.status), ["rejected", "rejected"]);
  assert.equal(attempts, 1);
});

test("a pre-aborted estimator never starts target or draft requests", async () => {
  const controller = new AbortController(); controller.abort();
  let attempts = 0;
  await assert.rejects(estimateModelMemory({ modelId: "org/model", draftModel: "org/draft", signal: controller.signal, fetch: async () => {
    attempts++; return Response.json({});
  } }), { name: "AbortError" });
  assert.equal(attempts, 0);
});

test("target and draft retain the caller's retry policy without recursive wrapping", async () => {
  const attempts = [];
  await assert.rejects(estimateModelMemory({
    modelId: "org/target", draftModel: "org/draft", maxRetries: 0,
    fetch: async (input) => { attempts.push(String(input)); return new Response(null, { status: 503 }); },
  }), /503/);
  assert.deepEqual(attempts.sort(), [
    "https://huggingface.co/api/models/org/draft/revision/main",
    "https://huggingface.co/api/models/org/target/revision/main",
  ]);
});
