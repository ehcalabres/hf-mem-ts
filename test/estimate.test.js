import { test } from "node:test";
import assert from "node:assert/strict";

import { arun, run } from "../dist/estimate.js";

for (const asynchronous of [false, true]) {
  for (const [options, message] of [
    [{ batchSize: 0 }, "must be positive"],
    [{ batchSize: -1 }, "must be positive"],
    [{ maxModelLen: 0 }, "must be positive"],
    [{ maxModelLen: -1 }, "must be positive"],
    [{ maxNumBatchedTokens: 0 }, "must be positive"],
    [{ maxNumBatchedTokens: -1 }, "must be positive"],
    [{ gpuMemoryUtilization: 0 }, "must be in"],
    [{ gpuMemoryUtilization: -0.1 }, "must be in"],
    [{ gpuMemoryUtilization: 1.1 }, "must be in"],
    [{ gpuMemoryUtilization: NaN }, "must be in"],
    [{ gpuMemoryUtilization: Infinity }, "must be in"],
    [{ dtype: "float8" }, "--dtype"],
  ]) {
    test(`api rejects invalid options before opening a model [${JSON.stringify(options)}-${asynchronous}]`, async () => {
      const unexpected = async () => assert.fail("Invalid options must fail before opening the model");
      await assert.rejects((asynchronous ? arun : run)("org/model", { ...options, fetch: unexpected }), (e) => {
        assert.ok(e instanceof Error);
        assert.match(e.message, new RegExp(message));
        return true;
      });
    });
  }
}
