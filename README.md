# hf-mem-ts

A zero-dependency TypeScript port of [hf-mem](https://github.com/alvarobartt/hf-mem) v1.0.0. It estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights: it reads the Safetensors, GGUF and ONNX metadata via HTTP Range requests and adds up the weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime).

The port mirrors hf-mem module by module, so estimates, warnings, errors and outputs match `hf-mem` for the same arguments. See the [hf-mem README](https://github.com/alvarobartt/hf-mem#readme) for how the estimate works, what each field means, and its limitations.

## CLI

From a local checkout, build once and run the CLI with the same arguments as `hf-mem`:

```sh
npm install && npm run build
npx --no-install . Qwen/Qwen3-8B --max-model-len 32768 --batch-size 4
npx --no-install . Qwen/Qwen3-8B --gpu-memory 8x80 -o json
npx --no-install . unsloth/Qwen3-8B-GGUF:Q4_K_M
npx --no-install . onnx-community/Qwen3-0.6B-ONNX --file onnx/model_q4f16.onnx
npx --no-install . BAAI/bge-base-en-v1.5 --dtype float16 --max-num-batched-tokens 2048 --max-model-len 512
npx --no-install . ./path/to/model
```

Options: `--revision`, `--file`, `--format`, `--max-model-len`, `--batch-size`, `--kv-cache-dtype`, `--gpu-memory`, `--gpu-memory-utilization`, `--max-num-batched-tokens`, `--dtype`, `--hf-token`, `--color`, `-o/--output {table,json,markdown}`. Gated and private models read the token from `--hf-token`, `HF_TOKEN`, or the token saved by `hf auth login`; `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE` and `HF_HUB_OFFLINE` behave as in hf-mem.

## TypeScript / JavaScript

```ts
import { run, Estimate } from "hf-mem-ts";

const result = await run("Qwen/Qwen3-8B", { maxModelLen: 32768, gpuMemory: "80" });
if (result instanceof Estimate) {
  console.log(result.total, result.weights.bytes, result.kvCache?.bytes, result.overhead?.bytes, result.fit?.fits);
}
console.log(result.toDict()); // same shape as `hf-mem -o json`
```

`run` takes the hf-mem options in camelCase (`revision`, `file`, `format`, `maxModelLen`, `batchSize`, `kvCacheDtype`, `hfToken`, `gpuMemory`, `gpuMemoryUtilization`, `maxNumBatchedTokens`, `dtype`, `progress`), plus `fetch` to inject a custom `fetch` (proxies, SSR, tests) and `signal` to cancel pending requests. It returns an `Estimate`, or a `Listing` with one `Estimate` per file for GGUF and ONNX repositories. `table` and `markdown` render a result as the CLI does.

The library only loads Node built-ins when they exist, for local paths, the Hugging Face cache and the saved token, so it also runs in browsers and edge runtimes against the Hub. Browser calls are subject to the Hub's CORS policy; keep tokens on the server.

## Development

```sh
npm install
npm test
```

Requires Node.js 18.17 or newer. The tests are ports of the hf-mem test suite and run against local fixture files, without network access.
