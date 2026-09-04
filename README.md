# hf-mem-ts

A small, zero-runtime-dependency TypeScript port of [hf-mem](https://github.com/alvarobartt/hf-mem). It estimates model-weight and KV-cache memory by reading only metadata through HTTP Range requests—model weights are never downloaded.

Supports canonical single and sharded Safetensors models, Diffusers components, GGUF files and sharded GGUF sets. It prefers Safetensors when a repository contains both formats; pass `ggufFile` to select GGUF explicitly.

For GGUF multimodal models, a sole projector or `mmproj-F16.gguf` is included automatically. Select another with `mmprojFile` / `--mmproj-file`, or disable projector accounting with `false` / `--no-mmproj`. Speculative-decoding draft models can be added from a second Hub repository.

## CLI

From a local checkout, install and build the package once with `npm install && npm run build`, then run it through `npx` without downloading another package:

```sh
npx --no-install . google/gemma-4-E2B-it --kv-cache --draft-model google/gemma-4-E2B-it-assistant
npx --no-install . zai-org/GLM-5.3 --kv-cache --concurrency 16 --max-model-len 248000
npx --no-install . unsloth/Qwen3.5-0.8B-GGUF --gguf-file Qwen3.5-0.8B-Q4_0.gguf --kv-cache
npx --no-install . black-forest-labs/FLUX.2-klein-4B --concurrency 16
```

After publication, replace `npx --no-install .` with `npx hf-mem-ts` to run the package directly from npm.

Use `--max-model-len`, `--batch-size`, and `--kv-cache-dtype` to change cache assumptions. Metadata for separate files is fetched concurrently (8 tasks per model by default); use `--concurrency` to tune the limit. Authentication uses `HF_TOKEN`, or `--token` (avoid the latter in shared shell history).

## Diffusers

Diffusers repositories are detected through `model_index.json`. Every referenced component with canonical Safetensors weights is included, such as `transformer`, `text_encoder`, `text_encoder_2`, and `vae`. Components can independently use a single weights file or a sharded index; their metadata requests still respect the configured concurrency limit.

The report shows the complete model-weight total and a row for each component. Alternative root-level checkpoints are ignored when they are not referenced by `model_index.json`, preventing an equivalent native or ComfyUI transformer checkpoint from being counted again alongside its Diffusers representation. The estimate covers resident tensor weights, not activations, framework/allocator overhead, or the effects of CPU offloading.

## TypeScript / JavaScript

```ts
import { estimateModelMemory } from "hf-mem-ts";

const estimate = await estimateModelMemory({
  modelId: "zai-org/GLM-5.3",
  kvCache: true,
  maxModelLen: 248_000,
  concurrency: 16,
});

console.log(estimate.weightsBytes, estimate.kvCacheBytes, estimate.totalBytes);
```

With a GGUF model and its automatically selected multimodal projector:

```ts
const estimate = await estimateModelMemory({
  modelId: "unsloth/Qwen3.5-0.8B-GGUF",
  ggufFile: "Qwen3.5-0.8B-Q4_0.gguf",
  kvCache: true,
});
```

With a speculative-decoding draft model:

```ts
const estimate = await estimateModelMemory({
  modelId: "google/gemma-4-E2B-it",
  draftModel: "google/gemma-4-E2B-it-assistant",
  kvCache: true, // includes both the main and draft KV caches
});
```

The main entry point uses web-standard `fetch` and has no Node imports, so it can be bundled in Svelte/SvelteKit and other browser applications. Browser calls are subject to the model's access rules and Hugging Face CORS policy. For private or gated models, passing a token from browser code exposes it to the client; prefer calling the library from a server route.

You can inject `fetch` for SvelteKit, SSR, a proxy, or tests:

```ts
const estimate = await estimateModelMemory({ modelId, fetch, kvCache: true });
```

Useful lower-level functions are exported from `hf-mem-ts/safetensors`, `hf-mem-ts/gguf`, and `hf-mem-ts/kv-cache`.

## Result shape

Single Safetensors or selected GGUF results contain numeric `weightsBytes` and `kvCacheBytes`. `totalBytes` includes the target weights and KV cache plus `mmproj.bytes` and the complete `draft.totalBytes` when configured. When no GGUF file is selected, each repository quantization is returned in records keyed by filename and `totalBytes` is `null`, since those files are alternatives rather than additive weights. The `files` field contains parameter, component, and dtype breakdowns.

`parameters` counts stored tensor elements, including packed tensors and auxiliary tensors; it is not necessarily the model's logical trainable parameter count. Safetensors bytes follow each stored dtype and shape. GGUF bytes use exact GGML block layouts, including quantization scales, and require block-aligned rows. Weight totals exclude file headers, alignment padding, and runtime allocations.

Metadata requests require valid `206` / `Content-Range` responses and stop reading at their byte budget; servers that ignore Range are rejected rather than downloading the model. EOF-shortened GGUF ranges are supported. `fetchGgufMetadata` accepts a positive safe-integer `maxBytes` budget (100,000,000 by default); Safetensors headers are capped at 512 MiB and JSON configuration/index responses at 32 MiB. GGUF metadata arrays are limited to 1,000,000 elements and 16 nesting levels. Safetensors offsets, when supplied, must match the tensor's storage size and must not overlap within a file; scalar and empty tensors remain valid.

GGUF storage definitions follow [GGML's Python constants](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py) and [C block layouts](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h). For Q8_1, the C layout is authoritative: 36 bytes per 32 elements (the Python table still lists 40).

KV-cache estimation is opt-in with `kvCache: true` / `--kv-cache`. Safetensors configuration is read from `config.json`; GGUF configuration comes from embedded metadata. `auto` uses the configured Safetensors precision and falls back to F16 for GGUF.

## Development

```sh
npm install
npm test
npm pack --dry-run
```

Requires Node.js 18 or newer for the CLI. The library works in modern runtimes with `fetch`, `BigInt`, and `DataView`.
