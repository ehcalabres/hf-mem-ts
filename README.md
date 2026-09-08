# hf-mem-ts

A small, zero-runtime-dependency TypeScript port of [hf-mem](https://github.com/alvarobartt/hf-mem). It estimates stored model-weight and optional cache memory using HTTP Range requests. Safetensors requests target metadata exactly; GGUF prefix requests may include some tensor payload while locating the end of the metadata. Complete model downloads are not required.

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

KV-cache estimation is opt-in with `kvCache: true` / `--kv-cache`. Safetensors configuration is read from `config.json`; GGUF configuration comes from embedded metadata. `auto` uses the configured Safetensors precision and falls back to F16 for GGUF.

### Interpreting an estimate

`totalBytes` is the sum of counted resident weights, the requested cache estimate, and selected accessories. It is **not peak inference RAM/VRAM** or a guarantee that a model will fit. When cache estimation is disabled, cache memory is absent from the total, not zero in the running model.

The estimate excludes activations, prefill/decode workspaces, framework and allocator overhead, CUDA graphs, runtime weight conversion/repacking, device offloading, and distributed replication. Stored weight precision need not equal runtime precision. Cache behavior is architecture- and backend-dependent; use explicit cache settings and treat unsupported layouts as unsupported rather than extrapolating ordinary attention.

The `parameters` fields count elements represented by stored tensors, including auxiliary tensors. Packed quantized Safetensors elements are not necessarily one logical model parameter each. Do not use this field alone to infer the advertised parameter count of a quantized model.

The text report displays cache dtype, tokens per sequence, and sequence count when a cache estimate is present. GGUF alternatives are never summed. An unresolved draft lists its repository and alternatives and asks for `--draft-gguf-file`; unresolved target selection uses `--gguf-file`.

For deployment sizing, measure your intended engine with the same model revision, dtype, batch/sequence lengths, and device placement. Record persistent weight/cache allocations separately from prefill and decode peaks. Metadata arithmetic and CPU allocation checks cannot establish GPU allocator overhead or a universal safety margin.

## Development

```sh
npm install
npm test
npm pack --dry-run
```

Requires Node.js 18.17 or newer for the CLI. CI checks the minimum runtime and Node.js 20, 22, and 24; use a currently maintained Node.js release for deployment. The library works in modern runtimes with `fetch`, `BigInt`, and `DataView`.

`npm test` builds the project and runs deterministic regressions, CLI subprocess checks, and a packed-consumer smoke test. The consumer test installs the tarball in an isolated directory, runs its executable, and compiles and executes imports from every public entry point. These checks do not need Hugging Face credentials or download model weights.
