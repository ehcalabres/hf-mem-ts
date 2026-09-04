# hf-mem-ts

A small, zero-runtime-dependency TypeScript port of [hf-mem](https://github.com/alvarobartt/hf-mem). It estimates stored model-weight and optional cache memory using HTTP Range requests. Safetensors requests target metadata exactly; GGUF prefix requests may include some tensor payload while locating the end of the metadata. Complete model downloads are not required.

Supports canonical single and sharded Safetensors models, Diffusers components, GGUF files and sharded GGUF sets. It prefers Safetensors when a repository contains both formats; pass `ggufFile` to select GGUF explicitly.

For GGUF multimodal models, a sole projector or `mmproj-F16.gguf` is included automatically. Select another with `mmprojFile` / `--mmproj-file`, or disable projector accounting with `false` / `--no-mmproj`. Speculative-decoding draft models can be added from a second Hub repository.

## CLI

From a local checkout, install and build the package once with `npm install && npm run build`, then run it through `npx` without downloading another package:

```sh
npx --no-install . Qwen/Qwen3-1.7B --kv-cache --draft-model Qwen/Qwen3-0.6B
npx --no-install . deepseek-ai/DeepSeek-V3 --kv-cache --mla-layout compressed --max-model-len 32768
npx --no-install . Qwen/Qwen3.5-0.8B --kv-cache --max-model-len 32768 --json
npx --no-install . unsloth/Qwen3.5-0.8B-GGUF --gguf-file Qwen3.5-0.8B-Q4_0.gguf
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
  modelId: "deepseek-ai/DeepSeek-V3",
  kvCache: true,
  mlaLayout: "compressed",
  maxModelLen: 32_768,
  concurrency: 16,
});

console.log(estimate.weightsBytes, estimate.kvCacheBytes, estimate.totalBytes);
```

With a GGUF model and its automatically selected multimodal projector:

```ts
const estimate = await estimateModelMemory({
  modelId: "unsloth/Qwen3.5-0.8B-GGUF",
  ggufFile: "Qwen3.5-0.8B-Q4_0.gguf",
});
```

This GGUF example estimates weights only: Qwen3.5's GGUF recurrent/hybrid cache layout is explicitly unsupported. Use its Safetensors configuration for the supported cache model below.

With a speculative-decoding draft model:

```ts
const estimate = await estimateModelMemory({
  modelId: "Qwen/Qwen3-1.7B",
  draftModel: "Qwen/Qwen3-0.6B",
  kvCache: true, // includes both the main and draft KV caches
});
```

The main entry point uses web-standard `fetch` and has no Node imports, so it can be bundled in Svelte/SvelteKit and other browser applications. Browser calls are subject to the model's access rules and Hugging Face CORS policy. For private or gated models, passing a token from browser code exposes it to the client; prefer calling the library from a server route.

You can inject `fetch` for SvelteKit, SSR, a proxy, or tests:

```ts
const estimate = await estimateModelMemory({ modelId, fetch, kvCache: true });
```

Each model's requested `revision` (default `main`) is resolved once to a commit before listing or downloading metadata. Results preserve that requested `revision` and expose the immutable `resolvedRevision`; projectors use the target's commit and drafts resolve their own revision. Pagination is restricted to the same origin and pinned repository tree, and pagination loops are rejected.

`concurrency` defaults to 8 and bounds in-flight metadata requests **including response-body reads and projectors**, per model. Target and draft intentionally have independent limits. Sharded GGUF selections must name an existing shard in a complete, consistently numbered set; missing, duplicate, conflicting, or ambiguous sets are rejected before weight metadata requests. Projector basenames (`mmproj.gguf`, `mmproj_f16.gguf`, and `mmproj-*.gguf`) are never treated as model variants, even with `mmprojFile: false`.

The Hub estimator accepts `signal?: AbortSignal`, `requestTimeoutMs?: number` (default **30,000**, integer 1–2,147,483,647), and `maxRetries?: number` (default **2**, integer 0–10). Each request's deadline includes fetch, retry delays, and body consumption; it starts when the request acquires its per-model slot. The same cancellation and policy apply to target and draft without wrapping requests twice. Cancellation also stops queued requests and retry backoff, including with injected fetch implementations that ignore abort signals.

Only GET/HEAD requests are retried, before a response is delivered, for fetch network `TypeError` failures or HTTP 429/502/503/504. Authentication and other permanent errors are not retried. Exponential backoff starts at 250 ms; numeric/date `Retry-After` is honored up to 5 seconds, within the original deadline. A body that fails or times out is cancelled and rejected, never replayed after bytes have reached the parser. These policies apply to `estimateModelMemory`, not the lower-level parser fetch functions.

Useful lower-level functions are exported from `hf-mem-ts/safetensors`, `hf-mem-ts/gguf`, and `hf-mem-ts/kv-cache`.

## Result shape

Single Safetensors or selected GGUF results contain numeric `weightsBytes` and `kvCacheBytes`. `totalBytes` includes the target weights and KV cache plus `mmproj.bytes` and the complete `draft.totalBytes` when configured. When no GGUF file is selected, each repository quantization is returned in records keyed by filename and `totalBytes` is `null`, since those files are alternatives rather than additive weights. The `files` field contains parameter, component, and dtype breakdowns.

`parameters` counts stored tensor elements, including packed tensors and auxiliary tensors; it is not necessarily the model's logical trainable parameter count. Safetensors bytes follow each stored dtype and shape. GGUF bytes use exact GGML block layouts, including quantization scales, and require block-aligned rows. Weight totals exclude file headers, alignment padding, and runtime allocations.

Metadata requests require valid `206` / `Content-Range` responses and stop reading at their byte budget; servers that ignore Range are rejected rather than downloading the model. EOF-shortened GGUF ranges are supported. `fetchGgufMetadata` accepts a positive safe-integer `maxBytes` budget (100,000,000 by default); Safetensors headers are capped at 512 MiB and JSON configuration/index responses and repository tree pages at 32 MiB each. GGUF metadata arrays are limited to 1,000,000 elements and 16 nesting levels. Safetensors offsets, when supplied, must match the tensor's storage size and must not overlap within a file; scalar and empty tensors remain valid.

GGUF storage definitions follow [GGML's Python constants](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py) and [C block layouts](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h). For Q8_1, the C layout is authoritative: 36 bytes per 32 elements (the Python table still lists 40).

KV-cache estimation is opt-in with `kvCache: true` / `--kv-cache`. Safetensors configuration is read from `config.json`; GGUF configuration comes from embedded metadata. Each `files[name].kvCache` exposes `attentionBytes`, `stateBytes`, `convolutionBytes`, `recurrentBytes`, `layout`, per-kind layer counts, storage dtypes, allocation policy, and `assumptions`. `bytes` and the existing aggregate `kvCacheBytes` include attention **and** persistent recurrent/convolution state. Inspect these fields through the library or CLI `--json`.

The human-readable CLI report includes a separate **Cache assumptions** section with each target/draft file's layout, attention/state split, storage precisions and backend assumptions.

## Cache layouts and assumptions

Support is intentionally bounded, not a claim that every architecture or inference backend has the same cache:

- **Conventional MHA/GQA:** separate K/V tensors, using explicit `head_dim` where present, otherwise the exact integral `hidden_size / num_attention_heads`. Safetensors supports Llama, Mistral/Mixtral, Qwen2/Qwen3 (including MoE), and explicit conventional configurations without a `model_type`. Unknown named architectures and unknown layer types produce an actionable unsupported error rather than a plausible numeric fallback.
- **GGUF attention:** resolves `general.architecture` (or one unambiguous legacy namespace), honors distinct `attention.key_length` and `attention.value_length`, and falls back to query heads only when optional `head_count_kv` is absent. Model-specific MLA/recurrent and ambiguous hybrid-window GGUF layouts are rejected. Supported quantized cache rows include their block scales/minima; unaligned head dimensions fail explicitly instead of inventing a backend's padding.
- **MLA:** `mlaLayout: "compressed"` / `--mla-layout compressed` assumes one `kv_lora_rank` latent plus one `qk_rope_head_dim` shared key per token per layer. This is the default for explicit MLA configurations, **not** a universal backend guarantee. Select `"expanded"` for per-query-head keys (`qk_nope_head_dim + qk_rope_head_dim`) and values (`v_head_dim`), without retaining the latent as well. [DeepSeek-V3's config](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json) has 61 layers, rank 512 and RoPE dimension 64: at 32,768 tokens and BF16, compressed payload is **2,302,672,896 bytes**. FP8 weights do not change that BF16 cache estimate.
- **Sliding windows:** `slidingWindowPolicy: "optimized"` / `--sliding-window-policy optimized` allocates `min(context, window)` slots for each genuinely sliding layer, even in a hybrid stack. `"full-context"` allocates full-length storage in every attention layer. Explicit `layer_types` takes precedence; unrecognized types are never treated as sliding. This models allocated token slots, not transient prefill buffers, speculative rollback history or a specific engine's window-minus-one implementation.
- **Qwen3.5 Gated DeltaNet (Safetensors):** full-attention layers use ordinary KV caches; linear-attention layers instead have fixed-size convolution `[batch, 2*keyHeads*keyDim + valueHeads*valueDim, kernel]` and recurrent `[batch, valueHeads, keyDim, valueDim]` states. The [Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json) has six full-attention and eighteen linear layers. At 32,768 tokens, batch 1, BF16 attention is **402,653,184 bytes**. Its eighteen BF16 `[1,6144,4]` convolution buffers add **884,736 bytes**, and eighteen FP32 `[1,16,128,128]` recurrent buffers add **18,874,368 bytes**: **422,412,288 bytes total** under that storage policy.

Safetensors `kvCacheDtype: "auto"` uses explicit `kv_cache_dtype` / quantization cache configuration when present, otherwise configured compute precision (`dtype` / `torch_dtype`). Weight `quant_method`, weight formats and weight tensor dtypes never imply FP8 KV storage. Conflicting dtype declarations fail unless an explicit cache override resolves them. GGUF `auto` remains an explicit F16 allocation assumption, independent of weight quantization.

Low-level API migration: `estimateSafetensorsKvCache` no longer accepts a `metadata` option, and `resolveSafetensorsKvDtype` accepts only configuration and requested dtype. Remove weight-metadata arguments; weight tensor precision is not cache configuration.

Qwen3.5 convolution storage uses compute precision independently of the attention-cache override. Recurrent precision is selected by `recurrentStateDtype` / `--recurrent-state-dtype`, then `mamba_ssm_dtype`, then a documented F32 default policy. Returned `convolutionDtype` and `recurrentDtype` make both assumptions visible. **Configuration does not guarantee actual backend storage precision.** The [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) and [cache implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py) evolve: a CPU eager calibration with torch 2.13.0 / Transformers 5.13.0 stored recurrent buffers in BF16 even with `mamba_ssm_dtype="float32"`. Set `recurrentStateDtype: "bfloat16"` to model that backend; FP32 storage remains appropriate only where actually used. Other recurrent families are not covered by the Qwen3.5 formula.

These are resident payload estimates, not peak process RAM/VRAM. No generic fudge factor is applied. Allocator/page alignment, quantized padding and residual/scaling buffers, prefill/decode workspace, offloading placement, prefix/cross-layer sharing, rollback history and multimodal cross-attention caches can change real allocation. Known unsupported sharing/layout metadata is rejected; metadata cannot expose every backend optimization. Cache options are inherited by draft models and can be overridden in `DraftModelOptions`.

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
