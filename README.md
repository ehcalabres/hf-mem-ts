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
npx --no-install . unsloth/Qwen3.5-0.8B-GGUF --gguf-file Qwen3.5-0.8B-Q4_0.gguf --kv-cache
npx --no-install . Qwen/Qwen3.5-0.8B --kv-cache --tensor-parallel-size 2 --json
npx --no-install . black-forest-labs/FLUX.2-klein-4B --concurrency 16
```

After publication, replace `npx --no-install .` with `npx hf-mem-ts` to run the package directly from npm.

With `--kv-cache` and no `--tensor-parallel-size`, both text and JSON compare **TP 1, 2, 4, and 8** using the same fetched metadata. These are alternatives, not additive allocations; equal-valued alternatives are still returned. Specify `--tensor-parallel-size 2` (or another positive integer) for a single estimate. Without `--kv-cache`, no cache comparison or cache-only configuration fetch is performed.

Use `--max-model-len`, `--batch-size`, and `--kv-cache-dtype` to change cache assumptions. Metadata for separate files is fetched concurrently (8 tasks per model by default); use `--concurrency` to tune the limit. Authentication uses `HF_TOKEN`, or `--token` (avoid the latter in shared shell history).

Use `--request-timeout-ms` to set the per-request deadline in milliseconds (default `30000`, range `1`–`2147483647`), including retry delays and response-body reads. Use `--max-retries` to set transient request retries (default `2`, range `0`–`10`); `0` disables retries. Both settings apply to the target and any draft model, using the same policy as the library API.

```sh
npx --no-install . HuggingFaceTB/SmolLM2-135M --request-timeout-ms 60000 --max-retries 0
```

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

console.log(estimate.weightsBytes, estimate.kvCacheBytesByTp, estimate.totalBytesByTp);
```

With a GGUF model and its automatically selected multimodal projector:

```ts
const estimate = await estimateModelMemory({
  modelId: "unsloth/Qwen3.5-0.8B-GGUF",
  ggufFile: "Qwen3.5-0.8B-Q4_0.gguf",
  kvCache: true,
});
```

GGUF recurrent/hybrid metadata now produces an approximate cache estimate too. Inspect the reported assumptions: GGUF state descriptors do not always identify the exact backend's recurrent layout.

With a speculative-decoding draft model:

```ts
const estimate = await estimateModelMemory({
  modelId: "Qwen/Qwen3-1.7B",
  draftModel: "Qwen/Qwen3-0.6B",
  kvCache: true, // includes both the main and draft KV caches
});
```

The draft inherits the target's TP choice: omission on both compares matching target/draft TP 1, 2, 4, and 8, while an explicit target TP produces matching single estimates. To hold the draft at one TP across target scenarios, use the library's object form:

```ts
const estimate = await estimateModelMemory({
  modelId: "Qwen/Qwen3-1.7B",
  kvCache: true,
  draftModel: { modelId: "Qwen/Qwen3-0.6B", tensorParallelSize: 1 },
});
// totalBytesByTp["4"] includes target TP 4 + the fixed draft TP 1.
```

Explicit target and draft TP values may differ. Same-model drafts with identical settings reuse metadata; a draft fixed to one of the target's four comparison TPs can reuse that computed scenario rather than fetching the model again.

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

`weightsBytes` is numeric for Safetensors and a selected GGUF (including a selected shard set). Without `ggufFile`, it is a filename-to-bytes record of alternative GGUF quantizations. Cache and total fields distinguish three modes:

| Mode | `kvCacheBytes` / `totalBytes` | `kvCacheBytesByTp` / `totalBytesByTp` | Per-file cache |
| --- | --- | --- | --- |
| Cache enabled, TP omitted | Both `null` | Maps with all four keys `"1"`, `"2"`, `"4"`, `"8"` | `kvCache: null`; `kvCacheByTp` maps each TP to a full `KvCacheEstimate` |
| Cache enabled, explicit TP | Single cache value and combined total | Both `null` | Single `kvCache`; `kvCacheByTp: null` |
| Cache disabled | `kvCacheBytes: null`; weight/accessory total unchanged | Both `null` | Both `kvCache` and `kvCacheByTp` are `null` |

For Safetensors or selected GGUF, each `kvCacheBytesByTp[tp]` is numeric. Each `totalBytesByTp[tp]` includes target weights, that TP's target cache, the selected `mmproj.bytes`, and the complete draft scenario (matching TP or its explicitly fixed TP). Read these combined totals rather than recomputing just target weights plus cache. No scalar TP 1 result is substituted for a comparison.

For example, this **synthetic, abridged JSON** shows the comparison structure (other result fields and cache-detail fields are omitted):

```json
{
  "weightsBytes": 4,
  "kvCacheBytes": null,
  "totalBytes": null,
  "kvCacheBytesByTp": { "1": 512, "2": 1024, "4": 2048, "8": 4096 },
  "totalBytesByTp": { "1": 516, "2": 1028, "4": 2052, "8": 4100 },
  "files": {
    "safetensors": {
      "kvCache": null,
      "kvCacheByTp": {
        "1": { "bytes": 512, "tensorParallelSize": 1 },
        "2": { "bytes": 1024, "tensorParallelSize": 2 },
        "4": { "bytes": 2048, "tensorParallelSize": 4 },
        "8": { "bytes": 4096, "tensorParallelSize": 8 }
      }
    }
  }
}
```

For **unselected GGUF**, each `kvCacheBytesByTp[tp]` is instead a filename-to-cache-bytes record, preserving both axes; for example, `"1": { "model-Q4.gguf": 512, "model-Q8.gguf": 512 }`. `totalBytesByTp` is `{ "1": null, "2": null, "4": null, "8": null }`, because quantization alternatives are never summed. Each `files[filename].kvCacheByTp` still contains all four full cache estimates. With explicit TP, the existing `kvCacheBytes` filename record is used instead and `totalBytes` is `null`.

An unresolved draft GGUF selection also makes the corresponding combined totals `null`; target cache values, projector accounting, and the draft's own filename alternatives remain available. Select `ggufFile` / `--gguf-file` and, when needed, `draftModel.ggufFile` / `--draft-gguf-file` to obtain numeric combined totals. The `files` field retains parameter, component, and dtype breakdowns in every mode.

`parameters` counts stored tensor elements, including packed tensors and auxiliary tensors; it is not necessarily the model's logical trainable parameter count. Safetensors bytes follow each stored dtype and shape. GGUF bytes use exact GGML block layouts, including quantization scales, and require block-aligned rows. Weight totals exclude file headers, alignment padding, and runtime allocations.

Metadata requests require valid `206` / `Content-Range` responses and stop reading at their byte budget; servers that ignore Range are rejected rather than downloading the model. EOF-shortened GGUF ranges are supported. `fetchGgufMetadata` accepts a positive safe-integer `maxBytes` budget (100,000,000 by default); Safetensors headers are capped at 512 MiB and JSON configuration/index responses and repository tree pages at 32 MiB each. GGUF metadata arrays are limited to 1,000,000 elements and 16 nesting levels. Safetensors offsets, when supplied, must match the tensor's storage size and must not overlap within a file; scalar and empty tensors remain valid.

GGUF storage definitions follow [GGML's Python constants](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py) and [C block layouts](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h). For Q8_1, the C layout is authoritative: 36 bytes per 32 elements (the Python table still lists 40).

KV-cache estimation is opt-in with `kvCache: true` / `--kv-cache`. Safetensors configuration is read from `config.json`; GGUF configuration comes from embedded metadata. `auto` uses the configured Safetensors precision and falls back to F16 for GGUF.

Each `files[name].kvCache` (explicit TP) or `files[name].kvCacheByTp[tp]` (comparison) exposes `attentionBytes`, `indexerBytes` (included in attention), `stateBytes`, `convolutionBytes`, `recurrentBytes`, generic `layout`, `tensorParallelSize`, per-kind layer counts, storage dtypes, allocation policy, `approximate`, and `assumptions`. `bytes` and aggregate `kvCacheBytes` / `kvCacheBytesByTp[tp]` are **tensor payload**, not occupied engine-pool memory. They include attention and one persistent recurrent/convolution state per sequence (or the disclosed proxy when state geometry is incomplete). `approximate: true` identifies defaults, structural proxies, or packing assumptions; `false` still does not guarantee exact runtime memory. Inspect these fields through the library or CLI `--json`.

The human-readable CLI report shows compact TP rows with target cache, draft cache when included (labeled when held at a fixed TP), and combined total. Its **Cache assumptions** section summarizes each target/draft file and deduplicates assumptions shared across TP alternatives; `--json` retains every full per-TP breakdown. Explicit TP keeps the detailed single-estimate report.

## Cache layouts and assumptions

The estimator is **metadata-driven, not an architecture registry**. `model_type` never gates support; GGUF's `general.architecture` selects a metadata namespace, not a whitelist. An unfamiliar model with usable dimensions can be estimated without a code update. Known metadata capabilities refine the estimate; otherwise the result uses an explicit approximation rather than rejecting the model.

- **Dimensions and precision:** normalize nested text/decoder configurations and common aliases such as `n_layer`/`n_layers`, `n_embd`/`d_model`, `n_head`/`n_heads`, and nested `attn_config.kv_n_heads`. Prefer explicit KV heads and separate key/value dimensions, including per-layer and global geometry. MQA flags imply one KV head; otherwise missing KV heads use ordinary MHA. Missing query heads use the full hidden width. Fractional derived head dimensions round upward.
- **Missing metadata:** context defaults to **4096 tokens** and precision to **F16**, both recorded as assumptions. Explicit options override these defaults. Invalid options, unsafe arithmetic, and configurations without enough layer/dimension information for a meaningful estimate still fail. No arbitrary model-size or layer-count defaults are invented.
- **Unknown layouts:** unknown layer labels use a full-context attention proxy. Incomplete recurrent geometry uses the same attention-equivalent proxy rather than silently counting zero state. This can substantially overestimate recurrent memory; it is not a universal upper bound. Sharing hints assume independent buffers and warn about possible overestimation. Encoder-decoder configurations approximate an additional cross-attention cache using the requested context as source length.
- **MLA capability fields:** `kv_lora_rank` and RoPE dimensions select shared compressed storage by default; `mlaLayout: "expanded"` / `--mla-layout expanded` uses separate query-head K/V projections. This does not depend on a DeepSeek/Kimi/GLM model name. Missing RoPE dimensions assume no extra RoPE key and mark the estimate approximate. Indexer-key dimensions add a per-token index vector approximation. For example, [DeepSeek-V3 metadata](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json) yields **2,302,672,896 bytes** of compressed BF16 cache at 32,768 tokens and TP 1.
- **Indexed attention:** `indexer_head_dim`, `indexer_kv_heads`, and `indexer_compress_ratio` (and canonical `index_*` equivalents) describe key-only caches, not query-head storage. Compressed metadata counts `ceil(context / ratio)` keys plus one raw group of `ratio` rows, without speculative or cached-position tails. `indexer_n_heads` is query-side and does not multiply persistent cache. The indexer byte total is reported separately within `attentionBytes`.
- **Tensor parallelism:** with cache enabled, high-level `estimateModelMemory` / CLI omission of `tensorParallelSize` / `--tensor-parallel-size` compares 1, 2, 4, and 8. An explicit TP requests just that estimate. The low-level `estimateSafetensorsKvCache` and `estimateGgufKvCache` remain single-estimate primitives and default to **TP 1**. Attention KV heads are divided into whole heads per rank; fewer KV heads than ranks require replication. Compressed MLA latents and single-head index keys are replicated across ranks. Linear recurrent head dimensions are sharded/rounded across ranks. Returned bytes sum all ranks **within one TP scenario**, never across alternative TPs; data/pipeline/context parallelism, replicated auxiliary state and engine padding are not modeled. This assumes the usual vLLM-style cache distribution and does not validate every backend's topology restrictions.
- **Sliding windows:** explicit `layer_types`, `sliding_window_pattern`, and per-layer GGUF schedules select windowed/full layers. `slidingWindowPolicy: "optimized"` uses `min(context, window)`; `"full-context"` uses the full context. Missing window width uses full context with a warning. Short per-layer schedules repeat/truncate with a warning; numeric `max_window_layers` assumes the first layers are windowed. Transient prefill and boundary buffers are excluded.
- **Recurrent capability fields:** complete `linear_*` dimensions imply convolution history and per-value-head key-by-value state matrices, regardless of model name. SSM fields (`state_size`/`d_state`, `intermediate_size`/`d_inner` or `hidden_size * expand`, `conv_kernel`/`d_conv`, optional groups) imply a generic SSM state/history estimate. Missing SSM expansion/history use 2 and 4 respectively with warnings. GGUF's overloaded state fields use this generic approximation, not a model-specific interpretation of `time_step_rank`. For [Qwen3.5-0.8B metadata](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json), 32,768 tokens at TP 1 yield **402,653,184 attention bytes** plus **19,759,104 state bytes** under BF16 convolution / FP32 recurrent storage.
- **Quantized GGUF cache:** use the format's effective bits per element, including block scales, and round to whole bytes. Nonaligned rows and unfamiliar cache packing remain estimable; assumptions explicitly exclude exact backend padding and dtype support. Only attention storage is rescaled; recurrent state precision stays independent.

`kvCacheDtype: "auto"` uses explicit cache precision first, then compute precision, then F16. Weight quantization never implies cache quantization. Conflicting recognized dtype declarations choose the wider storage width and mark the estimate approximate; unknown metadata precision falls back with a warning. Explicit invalid dtype options are still errors. GGUF auto uses explicit cache dtype metadata or F16.

Low-level API migration: `estimateSafetensorsKvCache` no longer accepts a `metadata` option, and `resolveSafetensorsKvDtype` accepts only configuration and requested dtype. Remove weight-metadata arguments; weight tensor precision is not cache configuration.

Convolution storage uses compute precision independently of attention precision. Recurrent precision uses `recurrentStateDtype` / `--recurrent-state-dtype`, then `mamba_ssm_dtype`/`state_dtype`, then F32. **Configuration does not guarantee backend storage precision.** A no-download CPU calibration with torch 2.13.0 / Transformers 5.13.0 stored a tiny Qwen3.5 model's recurrent buffers in BF16 even with an FP32 config hint. With an explicit BF16 recurrent override, the estimator matched actual cache buffers at 7 and 8 tokens (992 and 1024 bytes). These measurements calibrate that backend/layout, not every architecture or peak GPU memory.

These are resident payload estimates, not peak process RAM/VRAM. Defaults and structural proxies are disclosed rather than hidden behind a generic percentage margin. Allocator/page alignment, quantized scale/residual buffers, prefill/decode workspace, offloading, prefix/cross-layer sharing and source lengths can change allocation in either direction. Generic `layout` values are `attention`, `mla-compressed`, `mla-expanded`, `hybrid`, and `recurrent`; the model-specific `qwen3.5-hybrid` value has been removed. Options are inherited by draft models and can be overridden in `DraftModelOptions`.

## Interpreting an estimate

`totalBytes` (single estimate) or each `totalBytesByTp[tp]` (comparison) is the sum of counted resident weights, the requested cache estimate, and selected accessories including the complete draft scenario. It is **not peak inference RAM/VRAM** or a guarantee that a model will fit. When cache estimation is disabled, cache memory is absent from the total, not zero in the running model.

The estimate excludes activations, prefill/decode workspaces, framework and allocator overhead, CUDA graphs, runtime weight conversion/repacking, device offloading, and distributed replication. Stored weight precision need not equal runtime precision. Use explicit cache settings where known and inspect `approximate` and `assumptions` before using a result for capacity planning.

The `parameters` fields count elements represented by stored tensors, including auxiliary tensors. Packed quantized Safetensors elements are not necessarily one logical model parameter each. Do not use this field alone to infer the advertised parameter count of a quantized model.

The text report displays cache dtype, tokens per sequence, and sequence count when a cache estimate is present. GGUF alternatives are never summed. An unresolved draft lists its repository and alternatives and asks for `--draft-gguf-file`; unresolved target selection uses `--gguf-file`.

For deployment sizing, measure your intended engine with the same model revision, dtype, batch/sequence lengths, and device placement. Record persistent weight/cache allocations separately from prefill and decode peaks. Metadata arithmetic and CPU allocation checks cannot establish GPU allocator overhead or a universal safety margin.

### Comparing with vLLM runtime measurements

Do not compare tensor payload directly with `vllm:kv_cache_usage_perc × cache-pool capacity`. The latter is **occupied-pool attribution**: each referenced block ID consumes the backing width of a heterogeneous cache group, even if useful tensors occupy fewer bytes. vLLM's pool is preallocated; occupancy is not a new GPU allocation. Its reported token capacity is also group-aware, so utilization-scaled token slots need not equal prompt plus generation tokens.

For hybrid models, [vLLM 0.29's cache specs](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/kv_cache_interface.py) and [hybrid allocation](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/kv_cache_utils.py) include page padding, shared-pool group widths and recurrent checkpoints. These cannot be inferred uniquely from Hugging Face model metadata. The CLI explicitly reports that occupied vLLM pool memory is not predicted for recurrent models; it does not apply an empirical multiplier.

For example, at 261,956 tokens Qwen3.8-Flash-Next's unsharded main K/V payload is 6,437,830,656 bytes. Its compressed index keys plus raw ring add approximately 201,194,496 bytes; generic full-kernel recurrent/convolution state adds 116,195,328 bytes. The corrected tensor payload is approximately **6.755 GB**, not a prediction of occupied vLLM pool memory. TP=4 increases that payload to approximately **13.797 GB** through KV/index replication, before pool overhead. Neither topology can be inferred from a model ID.

To reconcile a measured run, retain its full serving command and resolved cache configuration: TP/DP/PP sizes, cache/state dtypes, effective block sizes, hybrid cache groups, Mamba cache mode/checkpoint settings, prefix/speculative settings, and whether capacity represents a per-rank buffer, total pool, or startup memory budget. Qwen's [QSA implementation](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/models/qwen4_exp/common/qsa_cache.py) supplies the compressed-index/ring shapes used above; actual recurrent history length, position tails and pool allocation remain backend-dependent.

## Development

```sh
npm install
npm test
npm pack --dry-run
```

Requires Node.js 18.17 or newer for the CLI. CI checks the minimum runtime and Node.js 20, 22, and 24; use a currently maintained Node.js release for deployment. The library works in modern runtimes with `fetch`, `BigInt`, and `DataView`.

`npm test` builds the project and runs deterministic regressions, CLI subprocess checks, and a packed-consumer smoke test. The consumer test installs the tarball in an isolated directory, runs its executable, and compiles and executes imports from every public entry point. These checks do not need Hugging Face credentials or download model weights.
