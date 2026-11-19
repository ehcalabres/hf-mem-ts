import * as config_ from "./config.js";
import type { Config } from "./config.js";
import * as gguf from "./gguf.js";
import * as onnx from "./onnx.js";
import * as overhead_ from "./overhead.js";
import { dirname, fixed, isInt, isRecord, truthy } from "./py.js";
import { Estimate, type Format, KVCache, Listing } from "./result.js";
import * as safetensors from "./safetensors.js";
import { type FetchLike, HubError, type HubOptions, openSource, type Progress, type Source } from "./source.js";
import { DTYPES, FLOAT_BYTES, Weights } from "./weights.js";

export const FORMATS: Format[] = ["safetensors", "gguf", "onnx"];

export interface RunOptions {
  revision?: string | null;
  file?: string | null;
  maxModelLen?: number | null;
  batchSize?: number;
  kvCacheDtype?: string;
  hfToken?: string | null;
  gpuMemory?: string | null;
  gpuMemoryUtilization?: number;
  maxNumBatchedTokens?: number | null;
  progress?: Progress | null;
  format?: string | null;
  dtype?: string | null;
  /** Custom `fetch`, e.g. for proxies or server-side rendering. */
  fetch?: FetchLike;
  /** Cancels pending Hub requests. */
  signal?: AbortSignal;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function ggufKv(
  metadata: Config,
  maxModelLen: number | null,
  batchSize: number,
  dtype: string,
): [KVCache, number, number] | null {
  const arch = metadata["general.architecture"];
  const contextLength = metadata[`${arch}.context_length`];
  const length = maxModelLen || contextLength;
  if (!isInt(length)) return null;
  const requested = dtype;
  const upper = dtype === "auto" ? "F16" : dtype.toUpperCase();
  if (!(gguf.CACHE_TYPES as readonly string[]).includes(upper)) {
    throw new Error(
      `\`--kv-cache-dtype=${requested}\` is not a llama.cpp KV cache type, use one of ` +
        `${gguf.CACHE_TYPES.map((t) => t.toLowerCase()).join(", ")}.`,
    );
  }
  const result = gguf.kvCache(metadata, length, batchSize, upper);
  if (result === null) return null;
  const halfLength = Math.max(1, Math.floor(length / 2));
  const half = gguf.kvCache(metadata, halfLength, batchSize, upper)!;
  const slope = length > 1 ? (result[0] - half[0]) / (length - halfLength) : 0.0;
  return [new KVCache(result[0], upper, length, batchSize, result[1]), contextLength || length, slope];
}

interface FitExtras {
  kvHeads?: number | null;
  padding?: ((n: number) => number) | null;
  kvBytes?: ((length: number, n: number) => number) | null;
  maxGpus?: number | null;
}

function applyFit(
  estimate: Estimate,
  contextLength: number,
  slope: number,
  gpuMemory: string | null,
  utilization: number,
  extras: FitExtras = {},
): void {
  if (gpuMemory === null) return;
  const kv = estimate.kvCache;
  estimate.fit = overhead_.fit({
    weights: estimate.weights.bytes,
    kvFixed: kv ? Math.trunc(kv.bytes - slope * kv.maxModelLen) : 0,
    kvPerToken: slope,
    maxModelLen: kv ? kv.maxModelLen : 0,
    contextLength: Math.max(contextLength, kv ? kv.maxModelLen : 0),
    overhead: estimate.overhead ? estimate.overhead.bytes : 0,
    gpuMemory,
    utilization,
    kvHeads: extras.kvHeads ?? null,
    context: estimate.overhead ? estimate.overhead.context : 0,
    padding: extras.padding ?? null,
    kvBytes: extras.kvBytes ?? null,
    maxGpus: extras.maxGpus ?? null,
  });
}

function mxfp4Padding(config: Config, text: Config, weights: Weights): ((n: number) => number) | null {
  const quantization: Config = config.quantization_config || text.quantization_config || {};
  if (quantization.quant_method !== "mxfp4" && quantization.expert_dtype !== "fp4") return null;
  let experts = 0;
  for (const d of ["U8", "I8", "F8_E8M0", "MXFP4"]) experts += weights.dtypes.get(d)?.bytes ?? 0;
  const intermediate = text.moe_intermediate_size || text.intermediate_size || 0;
  if (!experts || !intermediate || !text.hidden_size) return null;
  return overhead_.mxfp4Padding(experts, intermediate, text.hidden_size);
}

async function estimateGguf(
  source: Source,
  file: string | null,
  maxModelLen: number | null,
  batchSize: number,
  kvDtype: string,
  warnings: string[],
  gpuMemory: string | null,
  utilization: number,
): Promise<Estimate | Listing> {
  const groups = gguf.groupFiles(gguf.ggufPaths(source.files));
  if (!groups.size) throw new Error(noWeights(source, "gguf"));

  if (file !== null || groups.size === 1) {
    const [name, paths] = file !== null ? gguf.selectFile(groups, file) : [...groups.entries()][0]!;
    const projector = gguf.mmproj(source.files, name);
    const [header, ...extra] = await Promise.all([
      gguf.readGroup(source, paths),
      ...(projector ? [gguf.readGroup(source, [projector])] : []),
    ]);
    const kv = ggufKv(header!.metadata, maxModelLen, batchSize, kvDtype);
    const mmproj = extra[0];
    const estimate = new Estimate({
      model: source.name,
      revision: source.revision,
      url: source.url,
      format: "gguf",
      file: name,
      weights: mmproj ? new Weights().merge(header!.weights).merge(mmproj.weights) : header!.weights,
      components: mmproj
        ? new Map([
            ["Model", header!.weights],
            ["mmproj", mmproj.weights],
          ])
        : new Map(),
      offloaded: header!.offloaded.bytes ? header!.offloaded : null,
      kvCache: kv ? kv[0] : null,
      overhead: kv ? overhead_.llamaCpp(mmproj ? mmproj.metadata : null) : null,
      warnings,
    });
    applyFit(estimate, kv ? kv[1] : 0, kv ? kv[2] : 0.0, gpuMemory, utilization, {
      kvBytes: kv ? (length: number) => gguf.kvCache(header!.metadata, length, batchSize, kv[0].dtype)![0] : null,
    });
    return estimate;
  }

  const names = [...groups.keys()];
  const projectors = new Map(names.map((name) => [name, gguf.mmproj(source.files, name)]));
  const unique = [...new Set([...projectors.values()].filter((p): p is string => Boolean(p)))].sort();
  const [first, ...rest] = await Promise.allSettled([
    gguf.readHeader(source, groups.get(names[0]!)![0]!),
    ...names.map((name) => gguf.readMetadata(source, groups.get(name)![0]!)),
    ...unique.map((p) => gguf.readMetadata(source, p)),
  ]);
  const results = rest.slice(0, names.length);
  const loaded = rest.slice(names.length);
  const projected = new Map<string, Config>();
  unique.forEach((p, i) => {
    const r = loaded[i]!;
    if (r.status === "fulfilled") projected.set(p, (r.value as [Config, boolean])[0]);
  });
  const files: Estimate[] = [];
  names.forEach((name, i) => {
    const result = results[i]!;
    if (result.status === "rejected") {
      if (result.reason instanceof HubError) throw result.reason;
      warnings.push(`Skipping \`${name}\`: ${message(result.reason)}`);
      return;
    }
    const [metadata, tokenizer] = result.value as [Config, boolean];
    const headerSize = tokenizer && first!.status === "fulfilled" ? (first.value as gguf.Header).dataStart : 0;
    const paths = groups.get(name)!;
    const projector = projectors.get(name) ?? null;
    const kv = ggufKv(metadata, maxModelLen, batchSize, kvDtype);
    const bytes =
      paths.reduce((total, p) => total + source.files.get(p)!, 0) -
      headerSize +
      (projector ? source.files.get(projector)! : 0);
    files.push(
      new Estimate({
        model: source.name,
        revision: source.revision,
        url: source.url,
        format: "gguf",
        file: name,
        weights: new Weights(0, bytes),
        kvCache: kv ? kv[0] : null,
        overhead: kv ? overhead_.llamaCpp(projector ? (projected.get(projector) ?? null) : null) : null,
      }),
    );
  });
  if (!files.length) throw new Error(`No valid GGUF files found for ${source.name}.`);
  if (gpuMemory !== null) {
    warnings.push("`--gpu-memory` is only checked for a single GGUF file, use `--file` to pick one.");
  }
  return new Listing(source.name, source.revision, source.url, files, warnings);
}

async function onnxLengths(source: Source, labels: string[]): Promise<Map<string, [number | null, number | null]>> {
  const configPath = (label: string): string | null => {
    const directory = dirname(label);
    return [`${directory}/config.json`, "config.json"].find((p) => source.files.has(p)) ?? null;
  };
  const paths = new Map(labels.map((label) => [label, configPath(label)]));
  const unique = [...new Set([...paths.values()].filter((p): p is string => p !== null))].sort();
  const settled = await Promise.allSettled(unique.map((p) => source.json(p)));
  const loaded = new Map(unique.map((p, i) => [p, settled[i]!]));
  const lengths = new Map<string, [number | null, number | null]>();
  for (const [label, path] of paths) {
    const result = path ? loaded.get(path)! : null;
    const found = result && result.status === "fulfilled" ? result.value : null;
    const text: Config = isRecord(found) ? (config_.textConfig(found) ?? {}) : {};
    const encoder = isRecord(found) ? found.max_source_positions : null;
    const length = text.max_position_embeddings;
    lengths.set(label, [isInt(length) ? length : null, isInt(encoder) ? encoder : null]);
  }
  return lengths;
}

async function estimateOnnx(
  source: Source,
  file: string | null,
  maxModelLen: number | null,
  batchSize: number,
  kvDtype: string,
  warnings: string[],
  gpuMemory: string | null,
  utilization: number,
): Promise<Estimate | Listing> {
  let groups = onnx.groupFiles(source.files);
  if (!groups.size) throw new Error(noWeights(source, "onnx"));
  if (kvDtype !== "auto") {
    throw new Error("`--kv-cache-dtype` is set by the ONNX graph inputs, so only `auto` is valid.");
  }
  if (file !== null) groups = new Map([onnx.selectFile(groups, file)]);

  const labels = [...groups.keys()];
  const lengthsTask = onnxLengths(source, labels);
  const results = await Promise.allSettled(labels.map((label) => onnx.readGroup(source, groups.get(label)!)));
  const lengths = await lengthsTask;
  const files: Estimate[] = [];
  labels.forEach((label, i) => {
    const result = results[i]!;
    if (result.status === "rejected") {
      if (result.reason instanceof HubError || labels.length === 1) throw result.reason;
      warnings.push(`Skipping \`${label}\`: ${message(result.reason)}`);
      return;
    }
    const graph = result.value;
    const [context, encoderLength] = lengths.get(label)!;
    const length = maxModelLen || context;
    const kv = length ? onnx.kvCache(graph, length, batchSize, encoderLength) : null;
    if (graph.cache.length && !length) {
      warnings.push(
        `KV cache estimation skipped for \`${label}\`, as no context length was found in \`config.json\`; ` +
          "set `--max-model-len`.",
      );
    }
    const estimate = new Estimate({
      model: source.name,
      revision: source.revision,
      url: source.url,
      format: "onnx",
      file: label,
      weights: graph.weights,
      kvCache: kv && length ? new KVCache(kv[0], kv[1], length, batchSize, {}) : null,
    });
    if (labels.length === 1) {
      applyFit(estimate, context || 0, kv ? kv[2] : 0.0, gpuMemory, utilization, {
        kvBytes: kv ? (n: number) => onnx.kvCache(graph, n, batchSize, encoderLength)![0] : null,
      });
    }
    files.push(estimate);
  });
  if (!files.length) throw new Error(`No valid ONNX files found for ${source.name}.`);
  if (files.some((f) => !f.weights.dtypes.size)) {
    warnings.push(
      `ONNX files over ${onnx.SMALL_FILE >> 20} MiB with the weights inline are estimated from their file sizes, without the ` +
        "parameter count and dtypes.",
    );
  }
  warnings.push("The ONNX Runtime overhead (activations and workspaces) is not estimated.");
  if (labels.length === 1) {
    files[0]!.warnings = warnings;
    return files[0]!;
  }
  if (gpuMemory !== null) {
    warnings.push("`--gpu-memory` is only checked for a single ONNX file, use `--file` to pick one.");
  }
  return new Listing(source.name, source.revision, source.url, files, warnings);
}

const ENCODER_DTYPE_ONLY =
  "`--dtype` only supports unquantized Safetensors text encoders; other models keep their stored dtypes.";

const SENTENCE_TRANSFORMERS_CONFIGS = [
  "sentence_bert_config.json",
  "sentence_roberta_config.json",
  "sentence_distilbert_config.json",
  "sentence_camembert_config.json",
  "sentence_albert_config.json",
  "sentence_xlm-roberta_config.json",
  "sentence_xlnet_config.json",
];

async function encoderLength(source: Source, config: Config, override: number | null): Promise<number | null> {
  if (override !== null) return override;
  for (const name of SENTENCE_TRANSFORMERS_CONFIGS) {
    if (source.files.has(name)) {
      try {
        const length = (await source.json(name)).max_seq_length;
        if (isInt(length) && length > 0) return length;
      } catch {
        // Unreadable Sentence Transformers configs fall back to the model configuration.
      }
    }
  }
  const positions = config.max_position_embeddings;
  if (!isInt(positions) || positions <= 0) return null;
  const roberta = ["roberta", "xlm-roberta", "camembert", "mpnet"].includes(config.model_type);
  const length = positions - (roberta ? (config.pad_token_id || 0) + 1 : 0);
  return length > 0 ? length : null;
}

function loadingTransient(components: Map<string, Weights>, dtype: string): number {
  const [target] = DTYPES[dtype]!;
  let largest = 0;
  for (const component of components.values()) {
    for (const [name, entry] of component.dtypes) {
      const width = FLOAT_BYTES[name];
      if (width !== undefined && name !== target) largest = Math.max(largest, entry.largest * width);
    }
  }
  return largest;
}

const FLOAT_SIZES: Record<string, number> = { float64: 8, float32: 4, float16: 2, bfloat16: 2 };

function floatSize(config: Config, weights: Weights): number {
  const dtype = config.dtype || config.torch_dtype;
  if (typeof dtype === "string") {
    const name = dtype.startsWith("torch.") ? dtype.slice(6) : dtype;
    if (name in FLOAT_SIZES) return FLOAT_SIZES[name]!;
  }
  let largest: string | null = null;
  for (const [d, w] of weights.dtypes) {
    if (["F64", "F32", "F16", "BF16"].includes(d) && w.params) {
      if (largest === null || w.params > weights.dtypes.get(largest)!.params) largest = d;
    }
  }
  return FLOAT_BYTES[largest ?? "BF16"]!;
}

function settle<T>(promise: Promise<T>): Promise<PromiseSettledResult<T>> {
  return promise.then(
    (value) => ({ status: "fulfilled", value }),
    (reason: unknown) => ({ status: "rejected", reason }),
  );
}

async function estimateSafetensors(
  source: Source,
  file: string | null,
  maxModelLen: number | null,
  batchSize: number,
  kvDtype: string,
  warnings: string[],
  token: string | null,
  gpuMemory: string | null,
  utilization: number,
  maxNumBatchedTokens: number | null,
  dtype: string | null,
  hub: HubOptions,
): Promise<Estimate | null> {
  const files = source.files;
  const configTask = files.has("config.json") ? settle(source.json("config.json")) : null;
  const processorTasks = ["preprocessor_config.json", "processor_config.json"]
    .filter((name) => files.has(name))
    .map((name) => settle(source.json(name)));
  const selected = await safetensors.discover(source, file, warnings);
  if (!selected.size) return null;
  let config: Config = {};
  if (configTask) {
    const result = await configTask;
    if (result.status === "fulfilled") {
      config = result.value;
    } else {
      warnings.push(`KV cache estimation skipped, as \`config.json\` could not be read (${message(result.reason)}).`);
    }
  }
  const quantization: Config = config.quantization_config || (config.text_config || {}).quantization_config || {};
  const [components, rawExperts, offloaded] = await safetensors.readComponents(
    source,
    selected,
    warnings,
    quantization,
  );
  let experts = rawExperts;
  const processor: Config = {};
  for (const result of await Promise.all(processorTasks)) {
    if (result.status === "fulfilled" && isRecord(result.value)) {
      Object.assign(processor, result.value, result.value.image_processor || {});
    }
  }

  let weights = new Weights();
  for (const component of components.values()) weights.merge(component);
  const estimate = new Estimate({
    model: source.name,
    revision: source.revision,
    url: source.url,
    format: "safetensors",
    weights,
    components,
    offloaded: offloaded.bytes ? offloaded : null,
    file,
    warnings,
  });
  if (offloaded.bytes) {
    warnings.push(
      `vLLM keeps the ${fixed(offloaded.bytes / overhead_.GIB, 2)} GiB of embedding tables (Engram or n-gram embeddings) ` +
        "in pinned host memory, so they are not in the GPU total but need that much free CPU memory.",
    );
  }

  const hasConfig = Object.keys(config).length > 0;
  if (!hasConfig && files.has("params.json")) {
    warnings.push(
      "KV cache estimation skipped, as Mistral `params.json` configs are not supported, estimate a version of " +
        "the model with a Transformers `config.json` for the KV cache.",
    );
  }
  const textEncoder = hasConfig && config_.isTextEncoder(config, files);
  if (dtype !== null) {
    const plain = ["F64", "F32", "F16", "BF16", "I64", "I32", "I16", "BOOL"];
    if (
      !textEncoder ||
      Object.keys(quantization).length ||
      [...weights.dtypes.keys()].some((d) => !plain.includes(d))
    ) {
      throw new Error(ENCODER_DTYPE_ONLY);
    }
  }
  let weightDtype = dtype;
  if (
    !textEncoder &&
    hasConfig &&
    config_.isGenerative(config, files) &&
    !Object.keys(quantization).length &&
    floatSize(config_.textConfig(config) ?? config, weights) === 4
  ) {
    weightDtype = "float16";
    warnings.push("FP32 decoder weights use a 16-bit CUDA serving reference; engines may select F16 or BF16.");
  }
  if (weightDtype !== null) {
    const cast = weightDtype;
    estimate.components = new Map([...components].map(([name, component]) => [name, component.cast(cast)]));
    estimate.weights = weights = new Weights();
    for (const component of estimate.components.values()) weights.merge(component);
    experts = new Map(
      [...experts].map(([name, group]) => [
        name,
        new Map([...group].map(([index, expert]) => [index, expert.cast(cast)])),
      ]),
    );
  }
  if (textEncoder) {
    const encoderConfig = config_.normalize(config);
    const length = await encoderLength(source, encoderConfig, maxModelLen);
    const tokens = overhead_.batchedTokens(gpuMemory, maxNumBatchedTokens);
    if (!encoderConfig.hidden_size || !encoderConfig.num_attention_heads) {
      warnings.push("Encoder warmup estimation skipped, as the model dimensions are missing from `config.json`.");
    } else if (
      [...experts.values()].some((group) => group.size) ||
      ["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts", "kv_lora_rank"].some(
        (k) => truthy(encoderConfig[k]),
      ) ||
      config_.layerKinds(encoderConfig).includes("state")
    ) {
      warnings.push("Encoder warmup estimation skipped for MoE, recurrent or latent-attention models.");
    } else if (length === null) {
      warnings.push("Encoder warmup estimation skipped, as the sequence length is unknown, set `--max-model-len`.");
    } else {
      const size = dtype ? DTYPES[dtype]![1] : floatSize(encoderConfig, weights);
      const overhead = overhead_.encoderWarmup(encoderConfig, files.keys(), tokens, length, size);
      estimate.overhead = overhead;
      if (dtype) {
        const transient = loadingTransient(components, dtype);
        overhead.loading = Math.max(0, transient - overhead.activations - overhead.buffers);
      }
      const sequence = Math.min(tokens, length);
      const full = Math.floor(tokens / sequence);
      const remaining = tokens % sequence;
      let attention: string;
      if (overhead_.encoderEager(encoderConfig, size)) {
        const tensors = overhead_.encoderAttentionTensors(encoderConfig, full + Number(Boolean(remaining)));
        attention =
          `eager attention, keeping ${tensors} tensors of attention scores per sequence` +
          (size >= 4 ? ", as memory-efficient attention kernels need half precision" : "");
      } else {
        attention = "memory-efficient attention";
      }
      warnings.push(
        `Encoder warmup assumes ${tokens} tokens in sequences of up to ${length} tokens with ${attention}. ` +
          (dtype ? `Floating-point parameters and activations use ${dtype}. ` : "Weights are estimated as stored. ") +
          "Engine kernels, padding and workspaces can change memory use.",
      );
      if (overhead.buffers) {
        const positions = encoderConfig.max_position_embeddings;
        warnings.push(
          `Eager ALiBi attention keeps the biases of ${positions} x ${positions} positions per head, ` +
            `${fixed(overhead.buffers / overhead_.GIB, 2)} GiB for the model's lifetime.`,
        );
      }
      if (overhead.loading) {
        warnings.push(
          `Casting to ${dtype} while loading reserves the largest tensor in its stored precision, ` +
            `peaking at ${fixed((overhead.bytes + weights.bytes) / overhead_.GIB, 2)} GiB while loading, ` +
            "assuming tensors are cast on the device.",
        );
      }
    }
    if (gpuMemory && (overhead_.parseGpuMemory(gpuMemory)[0] || 1) > 1) {
      warnings.push("Encoder fit assumes one replica per GPU, without tensor parallelism.");
    }
    applyFit(estimate, 0, 0.0, gpuMemory, utilization, { maxGpus: 1 });
    return estimate;
  }
  if (!hasConfig || files.has("config_sentence_transformers.json") || !config_.isGenerative(config, files)) {
    applyFit(estimate, 0, 0.0, gpuMemory, utilization);
    return estimate;
  }
  let text = config_.textConfig(config);
  const reference: string = (config.text_config || {})._name_or_path || "";
  if (text === null && reference.includes("/") && !reference.startsWith("/") && !reference.startsWith(".")) {
    try {
      const [referenced] = await openSource(reference, null, token, null, hub);
      try {
        text = config_.textConfig({ ...(await referenced.json("config.json")), ...config.text_config });
      } finally {
        await referenced.close();
      }
    } catch {
      // An unreachable referenced config leaves the KV cache unestimated, as warned below.
    }
  }
  if (text === null) {
    warnings.push("KV cache estimation skipped, as no text decoder config was found in `config.json`.");
    applyFit(estimate, 0, 0.0, gpuMemory, utilization);
    return estimate;
  }

  if (components.size === 1) {
    const name = [...components.keys()][0]!;
    estimate.moe = safetensors.moeBreakdown(
      estimate.components.get(name)!,
      experts.get(name) ?? new Map(),
      text,
      warnings,
    );
  }
  const length = maxModelLen || text.max_position_embeddings;
  if (!isInt(length)) {
    warnings.push("KV cache estimation skipped, as the context length is not in `config.json`, set `--max-model-len`.");
    applyFit(estimate, 0, 0.0, gpuMemory, utilization);
    return estimate;
  }
  const decoder = text;
  const kvType = config_.kvDtype(decoder, kvDtype, weights, warnings);
  const tokens = overhead_.batchedTokens(gpuMemory, maxNumBatchedTokens);
  if (maxNumBatchedTokens === null && gpuMemory) {
    const gib = overhead_.parseGpuMemory(gpuMemory)[1] / overhead_.GIB;
    if (gib >= 70 && gib < 90) {
      warnings.push(
        "vLLM batches 8192 tokens on GPUs with 70 GiB or more except on A100s, which use 2048, so set " +
          "`--max-num-batched-tokens 2048` for an A100.",
      );
    }
  }
  const [size, layers] = config_.kvCache(decoder, kvType, length, batchSize, tokens);
  const halfLength = Math.max(1, Math.floor(length / 2));
  const half = config_.kvCache(decoder, kvType, halfLength, batchSize, tokens)[0];
  const slope = length > 1 ? (size - half) / (length - halfLength) : 0.0;
  estimate.kvCache = new KVCache(size, kvType, length, batchSize, layers);
  if (overhead_.supportsVideo(config, files)) {
    warnings.push(
      "The encoder is estimated for the largest image, but vLLM profiles the largest video when it has more " +
        "tokens, which can reserve a few GiB more unless videos are disabled with " +
        "`--limit-mm-per-prompt '{\"video\": 0}'`.",
    );
  }
  estimate.overhead = overhead_.vllm(
    decoder,
    estimate.moe !== null,
    (layers.state ?? 0) > 0,
    tokens,
    overhead_.encoder(config, processor),
  );
  applyFit(estimate, decoder.max_position_embeddings || length, slope, gpuMemory, utilization, {
    kvHeads: config_.kvHeads(decoder),
    padding: mxfp4Padding(config, decoder, weights),
    kvBytes: (n: number, gpus: number) => config_.kvCache(decoder, kvType, n, batchSize, tokens, gpus)[0],
  });
  return estimate;
}

const PICKLES = [".bin", ".pt", ".pth", ".ckpt", ".nemo"];

function noWeights(source: Source, format: string | null): string {
  const files = [...source.files.keys()];
  const found = (
    [
      ["Safetensors", files.some((p) => p.endsWith(".safetensors"))],
      ["GGUF", files.some((p) => p.endsWith(".gguf"))],
      ["ONNX", files.some((p) => p.endsWith(".onnx"))],
    ] as [string, boolean][]
  )
    .filter(([, present]) => present)
    .map(([name]) => name);
  if (format && found.length) {
    const name = { safetensors: "Safetensors", gguf: "GGUF", onnx: "ONNX" }[format as Format];
    return (
      `No ${name} weights found for \`${source.name}\`, which has ${found.join(" and ")} weights, use ` +
      `\`--format ${found[0]!.toLowerCase()}\` or drop \`--format\`.`
    );
  }
  const pickles = files
    .filter((p) => PICKLES.some((suffix) => p.endsWith(suffix)))
    .sort((a, b) => source.files.get(b)! - source.files.get(a)!);
  if (pickles.length) {
    return (
      `\`${source.name}\` only has PyTorch pickle weights (e.g. \`${pickles[0]}\`), which hf-mem cannot read without ` +
      "loading them; estimate a version of the model in Safetensors, GGUF or ONNX instead."
    );
  }
  return `No Safetensors, GGUF or ONNX weights found for \`${source.name}\`.`;
}

export async function run(model: string, options: RunOptions = {}): Promise<Estimate | Listing> {
  const revision = options.revision ?? null;
  const maxModelLen = options.maxModelLen ?? null;
  const batchSize = options.batchSize ?? 1;
  const kvCacheDtype = options.kvCacheDtype ?? "auto";
  const hfToken = options.hfToken ?? null;
  const gpuMemory = options.gpuMemory ?? null;
  const utilization = options.gpuMemoryUtilization ?? 0.92;
  const maxNumBatchedTokens = options.maxNumBatchedTokens ?? null;
  const format = options.format ?? null;
  const dtype = options.dtype ?? null;
  const hub: HubOptions = { fetch: options.fetch, signal: options.signal };

  if (batchSize < 1 || [maxModelLen, maxNumBatchedTokens].some((v) => v !== null && v < 1)) {
    throw new Error("`--batch-size`, `--max-model-len` and `--max-num-batched-tokens` must be positive.");
  }
  if (!(utilization > 0 && utilization <= 1)) throw new Error("`--gpu-memory-utilization` must be in (0, 1].");
  if (gpuMemory !== null) overhead_.parseGpuMemory(gpuMemory);
  if (format !== null && !(FORMATS as string[]).includes(format)) {
    throw new Error(`\`--format=${format}\` is not valid, use one of ${FORMATS.join(", ")}.`);
  }
  if (dtype !== null && !(dtype in DTYPES)) {
    throw new Error(`\`--dtype=${dtype}\` is not valid, use one of ${Object.keys(DTYPES).join(", ")}.`);
  }
  const [source, urlFile] = await openSource(model, revision, hfToken, options.progress ?? null, hub);
  const file = options.file || urlFile;
  const warnings: string[] = [];
  try {
    const paths = [...source.files.keys()];
    const ggufFiles = gguf.ggufPaths(source.files);
    const onnxFiles = paths.filter((p) => p.endsWith(".onnx"));
    const target = file !== null ? file.replace(/^\/+|\/+$/g, "") : null;
    const wantsGguf =
      target !== null &&
      (target.endsWith(".gguf") ||
        ggufFiles.some((p) => p.startsWith(target + "/")) ||
        gguf.tagged(gguf.groupFiles(ggufFiles), target).length > 0);
    const wantsOnnx =
      target !== null &&
      (target.endsWith(".onnx") ||
        onnxFiles.some((p) => p.startsWith(target + "/")) ||
        onnx.groupFiles(source.files).has(target));
    const hasSafetensors =
      source.files.has("model_index.json") ||
      paths.some((p) =>
        ["model.safetensors", "model.safetensors.index.json"].includes(p.replace(safetensors.SHARD, "")),
      );
    const onlyOnnx = onnxFiles.length > 0 && !paths.some((p) => p.endsWith(".safetensors")) && !ggufFiles.length;
    if (format === "onnx" || (format === null && ((wantsOnnx && !wantsGguf) || onlyOnnx))) {
      if (dtype !== null) throw new Error(ENCODER_DTYPE_ONLY);
      return await estimateOnnx(source, file, maxModelLen, batchSize, kvCacheDtype, warnings, gpuMemory, utilization);
    }
    if (format === "gguf" || (format === null && ggufFiles.length > 0 && (wantsGguf || !hasSafetensors))) {
      if (dtype !== null) throw new Error(ENCODER_DTYPE_ONLY);
      return await estimateGguf(source, file, maxModelLen, batchSize, kvCacheDtype, warnings, gpuMemory, utilization);
    }
    const estimate = await estimateSafetensors(
      source,
      file,
      maxModelLen,
      batchSize,
      kvCacheDtype,
      warnings,
      hfToken,
      gpuMemory,
      utilization,
      maxNumBatchedTokens,
      dtype,
      hub,
    );
    if (estimate) {
      for (const [other, found] of [
        ["GGUF", ggufFiles],
        ["ONNX", onnxFiles],
      ] as [string, string[]][]) {
        if (found.length && format === null && file === null) {
          warnings.push(
            `${other} files were also found and are not estimated; use \`--format ${other.toLowerCase()}\` ` +
              "to estimate them.",
          );
        }
      }
      return estimate;
    }
    if (onnxFiles.length && format === null) {
      if (dtype !== null) throw new Error(ENCODER_DTYPE_ONLY);
      return await estimateOnnx(source, file, maxModelLen, batchSize, kvCacheDtype, warnings, gpuMemory, utilization);
    }
    throw new Error(noWeights(source, format));
  } finally {
    await source.close();
  }
}

export const arun = run;
