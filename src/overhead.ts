import { type Config, layerKinds } from "./config.js";
import { int, isRecord, str, truthy } from "./py.js";
import { Fit, Overhead } from "./result.js";

export const GIB = 2 ** 30;
export const MIB = 2 ** 20;
export const LARGE_GPU = 70 * GIB;
export const LARGEST_GPU = 160 * GIB;
// llama.cpp grows the mmproj compute buffer up to the largest image: 4096 merged tokens for dynamic resolution
// encoders (unless `image_max_pixels` is lower, or llama.cpp caps the projector), `image_size` otherwise, and 30 s
// of audio.
export const CLIP_MAX_TOKENS: Record<string, number> = { pixtral: 1024, lfm2: 256 };
export const CLIP_DYNAMIC_TOKENS = 4096;
export const CLIP_AUDIO_POSITIONS = 1500;

interface Profile {
  vocab: number;
  inter: number;
  hidden: number;
  act: number;
  graph: number;
  graph_layer: number;
  graph_moe: number;
  graph_hybrid: number;
  graph_lh: number;
  graph_width: number;
  non_torch: number;
  context: number;
}

// Fitted on vLLM 0.33 memory profiles (A100 for "small", H200 for "large"), see the README.
const PROFILES: Record<"small" | "large", Profile> = {
  small: {
    vocab: 5004,
    inter: 6.6,
    hidden: 77.2,
    act: 0,
    graph: 0.16 * GIB,
    graph_layer: 5.4 * MIB,
    graph_moe: 5.6 * MIB,
    graph_hybrid: 0,
    graph_lh: 0,
    graph_width: 6174,
    non_torch: 0.27 * GIB,
    context: 0.83 * GIB,
  },
  large: {
    vocab: 12702,
    inter: 0,
    hidden: 13.2,
    act: 0.93 * GIB,
    graph: 0.29 * GIB,
    graph_layer: 0,
    graph_moe: 12.9 * MIB,
    graph_hybrid: 7.6 * MIB,
    graph_lh: 248.8,
    graph_width: 23908,
    non_torch: 0.87 * GIB,
    context: 1.01 * GIB,
  },
};
const GATED = ["silu", "swiglu", "swish"];
const GLU = ["modernbert", "new", "gte", "gemma3_text"];
const ENCODER = {
  encoder: 7,
  decoder: 10,
  mlp: 2,
  gated_mlp: 5,
  runtime: 0.32 * GIB,
};

export function batchedTokens(gpuMemory: string | null, override: number | null = null): number {
  if (override !== null) return override;
  const size = gpuMemory !== null ? parseGpuMemory(gpuMemory)[1] : LARGE_GPU;
  return size >= LARGEST_GPU ? 16384 : size >= LARGE_GPU ? 8192 : 2048;
}

function towerBytes(tower: Config, positions: number): number {
  const hidden = tower.hidden_size || tower.d_model || tower.embed_dim || 0;
  const inter = tower.intermediate_size || tower.encoder_ffn_dim || 4 * hidden;
  const activation = str(tower.hidden_act || tower.activation_function || "");
  const gated = GATED.some((a) => activation.includes(a));
  return positions * 2 * (4 * hidden + (gated ? 3 : 2) * inter);
}

function imagePatches(config: Config, vision: Config, processor: Config): number {
  const patch = vision.patch_size || processor.patch_size || 14;
  const size: Config = isRecord(processor.size) ? processor.size : {};
  const pixels = processor.max_pixels || ((size.longest_edge || 0) > 65536 ? size.longest_edge : null);
  if (truthy(pixels)) return Math.floor(pixels / patch ** 2);
  const soft = processor.max_soft_tokens || config.vision_soft_tokens_per_image;
  if (truthy(soft)) return soft * (processor.pooling_kernel_size || vision.pooling_kernel_size || 1) ** 2;
  const side = vision.image_size || config.force_image_size || size.height || 224;
  let tiles = 1;
  const dynamic = config.max_dynamic_patch;
  const ratio = str(config.vision_aspect_ratio || "");
  const tile = (processor.max_image_size || {}).longest_edge;
  const hd = processor.dynamic_hd;
  if (truthy(dynamic)) {
    tiles = dynamic + Number(truthy(config.use_thumbnail));
  } else if (ratio.startsWith("anyres_max_")) {
    tiles = Number.parseInt(ratio.slice(ratio.lastIndexOf("_") + 1), 10) + 1;
  } else if (truthy(tile) && truthy(size.longest_edge)) {
    tiles = Math.floor(size.longest_edge / tile) ** 2 + 1;
  } else if (truthy(hd)) {
    tiles = hd + 1;
  }
  return tiles * Math.floor(side / patch) ** 2;
}

function root(config: Config): Config {
  const thinker = config.thinker_config;
  return isRecord(thinker) ? thinker : config;
}

export function encoder(config: Config, processor: Config): number {
  const base = root(config);
  const vision = base.vision_config;
  const audio = base.audio_config;
  let largest = 0;
  if (isRecord(vision)) largest = towerBytes(vision, imagePatches(base, vision, processor));
  if (isRecord(audio)) largest = Math.max(largest, towerBytes(audio, audio.max_source_positions || 1500));
  return largest;
}

export function supportsVideo(config: Config, files: Map<string, number>): boolean {
  const raw = root(config).vision_config;
  const vision: Config = isRecord(raw) ? raw : {};
  return (
    truthy(vision) && (files.has("video_preprocessor_config.json") || truthy(vision.temporal_patch_size))
  );
}

export function vllm(config: Config, moe: boolean, hybrid: boolean, tokens: number, encoderBytes = 0): Overhead {
  const p = PROFILES[tokens >= 8192 ? "large" : "small"];
  const layers = config.num_hidden_layers || 0;
  const hidden = config.hidden_size || 0;
  const vocab = config.vocab_size || 32000;
  let intermediate = config.intermediate_size || config.ffn_hidden_size || 4 * hidden;
  if (moe && truthy(config.moe_intermediate_size)) {
    intermediate = config.moe_intermediate_size * (config.num_experts_per_tok || 1);
  }
  const activations = p.vocab * vocab + tokens * (p.inter * intermediate + p.hidden * hidden) + p.act;
  const graphs =
    p.graph +
    layers * (p.graph_layer + Number(moe) * p.graph_moe + Number(hybrid) * p.graph_hybrid + p.graph_lh * hidden) +
    p.graph_width * (hidden + intermediate);
  const context = int(p.context);
  return new Overhead(int(activations), encoderBytes, int(graphs), int(p.non_torch) + context, context);
}

export function encoderEager(config: Config, size: number): boolean {
  if (size >= 4) return true;
  const implementation = config._attn_implementation || config.attn_implementation;
  if (implementation !== null && implementation !== undefined) return implementation === "eager";
  return ["mpnet", "deberta", "deberta-v2"].includes(config.model_type);
}

export function encoderAttentionTensors(config: Config, sequences: number): number {
  const terms = config.pos_att_type || [];
  const biases =
    Number(truthy(config.relative_attention_num_buckets || config.model_type === "mpnet")) +
    Number(config.position_embedding_type === "alibi") +
    (truthy(config.relative_attention) ? (typeof terms === "string" ? terms.split("|") : terms).length : 0);
  const kinds = new Set(truthy(config.num_hidden_layers) ? layerKinds(config) : []);
  let masks: number;
  if ((truthy(config.local_attention) && truthy(config.global_attn_every_n_layers)) || (kinds.has("full") && kinds.has("sliding"))) {
    masks = 2;
  } else if (truthy(config.rms_norm_eps) && !truthy(config.use_bidirectional_attention)) {
    masks = 1;
  } else {
    masks = Number(sequences > 1 && !biases);
  }
  return 3 + biases + masks;
}

export function encoderBuffers(config: Config, size: number): number {
  if (config.position_embedding_type !== "alibi") return 0;
  return (config.num_attention_heads || 1) * (config.max_position_embeddings || 0) ** 2 * size;
}

export function encoderWarmup(
  config: Config,
  files: Iterable<string>,
  tokens: number,
  length: number,
  size: number,
): Overhead {
  const p = ENCODER;
  const names = [...files];
  const hidden = config.hidden_size || 0;
  const heads = config.num_attention_heads || 1;
  const inter = config.intermediate_size || config.n_inner || 4 * hidden;
  const act = str(config.hidden_act || config.activation_function || "");
  const gated =
    GATED.some((a) => act.includes(a)) ||
    GLU.includes(config.model_type) ||
    str(config.feed_forward_type || "").includes("glu");
  const masked = ((config.architectures || []) as unknown[]).some((a) => str(a).endsWith("ForMaskedLM"));
  const sparse = names.some((f) => f.endsWith("SpladePooling/config.json"));
  const pooled = names.includes("config_sentence_transformers.json") && !sparse;
  let perToken = (truthy(config.rms_norm_eps) ? p.decoder : p.encoder) * hidden;
  perToken += (gated ? p.gated_mlp : p.mlp) * inter;
  const headDim = config.head_dim || Math.floor(hidden / heads);
  const kvHeads = config.num_key_value_heads || heads;
  perToken += Math.max(0, (heads + 2 * kvHeads) * headDim - 3 * hidden);
  perToken += masked && !pooled ? (sparse ? 3 : 1) * (config.vocab_size || 0) : 0;
  const sequence = Math.min(tokens, length);
  const full = Math.floor(tokens / sequence);
  const remaining = tokens % sequence;
  const eager = encoderEager(config, size);
  const scores = eager ? heads * (full * sequence ** 2 + remaining ** 2) : 0;
  const tensors = encoderAttentionTensors(config, full + Number(Boolean(remaining)));
  const runtime = int(p.runtime);
  const buffers = eager ? encoderBuffers(config, size) : 0;
  return new Overhead(int(tokens * perToken + tensors * scores) * size, 0, 0, runtime, runtime, buffers);
}

function clipBytes(positions: number, hidden: number, ffn: number | null | undefined, gated: boolean): number {
  return int((gated ? 18 : 7.2) * positions * Math.max(hidden, ffn || 4 * hidden));
}

export function mmprojEncoder(metadata: Config): number {
  let total = 0;
  const hidden = metadata["clip.vision.embedding_length"];
  if (truthy(hidden)) {
    const patch = metadata["clip.vision.patch_size"] || 14;
    const projector = metadata["clip.projector_type"] || metadata["clip.vision.projector_type"];
    const window = metadata["clip.vision.n_wa_pattern"];
    const merge = metadata["clip.vision.spatial_merge_size"] || metadata["clip.vision.projector.scale_factor"];
    let patches: number;
    if (typeof projector === "string" && projector in CLIP_MAX_TOKENS) {
      patches = CLIP_MAX_TOKENS[projector]! * (merge || 1) ** 2;
    } else if (
      truthy(window) ||
      "clip.vision.spatial_merge_size" in metadata ||
      "clip.vision.image_max_pixels" in metadata
    ) {
      patches = CLIP_DYNAMIC_TOKENS * (merge || 2) ** 2;
      const maxPixels = metadata["clip.vision.image_max_pixels"];
      if (truthy(maxPixels)) patches = Math.min(patches, Math.floor(maxPixels / patch ** 2));
    } else {
      patches = Math.floor((metadata["clip.vision.image_size"] || 224) / patch) ** 2;
    }
    const ffn = metadata["clip.vision.feed_forward_length"];
    const gated = truthy(metadata["clip.use_silu"]);
    total += clipBytes(patches, hidden, ffn, gated) + (truthy(window) ? 2 * patches ** 2 : 0);
  }
  const audio = metadata["clip.audio.embedding_length"];
  if (truthy(audio)) {
    total += clipBytes(CLIP_AUDIO_POSITIONS, audio, metadata["clip.audio.feed_forward_length"], false);
  }
  return total;
}

export function llamaCpp(projector: Config | null = null): Overhead {
  const encoderBytes = projector ? mmprojEncoder(projector) : 0;
  return new Overhead(int(0.15 * GIB), encoderBytes, 0, int(0.47 * GIB), int(0.47 * GIB));
}

export function mxfp4Padding(experts: number, intermediate: number, hidden: number): (n: number) => number {
  // vLLM rounds the MXFP4 expert shards up to its kernel tiles (128 intermediate, 256 hidden on Hopper)
  return (n: number) => {
    const shard = Math.max(1, Math.floor(intermediate / n));
    const factor = ((Math.ceil(shard / 128) * 128) / shard) * ((Math.ceil(hidden / 256) * 256) / hidden);
    return int(experts * (factor - 1));
  };
}

function removeSuffix(text: string, suffix: string): string {
  return text.endsWith(suffix) ? text.slice(0, -suffix.length) : text;
}

/** Python's `float(...)` on a string: decimal literals, `inf`, `nan`, `1e3`, surrounding whitespace. */
function pyFloat(text: string): number {
  const trimmed = text.trim().replaceAll("_", "");
  if (/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(trimmed)) return Number(trimmed);
  if (/^[+-]?(inf|infinity)$/i.test(trimmed)) return trimmed.startsWith("-") ? -Infinity : Infinity;
  if (/^[+-]?nan$/i.test(trimmed)) return Number.NaN;
  throw new Error("invalid float");
}

export function parseGpuMemory(value: string): [number | null, number] {
  const text = removeSuffix(removeSuffix(value.toLowerCase(), "gib"), "gb");
  const sep = text.lastIndexOf("x");
  const countText = sep < 0 ? "" : text.slice(0, sep);
  const sizeText = sep < 0 ? text : text.slice(sep + 1);
  let gpus: number | null;
  let memory: number;
  try {
    if (sep >= 0) {
      if (!/^\s*[+-]?\d+(_\d+)*\s*$/.test(countText)) throw new Error("invalid int");
      gpus = Number.parseInt(countText.replaceAll("_", ""), 10);
    } else {
      gpus = null;
    }
    const size = pyFloat(sizeText) * GIB;
    if (!Number.isFinite(size)) throw new Error("overflow");
    memory = Math.trunc(size);
  } catch {
    gpus = 0;
    memory = 0;
  }
  if ((gpus !== null && gpus < 1) || !(memory > 0)) {
    throw new Error(`\`--gpu-memory=${value}\` must look like \`80\` or \`8x80\` (GiB per GPU).`);
  }
  return [gpus, memory];
}

export interface FitOptions {
  weights: number;
  kvFixed: number;
  kvPerToken: number;
  maxModelLen: number;
  contextLength: number;
  overhead: number;
  gpuMemory: string;
  utilization: number;
  kvHeads?: number | null;
  context?: number;
  padding?: ((n: number) => number) | null;
  kvBytes?: ((length: number, n: number) => number) | null;
  maxGpus?: number | null;
}

export function fit(options: FitOptions): Fit {
  const { kvFixed, kvPerToken, maxModelLen, contextLength, overhead, gpuMemory, utilization } = options;
  const kvHeads = options.kvHeads ?? null;
  const context = options.context ?? 0;
  const padding = options.padding ?? null;
  const kvBytes = options.kvBytes ?? null;
  const maxGpus = options.maxGpus ?? null;
  let [gpus, size] = parseGpuMemory(gpuMemory);
  if (maxGpus !== null && gpus !== null) gpus = Math.min(gpus, maxGpus);
  const usable = Math.min(size * utilization, size - context) - (overhead - context);
  const base = options.weights;

  const cache = (length: number, n: number): number => {
    const replication = kvHeads ? Math.max(1.0, n / kvHeads) : 1.0;
    if (kvBytes !== null) return kvBytes(length, n);
    return kvFixed + kvPerToken * length * replication;
  };
  const padded = (n: number): number => base + (padding ? padding(n) : 0);

  const candidates = [...new Set([1, 2, 4, 8, 16, 32, 64, gpus || 1])].sort((a, b) => a - b);
  const needed =
    candidates.find((n) => n <= (maxGpus || n) && padded(n) + cache(maxModelLen, n) <= n * usable) ?? null;
  const n = gpus || 1;
  const weights = padded(n);
  let longest: number;
  if (kvBytes !== null) {
    // Search the actual allocation: cache pages, sliding windows and recurrent state make it non-linear.
    let low = 0;
    let high = contextLength;
    while (low < high) {
      const middle = Math.floor((low + high + 1) / 2);
      if (weights + cache(middle, n) <= n * usable) low = middle;
      else high = middle - 1;
    }
    longest = low;
  } else {
    const budget = n * usable - weights - kvFixed;
    const perToken = kvPerToken * (kvHeads ? Math.max(1.0, n / kvHeads) : 1.0);
    longest = perToken > 0 && budget > 0 ? Math.floor(budget / perToken) : 0;
  }
  const required = weights + cache(maxModelLen, n) + n * overhead;
  return new Fit(
    size,
    int(Math.min(size * utilization, size - context) + context),
    int(required),
    gpus,
    needed,
    Math.min(longest, contextLength),
  );
}
