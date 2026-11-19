import type { Config } from "./config.js";
import { basename, dirname, isInt, product } from "./py.js";
import { MoE } from "./result.js";
import { HubError, Semaphore, type Source } from "./source.js";
import { choices, Weights } from "./weights.js";

export const PROBE_SIZE = 100_000;
export const MAX_HEADER_SIZE = 100_000_000;
export const SHARD = /-\d+-of-\d+(?=\.safetensors$)/;
const MODEL_SHARD = /^model-\d+-of-(\d+)\.safetensors$/;
// Embedding tables vLLM keeps in pinned host memory (`EngramConfig.cpu_offload`, on by default): the DeepSeek-V4.1
// Engram tables and the Qwen3.8 / Qwen4Exp per-layer n-gram embeddings
export const OFFLOADED =
  /(?:\.engram\.embed\.(?:weight|scale)|engram_tables\.\d+\.(?:weight|weight_scale_inv)|\.ngram_embedding\.shard_\d+\.weight)$/;
const EXPERT = /(?:^|\.)(?:local_|routed_)?experts?\.(\d+)(?:\.|$)/;
// Sub-models loaded next to the main model, e.g. the audio tokenizer that decodes the generated speech
export const AUXILIARY = ["audio_tokenizer", "speech_tokenizer", "audio_decoder", "vocoder", "codec"];

const decoder = new TextDecoder();

function invalid(path: string): Error {
  return new Error(`\`${path}\` is not a valid Safetensors file.`);
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

export async function readHeader(source: Source, path: string, probe = PROBE_SIZE): Promise<Uint8Array> {
  let head = await source.read(path, 0, probe);
  if (head.length < 8) throw invalid(path);
  const size = Number(new DataView(head.buffer, head.byteOffset, 8).getBigUint64(0, true));
  if (size > MAX_HEADER_SIZE) throw invalid(path);
  if (size + 8 > head.length) head = concat(head, await source.read(path, head.length, size + 8));
  return head.subarray(8, size + 8);
}

/** Per dtype `[params, bytes]`, per `expert + "\0" + dtype`, offloaded per dtype, and the largest tensor per dtype. */
export type Summary = [
  Map<string, [number, number]>,
  Map<string, [number, number]>,
  Map<string, [number, number]>,
  Map<string, number>,
];

function entry(map: Map<string, [number, number]>, key: string): [number, number] {
  let value = map.get(key);
  if (value === undefined) {
    value = [0, 0];
    map.set(key, value);
  }
  return value;
}

export function summarize(raw: Uint8Array, quantization: Config | null = null): Summary {
  const dtypes = new Map<string, [number, number]>();
  const experts = new Map<string, [number, number]>();
  const offloaded = new Map<string, [number, number]>();
  const largest = new Map<string, number>();
  const q: Config = quantization || {};
  let packed: string | null = null;
  let scales: string | null = null;
  if (q.format === "mxfp4-pack-quantized") {
    packed = "weight_packed";
    scales = "weight_scale";
  } else if (q.quant_method === "mxfp4") {
    packed = "_blocks";
    scales = "_scales";
  }
  const header: Record<string, any> = JSON.parse(decoder.decode(raw));
  for (const [tensor, info] of Object.entries(header)) {
    if (tensor === "__metadata__") continue;
    const [start, end] = info.data_offsets as [number, number];
    let params = product(info.shape as number[]);
    let dtype: string = info.dtype;
    const expert = tensor.includes("expert") ? EXPERT.exec(tensor) : null;
    if (packed && scales && dtype === "U8") {
      if (tensor.endsWith(packed)) {
        params *= 2;
        dtype = "MXFP4";
      } else if (tensor.endsWith(scales)) {
        params = 0;
        dtype = "MXFP4";
      }
    } else if (expert && q.expert_dtype === "fp4") {
      if (dtype === "I8" && tensor.endsWith(".weight")) {
        params *= 2;
        dtype = "MXFP4";
      } else if (dtype === "F8_E8M0" && tensor.endsWith(".scale")) {
        params = 0;
        dtype = "MXFP4";
      }
    }
    if (
      q.quant_method === "fp8" &&
      ((dtype === "F8_E8M0" && tensor.endsWith(".scale")) || (dtype === "F32" && tensor.endsWith(".weight_scale_inv")))
    ) {
      params = 0;
    }
    const isOffloaded = OFFLOADED.test(tensor);
    const total = entry(isOffloaded ? offloaded : dtypes, dtype);
    total[0] += params;
    total[1] += end - start;
    if (!isOffloaded) largest.set(dtype, Math.max(largest.get(dtype) ?? 0, params));
    if (expert) {
      const byExpert = entry(experts, `${Number.parseInt(expert[1]!, 10)}\0${dtype}`);
      byExpert[0] += params;
      byExpert[1] += end - start;
    }
  }
  return [dtypes, experts, offloaded, largest];
}

function removeSuffix(text: string, suffix: string): string {
  return text.endsWith(suffix) ? text.slice(0, -suffix.length) : text;
}

function groups(paths: string[]): Map<string, string[]> {
  const result = new Map<string, string[]>();
  for (const path of [...paths].sort()) {
    const name = removeSuffix(path.replace(SHARD, ""), ".safetensors");
    const group = result.get(name);
    if (group) group.push(path);
    else result.set(name, [path]);
  }
  return result;
}

function select(all: Map<string, string[]>, file: string): Map<string, string[]> {
  let target = removeSuffix(file.replace(/^\/+|\/+$/g, ""), ".safetensors");
  target = removeSuffix((target + ".safetensors").replace(SHARD, ""), ".safetensors");
  const exact = new Map([...all].filter(([name]) => name === target || name.startsWith(target + "/")));
  if (exact.size) return exact;
  return new Map([...all].filter(([name]) => basename(name) === target));
}

function complete(paths: string[]): boolean {
  const totals = new Set(paths.map((p) => MODEL_SHARD.exec(p)?.[1] ?? ""));
  return totals.size === 1 && !totals.has("") && Number.parseInt([...totals][0]!, 10) === paths.length;
}

export async function discover(source: Source, file: string | null, warnings: string[]): Promise<Map<string, string[]>> {
  const files = source.files;
  const all = groups([...files.keys()].filter((p) => p.endsWith(".safetensors")));
  if (!all.size) return new Map();

  if (file !== null) {
    const selected = select(all, file);
    if (!selected.size) {
      throw new Error(`No Safetensors file matching \`${file}\` found, use one of ${choices([...all.keys()])}.`);
    }
    return selected;
  }

  if (files.has("model_index.json")) {
    const index: Record<string, unknown> = await source.json("model_index.json");
    const components = new Map<string, string[]>();
    for (const [name, value] of Object.entries(index)) {
      if (name.startsWith("_") || !Array.isArray(value)) continue;
      const candidates = [...all.keys()].filter((g) => dirname(g) === name);
      const preferred = [`${name}/diffusion_pytorch_model`, `${name}/model`].filter((g) => all.has(g));
      const chosen = preferred.length ? preferred : candidates;
      if (chosen.length) components.set(name, all.get(chosen[0]!)!);
    }
    if (components.size) return components;
  }

  if (all.has("model") || files.has("model.safetensors.index.json")) {
    let paths = all.get("model") ?? [];
    if (files.has("model.safetensors.index.json") && !complete(paths)) {
      const index: Config = await source.json("model.safetensors.index.json");
      const mapped = [...new Set(Object.values((index.weight_map || {}) as Record<string, string>))].sort();
      const indexed = mapped.length ? mapped : paths;
      paths = indexed.filter((p) => files.has(p));
      if (paths.length < indexed.length) {
        warnings.push(
          `\`model.safetensors.index.json\` references ${indexed.length - paths.length} files that are not in ` +
            "the repository, so the estimate only covers the uploaded `*.safetensors` files.",
        );
        if (!paths.length) paths = [...all.values()].flat().sort();
      }
    }
    const inDirectory = (directory: string) =>
      [...all].filter(([g]) => dirname(g) === directory).flatMap(([, ps]) => ps);
    if (!files.has("config_sentence_transformers.json")) {
      const components = new Map<string, string[]>([["Transformer", paths]]);
      for (const name of AUXILIARY) {
        const auxiliary = inDirectory(name);
        if (auxiliary.length) components.set(name, auxiliary);
      }
      return components;
    }
    const components = new Map<string, string[]>([["0_Transformer", paths]]);
    if (files.has("modules.json")) {
      for (const module of (await source.json("modules.json")) as Config[]) {
        const path = String(module.path || "").replace(/^\/+|\/+$/g, "");
        if (path) {
          const modulePaths = inDirectory(path);
          if (modulePaths.length) components.set(path, modulePaths);
        }
      }
    }
    return components;
  }

  if (all.size > 1) {
    warnings.push(
      "No `model.safetensors`, `model.safetensors.index.json` or `model_index.json` found, so every " +
        "`*.safetensors` file is counted as a separate component, assuming all of them are loaded; use " +
        "`--file` to pick one.",
    );
  }
  return all;
}

/** Python's `RuntimeError`: a plain `Error`, not a `HubError` or a JS engine error. */
function isRuntimeError(error: unknown): boolean {
  return error instanceof Error && !(error instanceof HubError) && error.constructor === Error;
}

export async function readComponents(
  source: Source,
  input: Map<string, string[]>,
  warnings: string[] | null = null,
  quantization: Config | null = null,
): Promise<[Map<string, Weights>, Map<string, Map<number, Weights>>, Weights]> {
  let components = input;
  const paths = [...new Set([...components.values()].flat())].sort();
  const semaphore = new Semaphore(32);
  let probe = PROBE_SIZE;
  const summaries = new Map<string, Summary>();

  const read = async (path: string) => {
    const raw = await semaphore.run(() => readHeader(source, path, probe));
    try {
      summaries.set(path, summarize(raw, quantization));
    } catch {
      throw invalid(path);
    }
    probe = Math.max(probe, raw.length + 8);
  };

  const results = await Promise.allSettled(paths.map(read));
  const failed = new Map<string, unknown>();
  results.forEach((r, i) => {
    if (r.status === "rejected") failed.set(paths[i]!, r.reason);
  });
  if (failed.size) {
    const error = failed.values().next().value;
    const broken = [...components].filter(([, ps]) => ps.some((p) => failed.has(p))).map(([name]) => name);
    if (!isRuntimeError(error) || broken.length === components.size) throw error;
    components = new Map([...components].filter(([name]) => !broken.includes(name)));
    warnings?.push(`Skipping ${broken.length} invalid \`*.safetensors\` files, e.g. ${(error as Error).message}`);
  }
  const weights = new Map<string, Weights>();
  const experts = new Map<string, Map<number, Weights>>();
  const offloaded = new Weights();
  for (const [name, componentPaths] of components) {
    const component = new Weights();
    const merged = new Map<number, Weights>();
    for (const path of componentPaths) {
      const [dtypes, byExpert, cpu, largest] = summaries.get(path)!;
      for (const [dtype, [params, nbytes]] of dtypes) component.add(dtype, params, nbytes, largest.get(dtype) ?? 0);
      for (const [key, [params, nbytes]] of byExpert) {
        const split = key.indexOf("\0");
        const expert = Number(key.slice(0, split));
        let target = merged.get(expert);
        if (target === undefined) {
          target = new Weights();
          merged.set(expert, target);
        }
        target.add(key.slice(split + 1), params, nbytes);
      }
      for (const [dtype, [params, nbytes]] of cpu) offloaded.add(dtype, params, nbytes);
    }
    weights.set(name, component);
    experts.set(name, merged);
  }
  return [weights, experts, offloaded];
}

function pickInt(config: Config, keys: string[], minimum = 2): number | null {
  for (const key of keys) {
    const value = config[key];
    if (isInt(value) && value >= minimum) return value;
  }
  return null;
}

export function moeBreakdown(
  weights: Weights,
  experts: Map<number, Weights>,
  config: Config,
  warnings: string[],
): MoE | null {
  if (!experts.size) return null;
  const ids = [...experts.keys()].sort((a, b) => a - b);
  const count = pickInt(config, ["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"]);
  if (count !== null && (ids.length !== count || ids.some((id, i) => id !== i))) {
    warnings.push(
      `MoE breakdown skipped, as the expert indices in the weights (${ids.length}) do not match the ` +
        `\`config.json\` expert count (${count}).`,
    );
    return null;
  }
  const template = experts.get(ids[0]!)!;
  if (ids.slice(1).some((i) => !experts.get(i)!.equals(template))) {
    warnings.push("MoE breakdown skipped, as the experts in the weights are not uniform.");
    return null;
  }
  const total = new Weights();
  for (const expert of experts.values()) total.merge(expert);
  const base = new Weights();
  for (const [dtype, entry] of weights.dtypes) {
    const removed = total.dtypes.get(dtype) ?? new Weights();
    if (entry.params - removed.params || entry.bytes - removed.bytes) {
      base.add(dtype, entry.params - removed.params, entry.bytes - removed.bytes);
    }
  }
  return new MoE(
    count || ids.length,
    pickInt(config, ["num_experts_per_tok", "num_experts_per_token", "top_k"], 1),
    base,
    template,
    total,
  );
}
