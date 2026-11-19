import { VERSION } from "./version.js";
import { Weights, type WeightsDict } from "./weights.js";

export type Format = "safetensors" | "gguf" | "onnx";

export class KVCache {
  constructor(
    public bytes: number,
    public dtype: string,
    public maxModelLen: number,
    public batchSize: number,
    public layers: Record<string, number>,
  ) {}

  toDict() {
    return {
      bytes: this.bytes,
      dtype: this.dtype,
      max_model_len: this.maxModelLen,
      batch_size: this.batchSize,
      layers: Object.fromEntries(Object.entries(this.layers).filter(([, v]) => v)),
    };
  }
}

export class Overhead {
  constructor(
    public activations: number,
    public encoder: number,
    public graphs: number,
    public runtime: number,
    public context: number,
    public buffers = 0,
    public loading = 0,
  ) {}

  get bytes(): number {
    return this.activations + this.encoder + this.graphs + this.runtime + this.buffers + this.loading;
  }

  toDict() {
    return {
      bytes: this.bytes,
      activations: this.activations,
      encoder: this.encoder,
      graphs: this.graphs,
      runtime: this.runtime,
      buffers: this.buffers,
      loading: this.loading,
    };
  }
}

export class Fit {
  constructor(
    public gpuMemory: number,
    public usable: number,
    public required: number,
    public gpus: number | null,
    public needed: number | null,
    public maxModelLen: number,
  ) {}

  get fits(): boolean {
    return this.required <= (this.gpus || 1) * this.usable;
  }

  toDict() {
    return {
      gpu_memory: this.gpuMemory,
      usable: this.usable,
      required: this.required,
      gpus: this.gpus,
      needed: this.needed,
      fits: this.fits,
      max_model_len: this.maxModelLen,
    };
  }
}

export class MoE {
  constructor(
    public experts: number,
    public active: number | null,
    public base: Weights,
    public expert: Weights,
    public expertsTotal: Weights,
  ) {}

  toDict() {
    return {
      experts: this.experts,
      active: this.active,
      base: this.base.toDict(),
      expert: this.expert.toDict(),
      experts_total: { bytes: this.expertsTotal.bytes, params: this.expertsTotal.params },
    };
  }
}

export interface EstimateInit {
  model: string;
  revision: string | null;
  url: string;
  format: Format;
  weights: Weights;
  components?: Map<string, Weights>;
  offloaded?: Weights | null;
  kvCache?: KVCache | null;
  moe?: MoE | null;
  overhead?: Overhead | null;
  fit?: Fit | null;
  file?: string | null;
  warnings?: string[];
}

export class Estimate {
  model: string;
  revision: string | null;
  url: string;
  format: Format;
  weights: Weights;
  components: Map<string, Weights>;
  offloaded: Weights | null;
  kvCache: KVCache | null;
  moe: MoE | null;
  overhead: Overhead | null;
  fit: Fit | null;
  file: string | null;
  warnings: string[];

  constructor(init: EstimateInit) {
    this.model = init.model;
    this.revision = init.revision;
    this.url = init.url;
    this.format = init.format;
    this.weights = init.weights;
    this.components = init.components ?? new Map();
    this.offloaded = init.offloaded ?? null;
    this.kvCache = init.kvCache ?? null;
    this.moe = init.moe ?? null;
    this.overhead = init.overhead ?? null;
    this.fit = init.fit ?? null;
    this.file = init.file ?? null;
    this.warnings = init.warnings ?? [];
  }

  get total(): number {
    return this.weights.bytes + (this.kvCache ? this.kvCache.bytes : 0) + (this.overhead ? this.overhead.bytes : 0);
  }

  toDict() {
    let components: Record<string, WeightsDict> | null = null;
    if (this.components.size > 1) {
      components = {};
      for (const [name, w] of this.components) components[name] = w.toDict();
    }
    return {
      version: `hf-mem-ts@${VERSION}`,
      model: this.model,
      revision: this.revision,
      file: this.file,
      format: this.format,
      total: this.total,
      weights: this.weights.toDict(),
      offloaded: this.offloaded ? this.offloaded.toDict() : null,
      components,
      kv_cache: this.kvCache ? this.kvCache.toDict() : null,
      overhead: this.overhead ? this.overhead.toDict() : null,
      moe: this.moe ? this.moe.toDict() : null,
      fit: this.fit ? this.fit.toDict() : null,
      warnings: this.warnings,
    };
  }
}

export class Listing {
  constructor(
    public model: string,
    public revision: string | null,
    public url: string,
    public files: Estimate[],
    public warnings: string[] = [],
  ) {}

  toDict() {
    return {
      version: `hf-mem-ts@${VERSION}`,
      model: this.model,
      revision: this.revision,
      format: this.files[0]!.format,
      files: this.files.map((f) => ({
        file: f.file,
        total: f.total,
        weights: { bytes: f.weights.bytes, params: f.weights.params || null },
        kv_cache: f.kvCache ? f.kvCache.toDict() : null,
        overhead: f.overhead ? f.overhead.toDict() : null,
      })),
      warnings: this.warnings,
    };
  }
}
