export const DTYPES: Record<string, [string, number]> = {
  float16: ["F16", 2],
  bfloat16: ["BF16", 2],
  float32: ["F32", 4],
};
export const FLOAT_BYTES: Record<string, number> = { F64: 8, F32: 4, F16: 2, BF16: 2 };

export interface WeightsDict {
  bytes: number;
  params: number | null;
  dtypes?: Record<string, { bytes: number; params: number }>;
}

export class Weights {
  params: number;
  bytes: number;
  dtypes: Map<string, Weights>;
  largest: number;

  constructor(params = 0, bytes = 0) {
    this.params = params;
    this.bytes = bytes;
    this.dtypes = new Map();
    this.largest = 0;
  }

  add(dtype: string, params: number, nbytes: number, largest = 0): void {
    this.params += params;
    this.bytes += nbytes;
    this.largest = Math.max(this.largest, largest);
    let entry = this.dtypes.get(dtype);
    if (entry === undefined) {
      entry = new Weights();
      this.dtypes.set(dtype, entry);
    }
    entry.params += params;
    entry.bytes += nbytes;
    entry.largest = Math.max(entry.largest, largest);
  }

  merge(other: Weights): this {
    for (const [dtype, entry] of other.dtypes) this.add(dtype, entry.params, entry.bytes, entry.largest);
    if (other.dtypes.size === 0) {
      this.params += other.params;
      this.bytes += other.bytes;
    }
    return this;
  }

  cast(dtype: string): Weights {
    const [target, size] = DTYPES[dtype]!;
    const result = new Weights();
    for (const [name, entry] of this.dtypes) {
      const width = FLOAT_BYTES[name];
      if (width !== undefined) {
        if (entry.params) result.add(target, entry.params, entry.params * size, entry.largest);
        const buffers = entry.bytes - entry.params * width;
        if (buffers) result.add(name, 0, buffers);
      } else {
        result.add(name, entry.params, entry.bytes, entry.largest);
      }
    }
    if (this.dtypes.size === 0) {
      result.params = this.params;
      result.bytes = this.bytes;
    }
    return result;
  }

  /** Equality on params, bytes and per-dtype entries; `largest` is ignored. */
  equals(other: Weights): boolean {
    if (this.params !== other.params || this.bytes !== other.bytes || this.dtypes.size !== other.dtypes.size) {
      return false;
    }
    for (const [dtype, entry] of this.dtypes) {
      const match = other.dtypes.get(dtype);
      if (match === undefined || !entry.equals(match)) return false;
    }
    return true;
  }

  sortedDtypes(): [string, Weights][] {
    return [...this.dtypes.entries()].sort((a, b) => b[1].bytes - a[1].bytes);
  }

  toDict(): WeightsDict {
    const out: WeightsDict = { bytes: this.bytes, params: this.params || null };
    if (this.dtypes.size) {
      out.dtypes = {};
      for (const [d, w] of this.sortedDtypes()) out.dtypes[d] = { bytes: w.bytes, params: w.params };
    }
    return out;
  }
}

export function choices(names: string[], limit = 8): string {
  const sorted = [...names].sort();
  const shown = sorted
    .slice(0, limit)
    .map((n) => `\`${n}\``)
    .join(", ");
  return shown + (names.length > limit ? ` and ${names.length - limit} more` : "");
}
