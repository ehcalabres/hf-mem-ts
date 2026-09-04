import { fetchRange } from "./http.js";
import type { ComponentStats, FetchLike, WeightMetadata } from "./types.js";

export const SAFETENSORS_DTYPE_BYTES: Readonly<Record<string, number>> = {
  F64: 8, I64: 8, U64: 8,
  F32: 4, I32: 4, U32: 4,
  F16: 2, BF16: 2, I16: 2, U16: 2,
  F8_E8M0: 1, F8_E5M2: 1, F8_E4M3: 1, I8: 1, U8: 1, BOOL: 1,
};

export interface SafetensorsTensorInfo {
  dtype: string;
  shape: number[];
  data_offsets?: [number, number];
}

export type SafetensorsHeader = Record<string, SafetensorsTensorInfo | Record<string, string>>;

function uint64(bytes: Uint8Array): number {
  if (bytes.byteLength < 8) throw new Error("Truncated Safetensors header length.");
  const value = new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new RangeError("Safetensors header is too large.");
  return Number(value);
}

export async function fetchSafetensorsHeader(fetcher: FetchLike, url: string, headers: HeadersInit = {}): Promise<SafetensorsHeader> {
  const prefix = await fetchRange(fetcher, url, 0, 7, headers);
  const length = uint64(prefix);
  if (length <= 0 || length > 512 * 1024 * 1024) throw new RangeError(`Invalid Safetensors header length: ${length}.`);
  const raw = await fetchRange(fetcher, url, 8, 7 + length, headers);
  if (raw.byteLength !== length) throw new Error(`Truncated Safetensors metadata from ${url}.`);
  return JSON.parse(new TextDecoder().decode(raw)) as SafetensorsHeader;
}

export function parseSafetensorsHeaders(headers: Record<string, SafetensorsHeader>): WeightMetadata {
  const components: Record<string, ComponentStats> = {};
  let parameters = 0;
  let bytes = 0;
  for (const [componentName, header] of Object.entries(headers)) {
    const component: ComponentStats = { parameters: 0, bytes: 0, dtypes: {} };
    for (const [name, raw] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const tensor = raw as SafetensorsTensorInfo;
      const byteWidth = SAFETENSORS_DTYPE_BYTES[tensor.dtype];
      if (byteWidth === undefined) throw new Error(`Unsupported Safetensors dtype: ${tensor.dtype}.`);
      if (!Array.isArray(tensor.shape) || !tensor.shape.every((v) => Number.isSafeInteger(v) && v >= 0)) {
        throw new Error(`Invalid shape for Safetensors tensor ${name}.`);
      }
      const count = tensor.shape.reduce((total, dimension) => total * dimension, 1);
      if (!Number.isSafeInteger(count)) throw new RangeError(`Tensor ${name} parameter count exceeds JavaScript's safe integer range.`);
      const tensorBytes = count * byteWidth;
      const dtype = component.dtypes[tensor.dtype] ?? { parameters: 0, bytes: 0 };
      dtype.parameters += count;
      dtype.bytes += tensorBytes;
      component.dtypes[tensor.dtype] = dtype;
      component.parameters += count;
      component.bytes += tensorBytes;
      parameters += count;
      bytes += tensorBytes;
    }
    components[componentName] = component;
  }
  return { parameters, bytes, components };
}
