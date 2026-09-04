import { fetchRange } from "./http.js";
import type { ComponentStats, FetchLike, WeightMetadata } from "./types.js";

export const SAFETENSORS_DTYPE_BYTES: Readonly<Record<string, number>> = Object.freeze({
  F64: 8, I64: 8, U64: 8,
  F32: 4, I32: 4, U32: 4,
  F16: 2, BF16: 2, I16: 2, U16: 2,
  F8_E8M0: 1, F8_E5M2: 1, F8_E4M3: 1, I8: 1, U8: 1, BOOL: 1,
});

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

function safeCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError("Safetensors count or byte total exceeds JavaScript's safe integer range.");
  return value;
}

function parseHeader(header: unknown): ComponentStats {
  if (header === null || typeof header !== "object" || Array.isArray(header)) throw new Error("Invalid Safetensors header: expected an object.");
  const component: ComponentStats = { parameters: 0, bytes: 0, dtypes: Object.create(null) };
  const intervals: [number, number][] = [];
  for (const [name, tensor] of Object.entries(header)) {
    if (name === "__metadata__") {
      if (tensor === null || typeof tensor !== "object" || Array.isArray(tensor) || !Object.values(tensor).every((value) => typeof value === "string")) {
        throw new Error("Invalid Safetensors __metadata__: expected string values.");
      }
      continue;
    }
    if (tensor === null || typeof tensor !== "object" || Array.isArray(tensor) || !Object.hasOwn(tensor, "dtype") || typeof tensor.dtype !== "string" ||
      !Object.hasOwn(SAFETENSORS_DTYPE_BYTES, tensor.dtype)) throw new Error(`Unsupported or invalid Safetensors dtype for tensor ${name}.`);
    if (!Object.hasOwn(tensor, "shape") || !Array.isArray(tensor.shape)) {
      throw new Error(`Invalid shape for Safetensors tensor ${name}.`);
    }
    for (const dimension of tensor.shape) {
      if (!Number.isSafeInteger(dimension) || dimension < 0) throw new Error(`Invalid shape for Safetensors tensor ${name}.`);
    }
    // A scalar has one element; any zero dimension makes an empty tensor.
    const count = tensor.shape.includes(0) ? 0 : tensor.shape.reduce((total: number, dimension: number) => safeCount(total * dimension), 1);
    const tensorBytes = safeCount(count * SAFETENSORS_DTYPE_BYTES[tensor.dtype]!);
    if (Object.hasOwn(tensor, "data_offsets")) {
      const offsets = tensor.data_offsets;
      if (!Array.isArray(offsets) || offsets.length !== 2 ||
        !offsets.every((value) => Number.isSafeInteger(value) && value >= 0) ||
        offsets[1] < offsets[0] || offsets[1] - offsets[0] !== tensorBytes) {
        throw new Error(`Invalid data_offsets for Safetensors tensor ${name}.`);
      }
      if (tensorBytes > 0) intervals.push([offsets[0], offsets[1]]);
    }
    const dtype = component.dtypes[tensor.dtype] ?? { parameters: 0, bytes: 0 };
    dtype.parameters = safeCount(dtype.parameters + count);
    dtype.bytes = safeCount(dtype.bytes + tensorBytes);
    component.dtypes[tensor.dtype] = dtype;
    component.parameters = safeCount(component.parameters + count);
    component.bytes = safeCount(component.bytes + tensorBytes);
  }
  intervals.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < intervals.length; i++) {
    if (intervals[i]![0] < intervals[i - 1]![1]) throw new Error("Overlapping Safetensors data_offsets.");
  }
  return component;
}

export async function fetchSafetensorsHeader(fetcher: FetchLike, url: string, headers: HeadersInit = {}): Promise<SafetensorsHeader> {
  const { bytes: prefix } = await fetchRange(fetcher, url, 0, 7, headers);
  const length = uint64(prefix);
  if (length <= 0 || length > 512 * 1024 * 1024) throw new RangeError(`Invalid Safetensors header length: ${length}.`);
  const { bytes: raw } = await fetchRange(fetcher, url, 8, 7 + length, headers);
  if (raw.byteLength !== length) throw new Error(`Truncated Safetensors metadata from ${url}.`);
  const header: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
  parseHeader(header);
  return header as SafetensorsHeader;
}

export function parseSafetensorsHeaders(headers: Record<string, SafetensorsHeader>): WeightMetadata {
  if (headers === null || typeof headers !== "object" || Array.isArray(headers)) throw new Error("Invalid Safetensors component headers.");
  const components: Record<string, ComponentStats> = Object.create(null);
  let parameters = 0;
  let bytes = 0;
  for (const [componentName, header] of Object.entries(headers)) {
    const component = parseHeader(header);
    parameters = safeCount(parameters + component.parameters);
    bytes = safeCount(bytes + component.bytes);
    components[componentName] = component;
  }
  return { parameters, bytes, components };
}
