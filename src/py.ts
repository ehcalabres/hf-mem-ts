/** Ceiling division of integers. */
export function cdiv(a: number, b: number): number {
  return -Math.floor(-a / b);
}

/** Python's `int(...)`: truncation toward zero. */
export function int(value: number): number {
  return Math.trunc(value);
}

/** Python's `round(...)` to an integer: halves round to even. */
export function round(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Python's `f"{value:.{digits}f}"`, which rounds the exact binary value with halves to even. */
export function fixed(value: number, digits: number): string {
  const text = value.toFixed(digits);
  const scaled = value * 10 ** digits * 2;
  if (Number.isInteger(scaled) && Math.abs(scaled % 2) === 1 && scaled / 2 / 10 ** digits === value) {
    const even = round(scaled / 2);
    return (even / 10 ** digits).toFixed(digits);
  }
  return text;
}

/** Number of bits needed to represent a non-negative integer. */
export function bitLength(value: number): number {
  return value <= 0 ? 0 : Math.floor(Math.log2(value)) + 1;
}

export function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return Math.abs(a);
}

export function lcm(a: number, b: number): number {
  return a && b ? Math.abs(a * b) / gcd(a, b) : 0;
}

export function product(values: number[]): number {
  return values.reduce((a, b) => a * b, 1);
}

export function basename(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

export function dirname(path: string): string {
  const index = path.lastIndexOf("/");
  return index < 0 ? "" : path.slice(0, index).replace(/\/+$/, "") || (index === 0 ? "/" : "");
}

export function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Python's `isinstance(value, int)` for JSON values (booleans are not integers in JSON). */
export function isInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

/** Python truthiness for JSON values. */
export function truthy(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (isRecord(value)) return Object.keys(value).length > 0;
  return Boolean(value);
}

/** Python's `a or b` for JSON values. */
export function or<T>(...values: T[]): T {
  for (const value of values.slice(0, -1)) if (truthy(value)) return value;
  return values[values.length - 1] as T;
}

/** Python's `str(value)` for JSON scalars. */
export function str(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  return String(value);
}

export function count<T>(values: T[], item: T): number {
  let n = 0;
  for (const value of values) if (value === item) n += 1;
  return n;
}

export function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Serializes like Python's `json.dumps`: `", "` / `": "` separators and ASCII-only strings. */
export function dumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    const text = String(value);
    return /[.eE]/.test(text) ? text.replace(/e([+-])(\d)$/, "e$10$2") : `${text}.0`;
  }
  if (typeof value === "string") {
    return JSON.stringify(value).replace(
      /[\u007f-\uffff]/g,
      (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`,
    );
  }
  if (Array.isArray(value)) return `[${value.map(dumps).join(", ")}]`;
  const entries = Object.entries(value as Record<string, unknown>);
  return `{${entries.map(([k, v]) => `${dumps(k)}: ${dumps(v)}`).join(", ")}}`;
}
