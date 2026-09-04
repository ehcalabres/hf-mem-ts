import type { FetchLike } from "./types.js";

export class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly url: string,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

export async function checkedFetch(fetcher: FetchLike, url: string, init?: RequestInit): Promise<Response> {
  const response = await fetcher(url, init);
  if (!response.ok) {
    await response.body?.cancel();
    throw new HttpError(`Hugging Face request failed (${response.status} ${response.statusText}): ${url}`, response.status, url);
  }
  return response;
}

async function readBoundedBody(response: Response, limit: number, exact = false): Promise<Uint8Array> {
  const contentLength = response.headers.get("content-length");
  if (contentLength !== null && (!/^\d+$/.test(contentLength) || !Number.isSafeInteger(Number(contentLength)) ||
    Number(contentLength) > limit || (exact && Number(contentLength) !== limit))) {
    await response.body?.cancel();
    throw new Error("Invalid or oversized HTTP Content-Length.");
  }
  if (!response.body) throw new Error("Missing HTTP response body.");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  let exactBytes: Uint8Array | undefined;
  try {
    if (exact) exactBytes = new Uint8Array(limit);
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value.byteLength > limit - length) throw new RangeError("HTTP response exceeds the metadata byte budget.");
      if (exactBytes) exactBytes.set(value, length);
      else if (value.byteLength > 0) chunks.push(value);
      length += value.byteLength;
    }
    if (exact && length !== limit) throw new Error("Truncated HTTP range response.");
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (exactBytes) return exactBytes;
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes;
}

export async function fetchJson<T>(fetcher: FetchLike, url: string, headers: HeadersInit): Promise<T> {
  const response = await checkedFetch(fetcher, url, { headers });
  const bytes = await readBoundedBody(response, 32 * 1024 * 1024);
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as T;
}

export async function fetchRange(
  fetcher: FetchLike,
  url: string,
  start: number,
  end: number,
  headers: HeadersInit,
): Promise<{ bytes: Uint8Array; eof: boolean }> {
  if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(end) || end < start ||
    !Number.isSafeInteger(end - start + 1)) throw new RangeError("Invalid HTTP byte range.");
  const requestHeaders = new Headers(headers);
  requestHeaders.set("Range", `bytes=${start}-${end}`);
  const response = await checkedFetch(fetcher, url, { headers: requestHeaders });
  if (response.status !== 206) {
    await response.body?.cancel();
    throw new Error(`Server ignored the HTTP Range request for ${url}; refusing to download the complete model file.`);
  }
  const match = /^bytes (\d+)-(\d+)\/(\d+|\*)$/i.exec(response.headers.get("content-range") ?? "");
  const first = Number(match?.[1]);
  const last = Number(match?.[2]);
  const total = match?.[3] === "*" ? undefined : Number(match?.[3]);
  if (!match || !Number.isSafeInteger(first) || !Number.isSafeInteger(last) || first !== start ||
    last < first || last > end || (total !== undefined && (!Number.isSafeInteger(total) || total <= last)) ||
    (last < end && (total === undefined || last !== total - 1))) {
    await response.body?.cancel();
    throw new Error(`Invalid HTTP Content-Range from ${url}.`);
  }
  return { bytes: await readBoundedBody(response, last - first + 1, true), eof: total !== undefined && last === total - 1 };
}

export function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`${name} must be a positive safe integer.`);
}

export async function mapLimit<T, U>(items: readonly T[], limit: number, fn: (item: T) => Promise<U>): Promise<U[]> {
  assertPositiveInteger(limit, "concurrency");
  const output = new Array<U>(items.length);
  let cursor = 0;
  async function worker(): Promise<void> {
    while (cursor < items.length) {
      const index = cursor++;
      output[index] = await fn(items[index]!);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return output;
}
