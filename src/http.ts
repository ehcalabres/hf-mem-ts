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
    throw new HttpError(`Hugging Face request failed (${response.status} ${response.statusText}): ${url}`, response.status, url);
  }
  return response;
}

export async function fetchJson<T>(fetcher: FetchLike, url: string, headers: HeadersInit): Promise<T> {
  const response = await checkedFetch(fetcher, url, { headers });
  return response.json() as Promise<T>;
}

export async function fetchRange(
  fetcher: FetchLike,
  url: string,
  start: number,
  end: number,
  headers: HeadersInit,
): Promise<Uint8Array> {
  const response = await checkedFetch(fetcher, url, {
    headers: { ...headers, Range: `bytes=${start}-${end}` },
  });
  if (response.status !== 206) {
    await response.body?.cancel();
    throw new Error(`Server ignored the HTTP Range request for ${url}; refusing to download the complete model file.`);
  }
  return new Uint8Array(await response.arrayBuffer());
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
