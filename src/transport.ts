import type { EstimateOptions, FetchLike } from "./types.js";

export interface RequestPolicy {
  signal: AbortSignal | undefined;
  requestTimeoutMs: number;
  maxRetries: number;
}

export function requestPolicy(options: EstimateOptions): RequestPolicy {
  const requestTimeoutMs = options.requestTimeoutMs ?? 30_000;
  const maxRetries = options.maxRetries ?? 2;
  if (!Number.isSafeInteger(requestTimeoutMs) || requestTimeoutMs <= 0 || requestTimeoutMs > 2_147_483_647) {
    throw new RangeError("requestTimeoutMs must be an integer between 1 and 2147483647.");
  }
  if (!Number.isSafeInteger(maxRetries) || maxRetries < 0 || maxRetries > 10) {
    throw new RangeError("maxRetries must be an integer between 0 and 10.");
  }
  return { signal: options.signal, requestTimeoutMs, maxRetries };
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

function abortable<T>(work: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const abort = () => reject(abortReason(signal));
    signal.addEventListener("abort", abort, { once: true });
    // Observe work even when cancellation won the race.
    work.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
    if (signal.aborted) abort();
  });
}

async function backoff(ms: number, signal: AbortSignal): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await abortable(new Promise<void>((resolve) => { timer = setTimeout(resolve, ms); }), signal);
  } finally {
    clearTimeout(timer);
  }
}

function retryDelay(response: Response | undefined, attempt: number): number {
  const value = response?.headers.get("retry-after");
  if (value) {
    const seconds = Number(value);
    const delay = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(value) - Date.now();
    if (Number.isFinite(delay)) return Math.max(0, Math.min(5_000, delay));
  }
  return Math.min(5_000, 250 * 2 ** attempt);
}

/** One request limiter per model, held until its response body is consumed or cancelled. */
export function transportFetch(fetcher: FetchLike, concurrency: number, policy: RequestPolicy): FetchLike {
  let active = 0;
  const waiting: Array<() => void> = [];
  async function acquire(): Promise<() => void> {
    policy.signal?.throwIfAborted();
    if (active >= concurrency) {
      let wake!: () => void;
      const ready = new Promise<void>((resolve) => { wake = resolve; waiting.push(wake); });
      try {
        if (policy.signal) await abortable(ready, policy.signal);
        else await ready;
      } catch (error) {
        const index = waiting.indexOf(wake);
        if (index >= 0) waiting.splice(index, 1);
        else release();
        throw error;
      }
    } else active++;
    return release;
  }
  function release(): void {
    const next = waiting.shift();
    if (next) next();
    else active--;
  }

  return async (input, init) => {
    const release = await acquire();
    const controller = new AbortController();
    const signals = [policy.signal, init?.signal].filter((signal): signal is AbortSignal => Boolean(signal));
    const abort = () => {
      const source = signals.find((signal) => signal.aborted);
      controller.abort(source ? abortReason(source) : undefined);
    };
    for (const signal of signals) signal.addEventListener("abort", abort, { once: true });
    if (signals.some((signal) => signal.aborted)) abort();
    const timer = setTimeout(() => controller.abort(new DOMException("Metadata request timed out.", "TimeoutError")), policy.requestTimeoutMs);
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      for (const signal of signals) signal.removeEventListener("abort", abort);
      release();
    };
    try {
      const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
      const retryable = method === "GET" || method === "HEAD";
      let response: Response;
      for (let attempt = 0; ; attempt++) {
        controller.signal.throwIfAborted();
        try {
          const pending = fetcher(input, { ...init, signal: controller.signal });
          // An injected fetch may ignore cancellation and resolve after the deadline.
          void pending.then((late) => { if (controller.signal.aborted) void late.body?.cancel().catch(() => {}); }, () => {});
          response = await abortable(pending, controller.signal);
        } catch (error) {
          if (controller.signal.aborted || !retryable || attempt >= policy.maxRetries || !(error instanceof TypeError)) throw error;
          await backoff(retryDelay(undefined, attempt), controller.signal);
          continue;
        }
        if (!retryable || attempt >= policy.maxRetries || ![429, 502, 503, 504].includes(response.status)) break;
        void response.body?.cancel().catch(() => {});
        await backoff(retryDelay(response, attempt), controller.signal);
      }
      if (!response.ok || !response.body) {
        void response.body?.cancel().catch(() => {});
        finish();
        return new Response(null, { status: response.status, statusText: response.statusText, headers: response.headers });
      }
      const reader = response.body.getReader();
      let bodyController: ReadableStreamDefaultController<Uint8Array>;
      const cancel = () => {
        bodyController.error(abortReason(controller.signal));
        void reader.cancel(abortReason(controller.signal)).catch(() => {});
        finish();
      };
      const close = () => {
        controller.signal.removeEventListener("abort", cancel);
        finish();
      };
      const body = new ReadableStream<Uint8Array>({
        start(streamController) {
          bodyController = streamController;
          controller.signal.addEventListener("abort", cancel, { once: true });
          if (controller.signal.aborted) cancel();
        },
        async pull(streamController) {
          try {
            const { value, done } = await abortable(reader.read(), controller.signal);
            if (done) { close(); streamController.close(); }
            else streamController.enqueue(value);
          } catch (error) {
            close();
            if (!controller.signal.aborted) streamController.error(error);
          }
        },
        cancel(reason) {
          close();
          void reader.cancel(reason).catch(() => {});
        },
      }, { highWaterMark: 0 });
      return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
    } catch (error) {
      finish();
      throw error;
    }
  };
}
