import type * as Fs from "node:fs";
import type * as Os from "node:os";
import type * as Path from "node:path";
import { VERSION } from "./version.js";

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

type Env = Record<string, string | undefined>;
const env: Env = (globalThis as { process?: { env?: Env } }).process?.env ?? {};

export const ENDPOINT = (env.HF_ENDPOINT || "https://huggingface.co").replace(/\/+$/, "");
const CONCURRENCY = Number.parseInt(env.MAX_WORKERS ?? "32", 10);
const TIMEOUT = Number.parseFloat(env.REQUEST_TIMEOUT ?? "30");
const RETRIES = 3;

const TAG = /^([^/:@\s]+\/[^/:@\s]+(?:@[^:\s]+)?):([\w.-]+)$/;
const URL_PATH = /^\/(?<id>[^/]+\/[^/]+)(?:\/(?:tree|blob|resolve)\/(?<rev>[^/]+)(?:\/(?<file>.+))?)?\/?$/;
const AUTH = "authenticate with `--hf-token`, `HF_TOKEN` or `hf auth login`";
const SHA = /^[0-9a-f]{40}$/;

export class HubError extends Error {
  constructor(
    message: string,
    public status: number | null = null,
  ) {
    super(message);
    this.name = "HubError";
  }
}

interface Node {
  fs: typeof Fs;
  path: typeof Path;
  os: typeof Os;
}

let node: Promise<Node | null> | undefined;

/** Node built-ins, loaded lazily so the library also runs where they are missing (browsers, edge runtimes). */
export function nodeModules(): Promise<Node | null> {
  node ??= (async () => {
    if (!(globalThis as { process?: { versions?: { node?: string } } }).process?.versions?.node) return null;
    try {
      const [fs, path, os] = await Promise.all([import("node:fs"), import("node:path"), import("node:os")]);
      return { fs, path, os };
    } catch {
      return null;
    }
  })();
  return node;
}

function truthyEnv(name: string): boolean {
  return ["1", "true", "yes", "on"].includes((env[name] ?? "").toLowerCase());
}

function expandUser(n: Node, path: string): string {
  return path === "~" || path.startsWith("~/") ? n.os.homedir() + path.slice(1) : path;
}

function hfHome(n: Node): string {
  return expandUser(n, env.HF_HOME || n.path.join("~", ".cache", "huggingface"));
}

async function resolveToken(token: string | null | undefined): Promise<string | null> {
  if (token) return token;
  if (env.HF_TOKEN) return env.HF_TOKEN;
  const n = await nodeModules();
  if (!n) return null;
  const path = env.HF_TOKEN_PATH || n.path.join(hfHome(n), "token");
  if (isFile(n, path)) return n.fs.readFileSync(path, "utf-8").trim() || null;
  return null;
}

function isFile(n: Node, path: string): boolean {
  try {
    return n.fs.statSync(path).isFile();
  } catch {
    return false;
  }
}

function isDir(n: Node, path: string): boolean {
  try {
    return n.fs.statSync(path).isDirectory();
  } catch {
    return false;
  }
}

function cacheRepo(n: Node, modelId: string): string {
  const root = env.HF_HUB_CACHE || n.path.join(hfHome(n), "hub");
  return n.path.join(expandUser(n, root), "models--" + modelId.replaceAll("/", "--"));
}

function cacheSnapshot(n: Node, modelId: string, revision: string): string | null {
  const repo = cacheRepo(n, modelId);
  let sha = revision;
  if (!SHA.test(revision)) {
    const ref = n.path.join(repo, "refs", revision);
    if (!isFile(n, ref)) return null;
    sha = n.fs.readFileSync(ref, "utf-8").trim();
  }
  const path = n.path.join(repo, "snapshots", sha);
  return isDir(n, path) ? path : null;
}

/** Splits `org/name@revision:TAG` or a Hub URL into the model ID, revision and file or quantization tag. */
export async function parseModel(
  model: string,
  revision: string | null = null,
): Promise<[string, string | null, string | null]> {
  if (model.startsWith("http://") || model.startsWith("https://")) {
    const match = URL_PATH.exec(new URL(model).pathname);
    if (match?.groups) {
      const decode = (v: string | undefined) => (v ? decodeURIComponent(v) : null);
      return [match.groups.id!, revision || decode(match.groups.rev), decode(match.groups.file)];
    }
    throw new Error(`\`${model}\` is not a valid Hugging Face Hub model URL.`);
  }
  const n = await nodeModules();
  const exists = (path: string) => n !== null && n.fs.existsSync(path);
  let tag: string | null = null;
  const match = TAG.exec(model);
  if (match && !exists(model)) {
    model = match[1]!;
    tag = match[2]!;
  }
  if (model.includes("@") && !exists(model)) {
    const at = model.indexOf("@");
    revision = revision || model.slice(at + 1);
    model = model.slice(0, at);
  }
  return [model, revision, tag];
}

export type ProgressCallback = (done: number, total: number) => void;

export class Progress {
  done = 0;
  total = 0;

  constructor(public callback: ProgressCallback | null = null) {}

  start(): void {
    this.total += 1;
    this.callback?.(this.done, this.total);
  }

  finish(): void {
    this.done += 1;
    this.callback?.(this.done, this.total);
  }
}

export interface Source {
  name: string;
  revision: string | null;
  url: string;
  files: Map<string, number>;
  json(path: string): Promise<any>;
  read(path: string, start: number, end: number): Promise<Uint8Array>;
  close(): Promise<void>;
}

export class Semaphore {
  private queue: (() => void)[] = [];
  private active = 0;

  constructor(private limit: number) {}

  async run<T>(task: () => Promise<T>): Promise<T> {
    if (this.active >= this.limit) await new Promise<void>((resolve) => this.queue.push(resolve));
    this.active += 1;
    try {
      return await task();
    } finally {
      this.active -= 1;
      this.queue.shift()?.();
    }
  }
}

function localRevision(n: Node, root: string): string | null {
  const folder = n.path.join(root, ".cache", "huggingface", "download");
  if (!isDir(n, folder)) return null;
  const names = n.fs
    .readdirSync(folder)
    .sort()
    .sort((a, b) => Number(a !== "config.json.metadata") - Number(b !== "config.json.metadata"));
  for (const name of names) {
    if (name.endsWith(".metadata")) {
      const first = n.fs.readFileSync(n.path.join(folder, name), "utf-8").split("\n")[0]!.trim();
      return first || null;
    }
  }
  return null;
}

export class LocalSource implements Source {
  revision: string | null = null;
  url: string;

  constructor(
    private n: Node,
    public root: string,
    public files: Map<string, number>,
    public name: string,
    public progress: Progress,
  ) {
    this.url = name;
  }

  static open(n: Node, path: string, name: string | null = null, progress: Progress | null = null, recursive = true) {
    const root = n.path.resolve(expandUser(n, path));
    const files = new Map<string, number>();
    const walk = (directory: string) => {
      const entries = n.fs.readdirSync(directory).sort();
      const dirs: string[] = [];
      for (const entry of entries) {
        const full = n.path.join(directory, entry);
        let stat;
        try {
          stat = n.fs.statSync(full);
        } catch {
          continue;
        }
        if (stat.isDirectory()) {
          if (!entry.startsWith(".")) dirs.push(full);
        } else if (stat.isFile() && !entry.startsWith(".")) {
          files.set(n.path.relative(root, full).split(n.path.sep).join("/"), stat.size);
        }
      }
      if (recursive) for (const dir of dirs) walk(dir);
    };
    walk(root);
    const source = new LocalSource(n, root, files, name ?? root, progress ?? new Progress());
    source.revision = localRevision(n, root);
    return source;
  }

  async json(path: string): Promise<any> {
    this.progress.start();
    try {
      return JSON.parse(this.n.fs.readFileSync(this.n.path.join(this.root, path), "utf-8"));
    } finally {
      this.progress.finish();
    }
  }

  async read(path: string, start: number, end: number): Promise<Uint8Array> {
    this.progress.start();
    try {
      return readRange(this.n, this.n.path.join(this.root, path), start, end);
    } finally {
      this.progress.finish();
    }
  }

  async close(): Promise<void> {}
}

function readRange(n: Node, path: string, start: number, end: number): Uint8Array {
  const fd = n.fs.openSync(path, "r");
  try {
    const buffer = new Uint8Array(Math.max(0, end - start));
    const read = n.fs.readSync(fd, buffer, 0, buffer.length, start);
    return buffer.subarray(0, read);
  } finally {
    n.fs.closeSync(fd);
  }
}

function uuid4(): string {
  const crypto = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (crypto?.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function nextLink(header: string | null): string | null {
  if (!header) return null;
  for (const part of header.split(",")) {
    const match = /<([^>]+)>\s*;\s*rel="?next"?/.exec(part);
    if (match) return match[1]!;
  }
  return null;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export interface HubOptions {
  fetch?: FetchLike | undefined;
  signal?: AbortSignal | undefined;
}

export class HubSource implements Source {
  url: string;
  files = new Map<string, number>();
  authenticated: boolean;
  private oids = new Map<string, string>();
  private snapshot: string | null = null;
  private base: string;
  private headers: Record<string, string>;
  private semaphore = new Semaphore(CONCURRENCY);
  private fetch: FetchLike;
  private signal: AbortSignal | undefined;
  private node: Node | null = null;

  constructor(
    public name: string,
    public revision: string,
    token: string | null,
    public progress: Progress,
    options: HubOptions = {},
  ) {
    this.url = `https://hf.co/${name}`;
    this.base = `${ENDPOINT}/${name}/resolve/${encodeURIComponent(revision)}`;
    this.headers = { "User-Agent": `hf-mem-ts/${VERSION}; id=${uuid4()}; model_id=${name}; revision=${revision}` };
    if (token) this.headers.Authorization = `Bearer ${token}`;
    this.authenticated = Boolean(token);
    this.fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init));
    this.signal = options.signal;
  }

  private async request(url: string, headers: Record<string, string> = {}): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error(`timed out after ${TIMEOUT}s`)), TIMEOUT * 1000);
    const abort = () => controller.abort(this.signal?.reason);
    this.signal?.addEventListener("abort", abort, { once: true });
    if (this.signal?.aborted) abort();
    try {
      const response = await this.fetch(url, { headers: { ...this.headers, ...headers }, signal: controller.signal });
      const body = new Uint8Array(await response.arrayBuffer());
      return new Response(response.status === 204 || response.status === 304 ? null : body, {
        status: response.status,
        headers: response.headers,
      });
    } finally {
      clearTimeout(timer);
      this.signal?.removeEventListener("abort", abort);
    }
  }

  private async get(url: string, headers: Record<string, string> = {}): Promise<Response> {
    this.progress.start();
    try {
      return await this.semaphore.run(async () => {
        for (let attempt = 0; ; attempt++) {
          let response: Response;
          try {
            response = await this.request(url, headers);
          } catch (e) {
            if (this.signal?.aborted) throw e;
            if (attempt === RETRIES - 1) {
              const error = e as Error;
              throw new HubError(
                `Could not reach ${ENDPOINT} (${error.name}: ${error.message}), check the connection, or set ` +
                  "`HF_HUB_OFFLINE=1` to read models from the local cache.",
              );
            }
            await sleep(500 * 2 ** attempt);
            continue;
          }
          if (![429, 500, 502, 503, 504].includes(response.status) || attempt === RETRIES - 1) {
            if (response.status >= 400) throw this.error(response);
            return response;
          }
          await sleep(500 * 2 ** attempt);
        }
      });
    } finally {
      this.progress.finish();
    }
  }

  private error(response: Response): HubError {
    const status = response.status;
    const message = response.headers.get("x-error-message") || "";
    const page = `https://hf.co/${this.name}`;
    let text: string;
    if (status === 404 && message.toLowerCase().includes("rev")) {
      text = `Revision \`${this.revision}\` not found for \`${this.name}\`, see the branches and tags at ${page}.`;
    } else if ((status === 401 || status === 404) && !message.toLowerCase().includes("restricted")) {
      text =
        `\`${this.name}\` was not found on the Hub` +
        (this.authenticated ? ", or the token has no access to it." : `, or it is private (${AUTH}).`);
    } else if (status === 401 || status === 403) {
      text = `\`${this.name}\` is gated, request access at ${page}` + (this.authenticated ? "." : ` and ${AUTH}.`);
    } else {
      text = `The Hub returned HTTP ${status} for \`${this.name}\`` + (message ? `: ${message}` : ".");
    }
    return new HubError(text, status);
  }

  async list(): Promise<void> {
    let url: string | null =
      `${ENDPOINT}/api/models/${this.name}/tree/${encodeURIComponent(this.revision)}?recursive=true`;
    let commit: string | null = null;
    while (url) {
      const response = await this.get(url);
      commit = commit || response.headers.get("x-repo-commit");
      for (const entry of (await response.json()) as any[]) {
        if (entry.type === "file") {
          const lfs = entry.lfs || {};
          this.files.set(entry.path, lfs.size ?? entry.size ?? 0);
          this.oids.set(entry.path, lfs.oid || entry.oid || "");
        }
      }
      url = nextLink(response.headers.get("link"));
    }
    this.node = await nodeModules();
    this.snapshot = this.node ? cacheSnapshot(this.node, this.name, commit || this.revision) : null;
  }

  private cached(path: string): string | null {
    const n = this.node;
    if (this.snapshot === null || n === null) return null;
    const full = n.path.join(this.snapshot, path);
    try {
      if (n.fs.lstatSync(full).isSymbolicLink() && n.path.basename(n.fs.readlinkSync(full)) === this.oids.get(path)) {
        return full;
      }
    } catch {
      return null;
    }
    return null;
  }

  async json(path: string): Promise<any> {
    const cached = this.cached(path);
    if (cached) return JSON.parse(this.node!.fs.readFileSync(cached, "utf-8"));
    return JSON.parse(await (await this.get(`${this.base}/${path}`)).text());
  }

  async read(path: string, start: number, end: number): Promise<Uint8Array> {
    end = Math.min(end, this.files.get(path) ?? end);
    if (start >= end) return new Uint8Array(0);
    const cached = this.cached(path);
    if (cached) return readRange(this.node!, cached, start, end);
    const response = await this.get(`${this.base}/${path}`, { Range: `bytes=${start}-${end - 1}` });
    return new Uint8Array(await response.arrayBuffer()).subarray(0, end - start);
  }

  async close(): Promise<void> {}
}

export async function openSource(
  model: string,
  revision: string | null = null,
  token: string | null = null,
  progress: Progress | null = null,
  options: HubOptions = {},
): Promise<[Source, string | null]> {
  progress ??= new Progress();
  const n = await nodeModules();
  if (n) {
    const path = expandUser(n, model);
    if (isFile(n, path)) {
      const root = n.path.dirname(n.path.resolve(path));
      return [LocalSource.open(n, root, null, progress, false), n.path.basename(path)];
    }
    if (isDir(n, path)) return [LocalSource.open(n, path, null, progress), null];
  }

  const [modelId, parsedRevision, file] = await parseModel(model, revision);
  const resolved = parsedRevision || "main";
  const slashes = modelId.split("/").length - 1;
  if (
    model.startsWith(".") ||
    model.startsWith("/") ||
    model.startsWith("~") ||
    (model.replace(/^\/+|\/+$/g, "").includes("/") && slashes !== 1)
  ) {
    throw new Error(`The local path \`${model}\` does not exist.`);
  }
  if (slashes !== 1) {
    throw new Error(`\`${model}\` is not a Hub model ID (\`org/name\`), a Hub URL or an existing local path.`);
  }

  if (truthyEnv("HF_HUB_OFFLINE")) {
    const snapshot = n ? cacheSnapshot(n, modelId, resolved) : null;
    if (n && snapshot) {
      const source = LocalSource.open(n, snapshot, modelId, progress);
      source.revision = resolved;
      source.url = `https://hf.co/${modelId}`;
      return [source, file];
    }
    throw new Error(
      `\`${modelId}\` @ \`${resolved}\` is not in the local cache and \`HF_HUB_OFFLINE\` is set, download it first ` +
        `with \`hf download ${modelId}\` or unset \`HF_HUB_OFFLINE\`.`,
    );
  }

  const hub = new HubSource(modelId, resolved, await resolveToken(token), progress, options);
  try {
    await hub.list();
  } catch (e) {
    await hub.close();
    throw e;
  }
  return [hub, file];
}
