#!/usr/bin/env node
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { KV_CACHE_DTYPES } from "./config.js";
import { FORMATS, arun, type RunOptions } from "./estimate.js";
import { CACHE_TYPES } from "./gguf.js";
import { dumps } from "./py.js";
import { markdown, table } from "./render.js";
import type { Estimate, Listing } from "./result.js";
import { HubError, Progress } from "./source.js";
import { VERSION } from "./version.js";
import { DTYPES } from "./weights.js";

const PROG = "hf-mem-ts";

export interface Writer {
  write(chunk: string): unknown;
  isTTY?: boolean;
}

export interface Streams {
  stdout: Writer;
  stderr: Writer;
}

class Status {
  static FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
  enabled: boolean;
  last = 0.0;
  frame = 0;

  constructor(private stream: Writer) {
    this.enabled = Boolean(stream.isTTY);
  }

  update = (done: number, total: number): void => {
    const now = performance.now() / 1000;
    if (!this.enabled || (now - this.last < 0.08 && done < total)) return;
    this.last = now;
    this.frame += 1;
    const width = 24;
    const filled = Math.floor((width * done) / Math.max(total, 1));
    const bar = filled < width ? "━".repeat(filled) + "╺" + "─".repeat(width - filled - 1) : "━".repeat(width);
    const spinner = Status.FRAMES[this.frame % Status.FRAMES.length];
    this.stream.write(`\r\x1b[2K${spinner} Reading metadata ${bar} ${done}/${total} requests`);
  };

  clear(): void {
    if (this.enabled) this.stream.write("\r\x1b[2K");
  }
}

interface Option {
  flags: string[];
  dest: string;
  help: string;
  action: "store" | "store_true" | "help" | "version";
  type?: "int" | "float";
  choices?: readonly string[];
  default?: string | number | boolean;
}

const OPTIONS: Option[] = [
  { flags: ["-h", "--help"], dest: "help", action: "help", help: "show this help message and exit" },
  {
    flags: ["--revision"],
    dest: "revision",
    action: "store",
    help: "Hub revision (branch, tag or commit), defaults to `main`.",
  },
  {
    flags: ["--file"],
    dest: "file",
    action: "store",
    help: "Single file, shard set or directory to estimate.",
  },
  {
    flags: ["--format"],
    dest: "format",
    action: "store",
    choices: FORMATS,
    help: "Weights format to estimate, defaults to Safetensors, then GGUF, then the weight file sizes, then ONNX.",
  },
  {
    flags: ["--max-model-len"],
    dest: "max_model_len",
    action: "store",
    type: "int",
    help: "Context length, defaults to the model config.",
  },
  {
    flags: ["--batch-size"],
    dest: "batch_size",
    action: "store",
    type: "int",
    default: 1,
    help: "Concurrent sequences in the KV cache, defaults to 1.",
  },
  {
    flags: ["--kv-cache-dtype"],
    dest: "kv_cache_dtype",
    action: "store",
    default: "auto",
    help:
      "KV cache dtype, `auto` uses model precision (16-bit for FP32 decoders) or F16 for GGUF. " +
      `Safetensors: ${KV_CACHE_DTYPES.join(", ")}. GGUF: ${CACHE_TYPES.map((t) => t.toLowerCase()).join(", ")}. ONNX: set by the graph.`,
  },
  {
    flags: ["--gpu-memory"],
    dest: "gpu_memory",
    action: "store",
    help: "GPU memory in GiB, e.g. `80` or `8x80`, to check whether it fits.",
  },
  {
    flags: ["--gpu-memory-utilization"],
    dest: "gpu_memory_utilization",
    action: "store",
    type: "float",
    default: 0.92,
    help: "Fraction of the GPU memory that can be used, defaults to 0.92 (as vLLM).",
  },
  {
    flags: ["--max-num-batched-tokens"],
    dest: "max_num_batched_tokens",
    action: "store",
    type: "int",
    help: "Warmup tokens for text encoders, or tokens per batch as in vLLM, defaults to 8192 (16384 with `--gpu-memory` of 160 GiB or more, 2048 under 70 GiB), use 2048 for A100.",
  },
  {
    flags: ["--dtype"],
    dest: "dtype",
    action: "store",
    choices: Object.keys(DTYPES),
    help:
      "Floating-point weight and activation dtype for unquantized Safetensors text encoders only. " +
      "Defaults to checkpoint weights and configured activation dtype. Integer tensors are unchanged.",
  },
  {
    flags: ["--hf-token"],
    dest: "hf_token",
    action: "store",
    help: "Hub token, defaults to `HF_TOKEN` or the `hf auth login` token.",
  },
  { flags: ["--color"], dest: "color", action: "store_true", default: false, help: "Enable color in terminal tables." },
  {
    flags: ["-o", "--output"],
    dest: "output",
    action: "store",
    choices: ["table", "json", "markdown"],
    default: "table",
    help: "Output format, defaults to `table`.",
  },
  { flags: ["--version"], dest: "version", action: "version", help: "show program's version number and exit" },
];

const MODEL_HELP = "Hub model ID (`org/name`, `org/name@revision`, or URL) or local path.";
const DESCRIPTION =
  "Estimate the memory required to run a model from the Hugging Face Hub or a local path.";

const FLAGS = new Map(OPTIONS.flatMap((option) => option.flags.map((flag) => [flag, option] as const)));

interface Args {
  model: string;
  revision: string | null;
  file: string | null;
  format: string | null;
  max_model_len: number | null;
  batch_size: number;
  kv_cache_dtype: string;
  gpu_memory: string | null;
  gpu_memory_utilization: number;
  max_num_batched_tokens: number | null;
  dtype: string | null;
  hf_token: string | null;
  color: boolean;
  output: string;
}

class Exit {
  constructor(
    public code: number,
    public message = "",
  ) {}
}

class ArgumentError {
  constructor(public message: string) {}
}

function repr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const body = value.replace(/[\\\n\r\t\x00-\x1f\x7f]/g, (c) => {
    if (c === "\\") return "\\\\";
    if (c === "\n") return "\\n";
    if (c === "\r") return "\\r";
    if (c === "\t") return "\\t";
    return `\\x${c.charCodeAt(0).toString(16).padStart(2, "0")}`;
  });
  return quote + (quote === "'" ? body.replaceAll("'", "\\'") : body) + quote;
}

function metavar(option: Option): string {
  return option.choices ? `{${option.choices.join(",")}}` : option.dest.toUpperCase();
}

function terminalWidth(): number {
  const columns = Number.parseInt(process.env.COLUMNS ?? "", 10);
  if (columns > 0) return columns;
  return process.stdout.isTTY && process.stdout.columns > 0 ? process.stdout.columns : 80;
}

const WS = "[\\t\\n\\x0b\\x0c\\r ]";
const NWS = "[^\\t\\n\\x0b\\x0c\\r ]";
const WP = "[\\w!\"'&.,?]";
const LT = "[^\\d\\W]";
const WORDSEP = new RegExp(
  `(${WS}+|(?<=${WP})-{2,}(?=\\w)|${NWS}+?(?:-(?:(?<=${LT}{2}-)|(?<=${LT}-${LT}-))(?=${LT}-?${LT})|(?=${WS}|$)|(?<=${WP})(?=-{2,}\\w)))`,
);

function wrap(text: string, width: number): string[] {
  const chunks = text
    .split(WORDSEP)
    .filter(Boolean)
    .reverse();
  const lines: string[] = [];
  while (chunks.length) {
    const line: string[] = [];
    let length = 0;
    if (chunks[chunks.length - 1]!.trim() === "" && lines.length) chunks.pop();
    while (chunks.length) {
      const size = chunks[chunks.length - 1]!.length;
      if (length + size > width) break;
      line.push(chunks.pop()!);
      length += size;
    }
    if (chunks.length && chunks[chunks.length - 1]!.length > width) {
      const space = width < 1 ? 1 : width - length;
      const chunk = chunks[chunks.length - 1]!;
      let end = space;
      if (chunk.length > space) {
        const hyphen = chunk.lastIndexOf("-", space - 1);
        if (hyphen > 0 && /[^-]/.test(chunk.slice(0, hyphen))) end = hyphen + 1;
      }
      line.push(chunk.slice(0, end));
      chunks[chunks.length - 1] = chunk.slice(end);
    }
    if (line.length && line[line.length - 1]!.trim() === "") line.pop();
    if (line.length) lines.push(line.join(""));
  }
  return lines;
}

function formatUsage(width: number): string {
  const prefix = "usage: ";
  const optParts = OPTIONS.map((o) => (o.action === "store" ? `[${o.flags[0]} ${metavar(o)}]` : `[${o.flags[0]}]`));
  const posParts = ["model"];
  let usage = [PROG, ...optParts, ...posParts].join(" ");
  if (prefix.length + usage.length > width) {
    const getLines = (parts: string[], indent: string, first: string | null = null): string[] => {
      const lines: string[] = [];
      let line: string[] = [];
      let length = first !== null ? first.length - 1 : indent.length - 1;
      for (const part of parts) {
        if (length + 1 + part.length > width && line.length) {
          lines.push(indent + line.join(" "));
          line = [];
          length = indent.length - 1;
        }
        line.push(part);
        length += part.length + 1;
      }
      if (line.length) lines.push(indent + line.join(" "));
      if (first !== null) lines[0] = lines[0]!.slice(indent.length);
      return lines;
    };
    let lines: string[];
    if (prefix.length + PROG.length <= 0.75 * width) {
      const indent = " ".repeat(prefix.length + PROG.length + 1);
      lines = [...getLines([PROG, ...optParts], indent, prefix), ...getLines(posParts, indent)];
    } else {
      const indent = " ".repeat(prefix.length);
      lines = getLines([...optParts, ...posParts], indent);
      if (lines.length > 1) lines = [...getLines(optParts, indent), ...getLines(posParts, indent)];
      lines = [PROG, ...lines];
    }
    usage = lines.join("\n");
  }
  return `${prefix}${usage}\n`;
}

function formatHelp(): string {
  const width = terminalWidth() - 2;
  const invocation = (o: Option): string =>
    o.action === "store" ? `${o.flags.join(", ")} ${metavar(o)}` : o.flags.join(", ");
  const items: [string, string][] = [["model", MODEL_HELP]];
  const options: [string, string][] = OPTIONS.map((o) => [invocation(o), o.help]);
  const longest = Math.max(...[...items, ...options].map(([header]) => header.length + 2));
  const position = Math.min(longest + 2, Math.min(24, Math.max(width - 20, 4)));
  const helpWidth = Math.max(width - position, 11);
  const format = ([header, help]: [string, string]): string => {
    const lines = wrap(help.replace(/\s+/g, " ").trim(), helpWidth);
    const first = header.length <= position - 4 ? `  ${header.padEnd(position - 4)}  ` : `  ${header}\n${" ".repeat(position)}`;
    return [first + lines[0], ...lines.slice(1).map((line) => " ".repeat(position) + line)].join("\n") + "\n";
  };
  const description = wrap(DESCRIPTION.replace(/\s+/g, " ").trim(), Math.max(width, 11)).join("\n");
  return [
    formatUsage(width),
    description + "\n",
    "positional arguments:\n" + items.map(format).join(""),
    "options:\n" + options.map(format).join(""),
  ].join("\n");
}

function parseValue(option: Option, value: string): string | number {
  let result: string | number = value;
  if (option.type === "int") {
    if (!/^\s*[+-]?\d+(?:_\d+)*\s*$/.test(value)) {
      throw new ArgumentError(`argument ${option.flags.join("/")}: invalid int value: ${repr(value)}`);
    }
    result = Number(value.trim().replaceAll("_", ""));
  } else if (option.type === "float") {
    const digits = "\\d+(?:_\\d+)*";
    const number = new RegExp(
      `^\\s*[+-]?(?:(?:${digits}(?:\\.(?:${digits})?)?|\\.${digits})(?:[eE][+-]?${digits})?|inf(?:inity)?|nan)\\s*$`,
      "i",
    );
    if (!number.test(value)) {
      throw new ArgumentError(`argument ${option.flags.join("/")}: invalid float value: ${repr(value)}`);
    }
    const text = value.trim().replaceAll("_", "").toLowerCase();
    const sign = text.startsWith("-") ? -1 : 1;
    const bare = text.replace(/^[+-]/, "");
    result = bare === "nan" ? Number.NaN : bare.startsWith("inf") ? sign * Number.POSITIVE_INFINITY : Number(text);
  }
  if (option.choices && !option.choices.includes(String(result))) {
    throw new ArgumentError(
      `argument ${option.flags.join("/")}: invalid choice: ${repr(String(result))} (choose from ${option.choices.join(", ")})`,
    );
  }
  return result;
}

type Token =
  | { kind: "A"; arg: string }
  | { kind: "-" }
  | { kind: "O"; arg: string; option: Option | null; flag: string; sep: string | null; explicit: string | null };

function classify(arg: string): Token {
  if (!arg || arg[0] !== "-") return { kind: "A", arg };
  const exact = FLAGS.get(arg);
  if (exact) return { kind: "O", arg, option: exact, flag: arg, sep: null, explicit: null };
  if (arg.length === 1) return { kind: "A", arg };
  const eq = arg.indexOf("=");
  if (eq >= 0) {
    const option = FLAGS.get(arg.slice(0, eq));
    if (option) return { kind: "O", arg, option, flag: arg.slice(0, eq), sep: "=", explicit: arg.slice(eq + 1) };
  }
  const prefix = eq >= 0 ? arg.slice(0, eq) : arg;
  const sep = eq >= 0 ? "=" : null;
  const explicit = eq >= 0 ? arg.slice(eq + 1) : null;
  const matches: Token[] = [];
  for (const [flag, option] of FLAGS) {
    if (arg[1] !== "-" && flag === arg.slice(0, 2)) {
      matches.push({ kind: "O", arg, option, flag, sep: "", explicit: arg.slice(2) });
    } else if (flag.startsWith(prefix)) {
      matches.push({ kind: "O", arg, option, flag, sep, explicit });
    }
  }
  if (matches.length > 1) {
    const options = matches.map((m) => (m.kind === "O" ? m.flag : "")).join(", ");
    throw new ArgumentError(`ambiguous option: ${arg} could match ${options}`);
  }
  if (matches.length === 1) return matches[0]!;
  if (/^-\d+$|^-\d*\.\d+$/.test(arg)) return { kind: "A", arg };
  if (arg.includes(" ")) return { kind: "A", arg };
  return { kind: "O", arg, option: null, flag: arg, sep: null, explicit: null };
}

function parseArgs(argv: string[]): Args {
  const tokens: Token[] = [];
  let rest = false;
  for (const arg of argv) {
    if (rest) {
      tokens.push({ kind: "A", arg });
    } else if (arg === "--") {
      rest = true;
      tokens.push({ kind: "-" });
    } else {
      tokens.push(classify(arg));
    }
  }

  const values: Record<string, string | number | boolean | null> = {};
  for (const option of OPTIONS) if (option.action !== "help" && option.action !== "version") values[option.dest] = option.default ?? null;
  let model: string | null = null;
  const extras: string[] = [];

  const take = (option: Option, value: string | null): void => {
    if (option.action === "help") throw new Exit(0, formatHelp());
    if (option.action === "version") throw new Exit(0, `${PROG} ${VERSION}\n`);
    if (option.action === "store_true") values[option.dest] = true;
    else values[option.dest] = parseValue(option, value!);
  };

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]!;
    if (token.kind === "-") continue;
    if (token.kind === "A") {
      if (model === null) model = token.arg;
      else extras.push(token.arg);
      continue;
    }
    let { option, flag, sep, explicit } = token;
    if (option === null) {
      extras.push(token.arg);
      continue;
    }
    const actions: [Option, string | null][] = [];
    for (;;) {
      const name = option.flags.join("/");
      if (explicit === null) {
        if (option.action === "store") {
          const next = tokens[i + 1];
          if (next?.kind !== "A") throw new ArgumentError(`argument ${name}: expected one argument`);
          actions.push([option, next.arg]);
          i += 1;
        } else {
          actions.push([option, null]);
        }
        break;
      }
      if (option.action === "store") {
        actions.push([option, explicit]);
        break;
      }
      if (flag[1] === "-" || explicit === "") {
        throw new ArgumentError(`argument ${name}: ignored explicit argument ${repr(explicit)}`);
      }
      if (sep || explicit[0] === "-") {
        throw new ArgumentError(`argument ${name}: ignored explicit argument ${repr(explicit)}`);
      }
      actions.push([option, null]);
      flag = "-" + explicit[0];
      const chained = FLAGS.get(flag);
      if (chained === undefined) {
        extras.push("-" + explicit);
        break;
      }
      option = chained;
      explicit = explicit.slice(1);
      if (!explicit) {
        sep = explicit = null;
      } else if (explicit[0] === "=") {
        sep = "=";
        explicit = explicit.slice(1);
      } else {
        sep = "";
      }
    }
    for (const [action, value] of actions) take(action, value);
  }
  if (model === null) throw new ArgumentError("the following arguments are required: model");
  if (extras.length) throw new ArgumentError(`unrecognized arguments: ${extras.join(" ")}`);
  return { ...values, model } as unknown as Args;
}

function usageError(message: string): Exit {
  return new Exit(2, `${formatUsage(terminalWidth() - 2)}${PROG}: error: ${message}\n`);
}

function isRuntimeError(error: unknown): error is Error {
  return error instanceof HubError || (error instanceof Error && error.constructor === Error);
}

export async function main(
  argv: string[] = process.argv.slice(2),
  streams: Streams = { stdout: process.stdout, stderr: process.stderr },
): Promise<number> {
  const { stdout, stderr } = streams;
  let args: Args;
  try {
    args = parseArgs(argv);
    if (
      args.batch_size < 1 ||
      [args.max_model_len, args.max_num_batched_tokens].some((v) => v !== null && v < 1)
    ) {
      throw new ArgumentError("`--batch-size`, `--max-model-len` and `--max-num-batched-tokens` must be positive");
    }
    if (!(0 < args.gpu_memory_utilization && args.gpu_memory_utilization <= 1)) {
      throw new ArgumentError("`--gpu-memory-utilization` must be in (0, 1]");
    }
  } catch (error) {
    const exit = error instanceof ArgumentError ? usageError(error.message) : error;
    if (!(exit instanceof Exit)) throw exit;
    (exit.code ? stderr : stdout).write(exit.message);
    return exit.code;
  }

  const status = new Status(stderr);
  const options: RunOptions = {
    batchSize: args.batch_size,
    kvCacheDtype: args.kv_cache_dtype,
    gpuMemoryUtilization: args.gpu_memory_utilization,
    progress: new Progress(status.update),
  };
  if (args.revision !== null) options.revision = args.revision;
  if (args.file !== null) options.file = args.file;
  if (args.max_model_len !== null) options.maxModelLen = args.max_model_len;
  if (args.hf_token !== null) options.hfToken = args.hf_token;
  if (args.gpu_memory !== null) options.gpuMemory = args.gpu_memory;
  if (args.max_num_batched_tokens !== null) options.maxNumBatchedTokens = args.max_num_batched_tokens;
  if (args.format !== null) options.format = args.format as NonNullable<RunOptions["format"]>;
  if (args.dtype !== null) options.dtype = args.dtype as NonNullable<RunOptions["dtype"]>;

  const interrupt = (): void => {
    status.clear();
    process.exit(130);
  };
  process.once("SIGINT", interrupt);
  let result: Estimate | Listing;
  try {
    result = await arun(args.model, options);
  } catch (error) {
    if (!isRuntimeError(error)) throw error;
    status.clear();
    stderr.write(`${PROG}: error: ${error.message}\n`);
    return 1;
  } finally {
    process.removeListener("SIGINT", interrupt);
  }
  status.clear();

  if (args.output === "json") {
    stdout.write(dumps(result.toDict()) + "\n");
    return 0;
  }
  for (const warning of result.warnings) stderr.write(`${PROG}: warning: ${warning}\n`);
  stdout.write((args.output === "markdown" ? markdown(result) : table(result, args.color)) + "\n");
  return 0;
}

function isEntry(): boolean {
  const entry = process.argv[1];
  try {
    return entry !== undefined && realpathSync(entry) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

if (isEntry()) process.exitCode = await main();
