#!/usr/bin/env node
import { estimateModelMemory } from "./index.js";
import { formatResult } from "./report.js";

const VERSION = "0.1.0";

function usage(): string {
  return `hf-mem-ts ${VERSION}

Estimate resident model weights and optional cache from remote file metadata.

Usage:
  hf-mem-ts <owner/model> [options]
  hf-mem-ts --model-id <owner/model> [options]

Options:
  -m, --model-id <id>       Hugging Face model repository
  -r, --revision <revision> Git revision (default: main)
      --gguf-file <path>    Select one GGUF quantization (shards are combined)
      --mmproj-file <path>  Select a multimodal projector (auto: sole/F16)
      --no-mmproj           Exclude multimodal projector memory
      --draft-model <id>    Add a speculative-decoding draft model
      --draft-revision <r>  Draft-model revision (default: main)
      --draft-gguf-file <p> Select the draft model's GGUF file
      --kv-cache            Estimate KV-cache memory
      --max-model-len <n>   Override context length
      --batch-size <n>      Batch size (default: 1)
      --concurrency <n>     Parallel metadata requests per model (default: 8)
      --kv-cache-dtype <d>  KV dtype (default: auto; GGUF auto is F16)
      --token <token>       Hugging Face token (or use HF_TOKEN)
      --json                Print machine-readable JSON
  -h, --help                Show help
  -v, --version             Show version`;
}

interface Args {
  modelId?: string;
  revision?: string;
  ggufFile?: string;
  mmprojFile?: string | false;
  draftModel?: string;
  draftRevision?: string;
  draftGgufFile?: string;
  kvCache?: boolean;
  maxModelLen?: number;
  batchSize?: number;
  concurrency?: number;
  kvCacheDtype?: string;
  token?: string;
  json?: boolean;
  help?: boolean;
  version?: boolean;
}

function parseInteger(value: string | undefined, flag: string): number {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number <= 0) throw new Error(`${flag} requires a positive integer.`);
  return number;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {};
  const take = (index: number, flag: string): string => {
    const value = argv[index + 1];
    if (!value || value.startsWith("-")) throw new Error(`${flag} requires a value.`);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const value = argv[i]!;
    switch (value) {
      case "-m": case "--model-id": args.modelId = take(i++, value); break;
      case "-r": case "--revision": args.revision = take(i++, value); break;
      case "--gguf-file": args.ggufFile = take(i++, value); break;
      case "--mmproj-file": args.mmprojFile = take(i++, value); break;
      case "--no-mmproj": args.mmprojFile = false; break;
      case "--draft-model": args.draftModel = take(i++, value); break;
      case "--draft-revision": args.draftRevision = take(i++, value); break;
      case "--draft-gguf-file": args.draftGgufFile = take(i++, value); break;
      case "--kv-cache": args.kvCache = true; break;
      case "--max-model-len": args.maxModelLen = parseInteger(take(i++, value), value); break;
      case "--batch-size": args.batchSize = parseInteger(take(i++, value), value); break;
      case "--concurrency": args.concurrency = parseInteger(take(i++, value), value); break;
      case "--kv-cache-dtype": args.kvCacheDtype = take(i++, value); break;
      case "--token": args.token = take(i++, value); break;
      case "--json": args.json = true; break;
      case "-h": case "--help": args.help = true; break;
      case "-v": case "--version": args.version = true; break;
      default:
        if (value.startsWith("-")) throw new Error(`Unknown option: ${value}`);
        if (args.modelId) throw new Error(`Unexpected argument: ${value}`);
        args.modelId = value;
    }
  }
  return args;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(usage()); return; }
  if (args.version) { console.log(VERSION); return; }
  if (!args.modelId) throw new Error(`Missing model ID.\n\n${usage()}`);
  const token = args.token ?? process.env.HF_TOKEN;
  const draftModel = args.draftModel ? {
    modelId: args.draftModel,
    ...(args.draftRevision ? { revision: args.draftRevision } : {}),
    ...(args.draftGgufFile ? { ggufFile: args.draftGgufFile } : {}),
  } : undefined;
  const result = await estimateModelMemory({
    modelId: args.modelId,
    ...(args.revision ? { revision: args.revision } : {}),
    ...(args.ggufFile ? { ggufFile: args.ggufFile } : {}),
    ...(args.mmprojFile !== undefined ? { mmprojFile: args.mmprojFile } : {}),
    ...(draftModel ? { draftModel } : {}),
    ...(args.kvCache ? { kvCache: true } : {}),
    ...(args.maxModelLen ? { maxModelLen: args.maxModelLen } : {}),
    ...(args.batchSize ? { batchSize: args.batchSize } : {}),
    ...(args.concurrency ? { concurrency: args.concurrency } : {}),
    ...(args.kvCacheDtype ? { kvCacheDtype: args.kvCacheDtype } : {}),
    ...(token ? { token } : {}),
  });
  if (args.json) console.log(JSON.stringify(result, null, 2));
  else console.log(formatResult(result));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? `hf-mem-ts: ${error.message}` : error);
  process.exitCode = 1;
});
