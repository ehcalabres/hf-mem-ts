import { Estimate, type KVCache, Listing } from "./result.js";
import { fixed, int, round, str } from "./py.js";
import { VERSION } from "./version.js";
import type { Weights } from "./weights.js";

const NAME_WIDTH = 15;
const MAX_FILE_WIDTH = 64;
const MAX_TITLE_WIDTH = 72;

type Item = [string, string, string, number];
type Section = [string | null, Item[]];

interface Process {
  env?: Record<string, string | undefined>;
  stdout?: { isTTY?: boolean };
  cwd?: () => string;
}

const PROCESS: Process | undefined = (globalThis as { process?: Process }).process;

const len = (text: string): number => Array.from(text).length;

const repeat = (text: string, n: number): string => (n > 0 ? text.repeat(n) : "");

const ljust = (text: string, width: number): string => text + repeat(" ", width - len(text));

function center(text: string, width: number): string {
  const pad = width - len(text);
  if (pad <= 0) return text;
  const left = Math.floor(pad / 2);
  return repeat(" ", left) + text + repeat(" ", pad - left);
}

export function gib(n: number): string {
  return fixed(n / 1024 ** 3, 2);
}

export function short(n: number): string {
  for (const unit of ["", "K", "M", "B", "T"]) {
    if (Math.abs(n) < 1000) return unit === "" ? `${int(n)}` : `${fixed(n, 2)}${unit}`;
    n /= 1000;
  }
  return `${fixed(n, 2)}P`;
}

export function bar(fraction: number, width: number): string {
  const eighths = round(Math.min(Math.max(fraction, 0.0), 1.0) * width * 8);
  const value = fraction > 0 ? Math.max(eighths, 1) : eighths;
  const full = Math.floor(value / 8);
  const rest = value - full * 8;
  const text = repeat("█", full) + (rest ? "▏▎▍▌▋▊▉"[rest - 1]! : "");
  return text + repeat("░", width - len(text));
}

function fit(name: string, width: number): string {
  return len(name) > width ? Array.from(name).slice(0, width - 3).join("") + "..." : ljust(name, width);
}

function draw(titles: string[], sections: Section[], nameWidth: number): string[] {
  const texts = sections.flatMap(([, items]) => items.map(([, t]) => t));
  const headings = [...titles, ...sections.flatMap(([s]) => (s ? [s] : []))];
  const inner = Math.max(
    Math.max(...headings.map(len)) + 8,
    nameWidth + Math.max(...texts.map(len)) + 7,
  );
  const data = inner - nameWidth - 5;
  const badge = ` hf-mem-ts v${VERSION} `;
  const lines: string[] = ["┌" + repeat("┬", inner - len(badge)) + badge + "┐", "├" + repeat("┴", inner) + "┤"];
  lines.push(...titles.map((t) => `│${center(t, inner)}│`));

  const divider = (left: string, mid: string, right: string): string =>
    left + repeat("─", nameWidth + 2) + mid + repeat("─", data + 2) + right;

  sections.forEach(([title, items], index) => {
    if (title !== null) {
      if (index) {
        lines.push(divider("├", "┴", "┤"));
      } else {
        lines.push("├" + repeat("─", inner) + "┤");
      }
      lines.push(`│${center(title, inner)}│`);
      lines.push(divider("├", "┬", "┤"));
    } else if (index === 0) {
      lines.push(divider("├", "┬", "┤"));
    } else {
      lines.push(divider("├", "┼", "┤"));
    }
    items.forEach(([top, text, bottom, fraction], i) => {
      if (i) lines.push(divider("├", "┼", "┤"));
      lines.push(`│ ${fit(top, nameWidth)} │ ${ljust(text, data)} │`);
      lines.push(`│ ${fit(bottom, nameWidth)} │ ${bar(fraction, data)} │`);
    });
  });
  lines.push(divider("└", "┴", "┘"));
  return lines;
}

function shorten(text: string, width = MAX_TITLE_WIDTH): string {
  const chars = Array.from(text);
  return chars.length <= width ? text : "..." + chars.slice(3 - width).join("");
}

function normpath(path: string): string {
  const parts: string[] = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") parts.pop();
    else parts.push(part);
  }
  return "/" + parts.join("/");
}

function relpath(path: string, start: string): string {
  const target = normpath(path).split("/").filter(Boolean);
  const base = normpath(start).split("/").filter(Boolean);
  let i = 0;
  while (i < target.length && i < base.length && target[i] === base[i]) i += 1;
  const parts = [...Array<string>(base.length - i).fill(".."), ...target.slice(i)];
  return parts.length ? parts.join("/") : ".";
}

export function where(url: string, revision: string | null): string {
  if (url.startsWith("/")) {
    const env = PROCESS?.env?.HOME;
    const home = env === undefined ? null : env.replace(/\/+$/, "") || "/";
    const relative = PROCESS?.cwd ? relpath(url, PROCESS.cwd()) : "..";
    if (!relative.startsWith("..")) {
      url = relative;
    } else if (home !== null && url.startsWith(home + "/")) {
      url = "~" + url.slice(home.length);
    }
  }
  if (revision && /^[0-9a-f]{40}$/.test(revision)) revision = revision.slice(0, 7);
  return shorten(revision ? `${url} @ ${revision}` : url);
}

function titlesFor(url: string, revision: string | null, file: string | null, kv: KVCache | null): string[] {
  const titles = ["INFERENCE MEMORY ESTIMATE FOR", where(url, revision)];
  if (file) titles.push(shorten(`FOR \`${file}\``));
  if (kv !== null) titles.push(`w/ max-model-len=${kv.maxModelLen}, batch-size=${kv.batchSize}`);
  return titles;
}

function dtypeItems(weights: Weights, total: number): Item[] {
  const sorted = weights.sortedDtypes();
  const width = Math.max(...sorted.map(([, w]) => len(`${short(w.params)} PARAMS`)));
  return sorted.map(([dtype, w]) => [
    ljust(dtype, width),
    `${gib(w.bytes)} / ${gib(total)} GiB`,
    `${short(w.params)} PARAMS`,
    w.bytes / total,
  ]);
}

export function table(estimate: Estimate | Listing, color = false): string {
  if (estimate instanceof Listing) return listingTable(estimate, color);
  const total = estimate.total || 1;
  const kv = estimate.kvCache;
  const weights = estimate.weights;
  const offloaded = estimate.offloaded;
  const allParams = weights.params + (offloaded ? offloaded.params : 0);
  const params = weights.params ? `${short(allParams)} PARAMS` : "FROM FILE SIZES";
  const parts = [params, ...(kv ? ["KV CACHE"] : []), ...(estimate.overhead ? ["OVERHEAD"] : [])];
  const summary = `${gib(estimate.total)} GiB (${parts.join(" + ")})`;
  const first: Item[] = [["TOTAL MEMORY", summary, "REQUIREMENTS", 1.0]];
  const sections: Section[] = [[null, first]];
  if (offloaded) {
    const cpu = `${gib(offloaded.bytes)} GiB (${short(offloaded.params)} PARAMS OFFLOADED, NOT ON GPU)`;
    first.push(["CPU MEMORY", cpu, "HOST MEMORY", offloaded.bytes / (estimate.total + offloaded.bytes)]);
  }

  const heading = (name: string, w: Weights): string =>
    w.params
      ? `${name.toUpperCase()} (${short(w.params)} PARAMS, ${gib(w.bytes)} GiB)`
      : `${name.toUpperCase()} (${gib(w.bytes)} GiB)`;

  if (estimate.moe !== null) {
    const moe = estimate.moe;
    sections.push([heading("MODEL", moe.base), dtypeItems(moe.base, total)]);
    const experts = `EXPERTS (${moe.experts} x ${short(moe.expert.params)} PARAMS, ${gib(moe.expertsTotal.bytes)} GiB)`;
    sections.push([experts, dtypeItems(moe.expertsTotal, total)]);
  } else {
    const many = estimate.components.size > 1;
    const components = estimate.components.size ? estimate.components : new Map([["MODEL", weights]]);
    for (const [name, component] of components) {
      const title = heading(many ? name : "MODEL", component);
      if (component.dtypes.size) {
        sections.push([title, dtypeItems(component, total)]);
      } else if (many || kv || estimate.overhead) {
        sections.push([
          title,
          [["FILE SIZE", `${gib(component.bytes)} / ${gib(total)} GiB`, "", component.bytes / total]],
        ]);
      }
    }
  }
  if (kv !== null) {
    const tokens = kv.maxModelLen * kv.batchSize;
    const title = `KV CACHE (${tokens} TOKENS, ${gib(kv.bytes)} GiB)`;
    sections.push([title, [[kv.dtype, `${gib(kv.bytes)} / ${gib(total)} GiB`, `${tokens} TOKENS`, kv.bytes / total]]]);
  }
  const overhead = estimate.overhead;
  if (overhead) {
    const entries: [string, number][] = [
      ["ACTIVATIONS", overhead.activations],
      ["ENCODER", overhead.encoder],
      ["GRAPHS", overhead.graphs],
      ["RUNTIME", overhead.runtime],
      ["BUFFERS", overhead.buffers],
      ["LOADING", overhead.loading],
    ];
    const items: Item[] = entries
      .filter(([, value]) => value)
      .map(([name, value]) => [name, `${gib(value)} / ${gib(total)} GiB`, "", value / total]);
    sections.push([`OVERHEAD (${gib(overhead.bytes)} GiB)`, items]);
  }
  const fitted = estimate.fit;
  if (fitted) {
    const gpus = fitted.gpus || 1;
    const capacity = gpus * fitted.usable;
    let text = `${gib(fitted.required)} / ${gib(capacity)} GiB [${fixed((fitted.required / capacity) * 100, 0)}%] ON ${gpus} x ${gib(fitted.gpuMemory)} GiB`;
    if (kv && fitted.maxModelLen) text += `, UP TO ${fitted.maxModelLen} TOKENS`;
    let verdict: string;
    if (fitted.fits && fitted.needed && fitted.needed < gpus) {
      verdict = `FITS ON ${fitted.needed} GPU` + (fitted.needed > 1 ? "S" : "");
    } else {
      verdict = fitted.fits
        ? "FITS"
        : fitted.needed && fitted.needed > gpus
          ? `⚠ NEEDS ${fitted.needed} GPUS`
          : "⚠ DOES NOT FIT";
    }
    first.push(["GPU MEMORY", text, verdict, fitted.required / capacity]);
  }
  return colorize(draw(titlesFor(estimate.url, estimate.revision, estimate.file, kv), sections, NAME_WIDTH), color);
}

function listingTable(listing: Listing, color = false): string {
  const kv = listing.files.find((f) => f.kvCache)?.kvCache ?? null;
  const largest = Math.max(...listing.files.map((f) => f.total)) || 1;
  const nameWidth = Math.min(Math.max(...listing.files.map((f) => len(f.file || ""))), MAX_FILE_WIDTH);
  const items: Item[] = [];
  for (const f of listing.files) {
    let text = `${gib(f.total)} GiB`;
    if (f.kvCache) {
      const parts = [`${gib(f.weights.bytes)} GiB WEIGHTS`, `${gib(f.kvCache.bytes)} GiB KV`];
      if (f.overhead) parts.push(`${gib(f.overhead.bytes)} GiB OVERHEAD`);
      text += ` (${parts.join(" + ")})`;
    }
    items.push(["", text, f.file || "", f.total / largest]);
  }
  const titles = titlesFor(listing.url, listing.revision, null, kv);
  return colorize(draw(titles, [["FILES", items]], nameWidth), color);
}

function colorize(lines: string[], enabled = false): string {
  if (enabled && PROCESS?.stdout?.isTTY && !PROCESS.env?.NO_COLOR) {
    return lines.map((line) => `\x1b[38;2;244;183;63m${line}\x1b[0m`).join("\n");
  }
  return lines.join("\n");
}

export function markdown(estimate: Estimate | Listing): string {
  const at = `[\`${estimate.model}\`](${estimate.url})` + (estimate.revision ? ` @ \`${estimate.revision}\`` : "");
  if (estimate instanceof Listing) {
    const rows = ["| File | Total | Weights | KV cache |", "|---|--:|--:|--:|"];
    for (const f of estimate.files) {
      const kv = f.kvCache ? `${gib(f.kvCache.bytes)} GiB` : "-";
      rows.push(`| \`${str(f.file)}\` | ${gib(f.total)} GiB | ${gib(f.weights.bytes)} GiB | ${kv} |`);
    }
    return [`Memory estimate for ${at}`, "", ...rows].join("\n");
  }

  const rows = ["| | Memory | Params |", "|---|--:|--:|"];
  rows.push(
    `| **Total** | **${gib(estimate.total)} GiB** | ${estimate.weights.params ? short(estimate.weights.params) : "-"} |`,
  );
  const components = estimate.components.size > 1 ? estimate.components : new Map([["Weights", estimate.weights]]);
  for (const [name, component] of components) {
    const dtypes = component
      .sortedDtypes()
      .map(([d]) => d)
      .join(", ");
    const label = dtypes ? `${name} (${dtypes})` : name;
    rows.push(`| ${label} | ${gib(component.bytes)} GiB | ${component.params ? short(component.params) : "-"} |`);
  }
  const offloaded = estimate.offloaded;
  if (offloaded) rows.push(`| Offloaded to CPU memory | ${gib(offloaded.bytes)} GiB | ${short(offloaded.params)} |`);
  const cache = estimate.kvCache;
  if (cache) {
    rows.push(
      `| KV cache (${cache.dtype}, ${cache.maxModelLen} tokens x ${cache.batchSize}) | ${gib(cache.bytes)} GiB | |`,
    );
  }
  const overhead = estimate.overhead;
  if (overhead) {
    const parts = Object.entries(overhead.toDict())
      .filter(([n, v]) => v && n !== "bytes")
      .map(([n]) => n)
      .join(", ");
    rows.push(`| Overhead (${parts}) | ${gib(overhead.bytes)} GiB | |`);
  }
  const title = `Memory estimate for ${at}` + (estimate.file ? `, file \`${estimate.file}\`` : "");
  const lines = [title, "", ...rows];
  const fitted = estimate.fit;
  if (fitted) {
    const gpus = `${fitted.gpus || 1} x ${gib(fitted.gpuMemory)} GiB`;
    let verdict = fitted.fits ? `Fits on ${gpus}` : `Does not fit on ${gpus}`;
    if (fitted.needed && fitted.needed !== (fitted.gpus || 1)) {
      const action = !fitted.fits && fitted.needed < (fitted.gpus || 1) ? "fits on" : "needs";
      verdict += `, ${action} ${fitted.needed} GPU${fitted.needed > 1 ? "s" : ""}`;
    }
    if (estimate.kvCache && fitted.maxModelLen) verdict += `, up to ${fitted.maxModelLen} tokens`;
    lines.push("", verdict + ".");
  }
  return lines.join("\n");
}
