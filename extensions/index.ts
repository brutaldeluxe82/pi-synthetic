/**
 * Synthetic.new Provider Extension
 *
 * Provides access to open-source LLMs via Synthetic.new's OpenAI-compatible API.
 * https://synthetic.new
 *
 * Usage:
 *   export SYNTHETIC_API_KEY=your-key-here
 *   pi -e ~/.config/pi/extensions/synthetic-new
 *
 * Then use /model to select synthetic/model-name,
 * /synthetic:refresh to refresh the model catalog, or
 * /synthetic:quotas to check your usage.
 */

import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext, AgentToolResult, ProviderModelConfig } from "@mariozechner/pi-coding-agent";
import { existsSync, readFileSync, statSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

// =============================================================================
// Constants
// =============================================================================

const SYNTHETIC_BASE_URL = "https://api.synthetic.new/openai/v1";
const SYNTHETIC_MODELS_URL = `${SYNTHETIC_BASE_URL}/models`;
const SYNTHETIC_QUOTAS_URL = "https://api.synthetic.new/v2/quotas";
const SYNTHETIC_SEARCH_URL = "https://api.synthetic.new/v2/search";
const CACHE_FILE = fileURLToPath(new URL("./models-cache.json", import.meta.url));
const FETCH_TIMEOUT_MS = 12_000;
const FETCH_RETRIES = 2;
const AUTO_REFRESH_MAX_AGE_MS = 6 * 60 * 60 * 1000; // 6 hours

// Search tool name
const SYNTHETIC_WEB_SEARCH_TOOL = "synthetic_web_search";

// Quota display constants
const WEEKLY_REGEN_INTERVAL_MS = 3 * 60 * 60 * 1000; // 2% every 3 hours
const ROLLING_REGEN_INTERVAL_MS = 3 * 60 * 1000; // 5% every 3 minutes
const ANSI_BLUE = "\x1b[34m";
const ANSI_GREEN = "\x1b[32m";
const ANSI_DIM = "\x1b[2m";
const ANSI_BOLD = "\x1b[1m";
const ANSI_RESET = "\x1b[0m";

// =============================================================================
// Types
// =============================================================================

// Maps Synthetic's reasoning effort levels to pi's internal levels.
// Synthetic exposes minimal/low/medium/high/xhigh; pi uses low/medium/high.
const SYNTHETIC_REASONING_EFFORT_MAP = {
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "high",
} as const;

// Per-model compat overrides for models with non-standard behaviour.
// These are merged into the compat block when present.
const MODEL_COMPAT_OVERRIDES: Record<string, Partial<NonNullable<ProviderModelConfig["compat"]>>> = {
  "hf:MiniMaxAI/MiniMax-M2.5": {
    maxTokensField: "max_completion_tokens",
  },
};

interface SyntheticModel {
  id?: string;
  name?: string;
  provider?: string;
  always_on?: boolean;
  hugging_face_id?: string;
  input_modalities?: string[];
  output_modalities?: string[];
  context_length?: number;
  max_output_length?: number;
  pricing?: {
    prompt?: string;
    completion?: string;
    image?: string;
    request?: string;
    input_cache_reads?: string;
    input_cache_writes?: string;
  };
  supported_sampling_parameters?: string[];
  supported_features?: string[];
  quantization?: string;
  created?: number;
  openrouter?: { slug?: string };
  datacenters?: { country_code?: string }[];
}

interface SyntheticModelsResponse {
  data?: unknown;
}

// Search types
interface SyntheticSearchResult {
  url: string;
  title: string;
  text: string;
  published: string;
}

interface SyntheticSearchResponse {
  results: SyntheticSearchResult[];
}

// Quota types
type QuotasErrorKind = "cancelled" | "timeout" | "config" | "http" | "network";

type QuotasResult =
  | { success: true; data: { quotas: QuotasResponse } }
  | { success: false; error: { message: string; kind: QuotasErrorKind } };

interface QuotasResponse {
  subscription?: {
    limit: number;
    requests: number;
    renewsAt: string;
  };
  search?: {
    hourly?: {
      limit: number;
      requests: number;
      renewsAt: string;
    };
  };
  freeToolCalls?: {
    limit: number;
    requests: number;
    renewsAt: string;
  };
  weeklyTokenLimit?: {
    nextRegenAt: string;
    percentRemaining: number;
    maxCredits: string;
    remainingCredits: string;
    nextRegenCredits: string;
  };
  rollingFiveHourLimit?: {
    nextTickAt: string;
    tickPercent: number;
    remaining: number;
    max: number;
    limited: boolean;
  };
}

interface ModelsCache {
  version: 1;
  updatedAt: string;
  source: string;
  modelCount: number;
  models: ProviderModelConfig[];
}

interface RefreshResult {
  models: ProviderModelConfig[];
  rawCount: number;
  eligibleCount: number;
  skippedCount: number;
  cacheWarning?: string;
}

interface ModelDiffSummary {
  added: ProviderModelConfig[];
  removed: ProviderModelConfig[];
  changed: Array<{
    before: ProviderModelConfig;
    after: ProviderModelConfig;
    fields: string[];
  }>;
}

// =============================================================================
// Helpers
// =============================================================================

function isTimeoutReason(reason: unknown): boolean {
  return (
    (reason instanceof DOMException && reason.name === "TimeoutError") ||
    (reason instanceof Error && reason.name === "TimeoutError")
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function truncate(value: string, maxLength = 240): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength)}…`;
}

function normalizeError(error: unknown): Error {
  if (error instanceof Error) return error;
  return new Error(String(error));
}

// =============================================================================
// Quota Display Helpers
// =============================================================================

/** Parse a credit string that may contain commas or formatting into a number */
function parseCredits(value: string): number {
  const cleaned = value.replace(/[^0-9.eE+-]/g, "");
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : 0;
}

/** Format a duration in milliseconds as a human-readable string like "1d 3h 30m" */
function formatDuration(ms: number): string {
  if (ms <= 0) return "now";
  const totalMinutes = Math.floor(ms / (1000 * 60));
  const days = Math.floor(totalMinutes / (60 * 24));
  const hours = Math.floor((totalMinutes % (60 * 24)) / 60);
  const minutes = totalMinutes % 60;
  const parts: string[] = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  return parts.length > 0 ? parts.join(" ") : "<1m";
}

/** Render a colored progress bar using ANSI block characters */
function renderBar(ratio: number, width: number, color: string): string {
  const filled = Math.round(ratio * width);
  const empty = width - filled;
  const bar = "█".repeat(Math.max(0, filled)) + "░".repeat(Math.max(0, empty));
  return `${color}${bar}${ANSI_RESET}`;
}

function formatResetTime(resetAt: string): string {
  const date = new Date(resetAt);
  const diffMs = date.getTime() - Date.now();
  if (Number.isNaN(date.getTime())) return resetAt;
  if (diffMs <= 0) return "soon";
  return `in ${formatDuration(diffMs)}`;
}

/** Format the full quotas response into an ANSI-colored string for the TUI */
function formatSyntheticQuotas(quotas: QuotasResponse): string {
  const lines: string[] = [];
  const BAR_WIDTH = 24;

  if (quotas.weeklyTokenLimit) {
    const wt = quotas.weeklyTokenLimit;
    const remaining = parseCredits(wt.remainingCredits);
    const max = parseCredits(wt.maxCredits);
    const ratio = max > 0 ? remaining / max : 0;
    const regenCredits = parseCredits(wt.nextRegenCredits);
    const regenTimeStr = formatResetTime(wt.nextRegenAt);

    // Weekly regenerates 2% every 3 hours at a fixed cadence.
    // First regen arrives at nextRegenAt, subsequent regens every 3h.
    let fullRegenStr = "N/A";
    if (regenCredits > 0 && remaining < max) {
      const creditsNeeded = max - remaining;
      const intervalsNeeded = Math.ceil(creditsNeeded / regenCredits);
      const regenDate = new Date(wt.nextRegenAt);
      const firstIntervalMs = Math.max(0, regenDate.getTime() - Date.now());
      const fullRegenMs = firstIntervalMs + WEEKLY_REGEN_INTERVAL_MS * (intervalsNeeded - 1);
      fullRegenStr = formatDuration(fullRegenMs);
    } else if (remaining >= max) {
      fullRegenStr = "full";
    }

    lines.push(`${ANSI_BOLD}Weekly Tokens${ANSI_RESET}`);
    lines.push(
      `  ${remaining.toLocaleString()}/${max.toLocaleString()} credits  ${ANSI_DIM}(${(ratio * 100).toFixed(1)}%)${ANSI_RESET}`,
    );
    lines.push(`  ${renderBar(ratio, BAR_WIDTH, ANSI_BLUE)}`);
    lines.push(
      `  Regen +${regenCredits.toLocaleString()} ${regenTimeStr}  ${ANSI_DIM}Full: ${fullRegenStr}${ANSI_RESET}`,
    );
  }

  if (quotas.rollingFiveHourLimit) {
    const rf = quotas.rollingFiveHourLimit;
    const remainingInt = Math.round(rf.remaining);
    const maxInt = Math.round(rf.max);
    const ratio = rf.max > 0 ? rf.remaining / rf.max : 0;
    const state = rf.limited ? `${ANSI_DIM}limited${ANSI_RESET}` : `${ANSI_GREEN}available${ANSI_RESET}`;
    const tickTimeStr = formatResetTime(rf.nextTickAt);

    // Rolling 5h regenerates 5% of max every 3 minutes at a fixed cadence.
    // First regen arrives at nextTickAt, subsequent regens every 3 min.
    const regenPerTick = Math.max(1, Math.round(rf.max * 0.05));
    let fullRegenStr = "N/A";
    if (remainingInt < maxInt) {
      const needed = maxInt - remainingInt;
      const ticksNeeded = Math.ceil(needed / regenPerTick);
      const tickDate = new Date(rf.nextTickAt);
      const firstTickMs = Math.max(0, tickDate.getTime() - Date.now());
      const fullRegenMs = firstTickMs + ROLLING_REGEN_INTERVAL_MS * (ticksNeeded - 1);
      fullRegenStr = formatDuration(fullRegenMs);
    } else {
      fullRegenStr = "full";
    }

    lines.push(`${ANSI_BOLD}Rolling 5h${ANSI_RESET}`);
    lines.push(
      `  ${remainingInt}/${maxInt} requests  ${state}  ${ANSI_DIM}(tick ${(rf.tickPercent * 100).toFixed(0)}%)${ANSI_RESET}`,
    );
    lines.push(`  ${renderBar(ratio, BAR_WIDTH, ANSI_GREEN)}`);
    lines.push(
      `  Regen +${regenPerTick} ${tickTimeStr}  ${ANSI_DIM}Full: ${fullRegenStr}${ANSI_RESET}`,
    );
  }

  if (lines.length === 0) {
    lines.push(JSON.stringify(quotas, null, 2));
  }

  return lines.join("\n");
}

// =============================================================================
// Quota Fetching
// =============================================================================

async function fetchSyntheticQuotas(apiKey: string, signal?: AbortSignal): Promise<QuotasResult> {
  if (apiKey.length === 0) {
    return {
      success: false,
      error: { message: `No API key configured. Set SYNTHETIC_API_KEY or run /login synthetic.`, kind: "config" },
    };
  }

  const signals = [AbortSignal.timeout(FETCH_TIMEOUT_MS)];
  if (signal) signals.push(signal);
  const combinedSignal = AbortSignal.any(signals);

  try {
    const response = await fetch(SYNTHETIC_QUOTAS_URL, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "X-Title": "synthetic-new",
      },
      signal: combinedSignal,
    });

    if (!response.ok) {
      let message = response.statusText;
      const body = await response.text();
      if (body.length > 0) {
        try {
          const parsed = JSON.parse(body) as { error?: unknown; message?: unknown };
          if (typeof parsed.error === "string") message = parsed.error;
          else if (typeof parsed.message === "string") message = parsed.message;
          else message = body;
        } catch {
          message = body;
        }
      }
      return { success: false, error: { message, kind: "http" } };
    }

    return { success: true, data: { quotas: (await response.json()) as QuotasResponse } };
  } catch (error: unknown) {
    const aborted = combinedSignal.aborted || (error instanceof DOMException && error.name === "AbortError");
    if (aborted) {
      if (isTimeoutReason(combinedSignal.reason)) {
        return { success: false, error: { message: "Request timed out", kind: "timeout" } };
      }
      return { success: false, error: { message: "Request cancelled", kind: "cancelled" } };
    }

    const message = error instanceof Error ? error.message : "Unknown error";
    return { success: false, error: { message, kind: "network" } };
  }
}

/** Resolve the API key from auth storage or environment */
async function getSyntheticApiKey(ctx: ExtensionCommandContext): Promise<string> {
  const storedKey = await ctx.modelRegistry.authStorage.getApiKey("synthetic", { includeFallback: false });
  return storedKey ?? process.env.SYNTHETIC_API_KEY ?? "";
}

// =============================================================================
// Web Search Tool
// =============================================================================

function registerSyntheticWebSearchTool(pi: ExtensionAPI): void {
  pi.registerTool({
    name: SYNTHETIC_WEB_SEARCH_TOOL,
    label: "Synthetic: Web Search",
    description:
      "Search the web using Synthetic's zero-data-retention API. Returns search results with titles, URLs, content snippets, and publication dates. Use for finding documentation, articles, recent information, or any web content. Results are fresh and not cached by Synthetic.",
    promptSnippet: "Search the web using Synthetic's zero-data-retention API",
    promptGuidelines: [
      "Use synthetic_web_search for finding documentation, articles, recent information, or any web content.",
      "Write specific queries with names, dates, versions, or locations for synthetic_web_search.",
      "synthetic_web_search results are fresh and not cached by Synthetic.",
    ],
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query. Be specific for best results.",
        },
      },
      required: ["query"],
    },

    async execute(
      _toolCallId: string,
      params: { query: string },
      signal: AbortSignal | undefined,
      _onUpdate: unknown,
      ctx: ExtensionContext,
    ): Promise<AgentToolResult> {
      const apiKey = await resolveApiKey(ctx);
      if (!apiKey) {
        throw new Error(
          "Synthetic web search requires a Synthetic API key. Set SYNTHETIC_API_KEY or run /login synthetic.",
        );
      }

      const response = await fetch(SYNTHETIC_SEARCH_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: params.query }),
        signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Synthetic search API error: ${response.status} ${truncate(errorText)}`);
      }

      let data: SyntheticSearchResponse;
      try {
        data = await response.json();
      } catch (parseError) {
        throw new Error(
          parseError instanceof Error
            ? `Failed to parse search results: ${parseError.message}`
            : "Failed to parse search results",
        );
      }

      let content = `Found ${data.results.length} result(s) for "${params.query}":\n\n`;
      for (const result of data.results) {
        content += `## ${result.title}\n`;
        content += `URL: ${result.url}\n`;
        if (result.published) {
          content += `Published: ${result.published}\n`;
        }
        content += `\n${result.text}\n`;
        content += "\n---\n\n";
      }

      return {
        content: [{ type: "text", text: content }],
      };
    },
  });
}

/** Resolve the API key from auth storage or environment */
async function resolveApiKey(ctx: ExtensionContext): Promise<string | undefined> {
  try {
    const storedKey = await ctx.modelRegistry.authStorage.getApiKey("synthetic", { includeFallback: false });
    if (storedKey) return storedKey;
  } catch {
    // authStorage may not be available in all contexts
  }
  return process.env.SYNTHETIC_API_KEY;
}

async function handleQuotasCommand(ctx: ExtensionCommandContext): Promise<void> {
  const apiKey = await getSyntheticApiKey(ctx);
  const result = await fetchSyntheticQuotas(apiKey, ctx.signal);
  if (!result.success) {
    ctx.ui.notify(
      `Synthetic quotas failed: ${result.error.message}`,
      result.error.kind === "config" ? "warning" : "error",
    );
    return;
  }

  ctx.ui.notify(formatSyntheticQuotas(result.data.quotas), "info");
}

function formatModelLabel(model: ProviderModelConfig): string {
  return model.name.split("/").pop() || model.name;
}

function formatModelList(models: ProviderModelConfig[], limit = 5): string {
  const labels = models.slice(0, limit).map(formatModelLabel);
  const more = models.length > limit ? ` +${models.length - limit} more` : "";
  return labels.join(", ") + more;
}

/**
 * Parse price string like "$0.00000045" to dollars per million tokens.
 * Returns 0 for empty/zero/invalid values.
 */
function parsePriceToPerMillion(priceStr: string | undefined): number {
  if (!priceStr) return 0;

  const normalized = priceStr.trim();
  if (!normalized || normalized === "0") return 0;

  const match = normalized.match(/^\$?([\d.]+(?:e[+-]?\d+)?)$/i);
  if (!match) return 0;

  const perToken = Number.parseFloat(match[1]);
  if (!Number.isFinite(perToken)) return 0;

  return perToken * 1_000_000;
}

function supportsReasoning(model: SyntheticModel): boolean {
  if (model.supported_features?.includes("reasoning")) {
    return true;
  }

  const haystack = `${model.id ?? ""} ${model.name ?? ""}`;
  const reasoningPatterns = [
    /(?:^|[-/])thinking(?:[-/]|$)/i,
    /(?:^|[-/])reason(?:ing)?(?:[-/]|$)/i,
    /(?:^|[-/])r1(?:[-/]|$)/i,
    /deepseek-r1/i,
    /gpt-oss/i,
  ];

  return reasoningPatterns.some(pattern => pattern.test(haystack));
}

function supportsImages(model: SyntheticModel): boolean {
  return Array.isArray(model.input_modalities) && model.input_modalities.includes("image");
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

function isEligibleModel(model: SyntheticModel): boolean {
  return typeof model.id === "string" && model.id.startsWith("hf:") && model.always_on !== false;
}

function convertToPiModel(model: SyntheticModel): ProviderModelConfig | null {
  if (!isEligibleModel(model)) {
    return null;
  }

  const displayName =
    (typeof model.name === "string" && model.name.trim())
      ? model.name.trim()
      : model.id!.replace(/^hf:/, "");

  const contextWindow = isPositiveInteger(model.context_length) ? model.context_length : 128000;
  const maxTokens = isPositiveInteger(model.max_output_length)
    ? model.max_output_length
    : Math.min(16384, contextWindow);

  const reasoning = supportsReasoning(model);
  const hasImages = supportsImages(model);

  const config: ProviderModelConfig = {
    id: model.id!,
    name: displayName,
    reasoning,
    input: hasImages ? ["text", "image"] : ["text"],
    cost: {
      input: parsePriceToPerMillion(model.pricing?.prompt),
      output: parsePriceToPerMillion(model.pricing?.completion),
      cacheRead: parsePriceToPerMillion(model.pricing?.input_cache_reads),
      cacheWrite: parsePriceToPerMillion(model.pricing?.input_cache_writes),
    },
    contextWindow,
    maxTokens,
  };

  // Apply reasoning effort map and per-model compat overrides.
  // Models that support reasoning get the effort map so pi can translate
  // effort levels correctly; non-reasoning models still get override fields.
  const overrides = MODEL_COMPAT_OVERRIDES[model.id!];
  if (reasoning) {
    config.compat = {
      supportsReasoningEffort: true,
      reasoningEffortMap: SYNTHETIC_REASONING_EFFORT_MAP,
      ...overrides,
    };
  } else if (overrides) {
    config.compat = { ...overrides };
  }

  return config;
}

function sortAndDedupeModels(models: ProviderModelConfig[]): ProviderModelConfig[] {
  const seen = new Set<string>();
  const deduped: ProviderModelConfig[] = [];

  for (const model of models) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    deduped.push(model);
  }

  deduped.sort((a, b) => a.name.localeCompare(b.name));
  return deduped;
}

function compareModels(before: ProviderModelConfig, after: ProviderModelConfig): string[] {
  const changed: string[] = [];

  if (before.name !== after.name) changed.push("name");
  if (before.reasoning !== after.reasoning) changed.push("reasoning");
  if (before.contextWindow !== after.contextWindow) changed.push("context");
  if (before.maxTokens !== after.maxTokens) changed.push("maxTokens");
  if (before.input.join(",") !== after.input.join(",")) changed.push("input");

  if (
    before.cost.input !== after.cost.input ||
    before.cost.output !== after.cost.output ||
    before.cost.cacheRead !== after.cost.cacheRead ||
    before.cost.cacheWrite !== after.cost.cacheWrite
  ) {
    changed.push("pricing");
  }

  return changed;
}

function diffModelSets(beforeModels: ProviderModelConfig[], afterModels: ProviderModelConfig[]): ModelDiffSummary {
  const beforeById = new Map(beforeModels.map(model => [model.id, model]));
  const afterById = new Map(afterModels.map(model => [model.id, model]));

  const added = afterModels.filter(model => !beforeById.has(model.id));
  const removed = beforeModels.filter(model => !afterById.has(model.id));
  const changed = afterModels
    .map(after => {
      const before = beforeById.get(after.id);
      if (!before) return null;

      const fields = compareModels(before, after);
      return fields.length > 0 ? { before, after, fields } : null;
    })
    .filter((value): value is NonNullable<typeof value> => value !== null);

  return { added, removed, changed };
}

function isProviderModelConfig(value: unknown): value is ProviderModelConfig {
  if (!value || typeof value !== "object") return false;
  const model = value as Partial<ProviderModelConfig>;
  return (
    typeof model.id === "string" &&
    typeof model.name === "string" &&
    typeof model.reasoning === "boolean" &&
    Array.isArray(model.input) &&
    typeof model.contextWindow === "number" &&
    typeof model.maxTokens === "number" &&
    !!model.cost &&
    typeof model.cost.input === "number" &&
    typeof model.cost.output === "number" &&
    typeof model.cost.cacheRead === "number" &&
    typeof model.cost.cacheWrite === "number"
  );
}

function loadCachedModels(): ProviderModelConfig[] | null {
  if (!existsSync(CACHE_FILE)) return null;

  try {
    const raw = readFileSync(CACHE_FILE, "utf8");
    const parsed = JSON.parse(raw) as Partial<ModelsCache>;

    if (!Array.isArray(parsed.models)) {
      return null;
    }

    const models = sortAndDedupeModels(parsed.models.filter(isProviderModelConfig));
    return models.length > 0 ? models : null;
  } catch {
    return null;
  }
}

function getCacheAgeMs(): number | null {
  if (!existsSync(CACHE_FILE)) return null;

  try {
    return Date.now() - statSync(CACHE_FILE).mtimeMs;
  } catch {
    return null;
  }
}

async function saveCachedModels(models: ProviderModelConfig[]): Promise<void> {
  const payload: ModelsCache = {
    version: 1,
    updatedAt: new Date().toISOString(),
    source: SYNTHETIC_MODELS_URL,
    modelCount: models.length,
    models,
  };

  await mkdir(dirname(CACHE_FILE), { recursive: true });
  await writeFile(CACHE_FILE, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function registerSyntheticProvider(pi: ExtensionAPI, models: ProviderModelConfig[]): void {
  pi.registerProvider("synthetic", {
    baseUrl: SYNTHETIC_BASE_URL,
    apiKey: "SYNTHETIC_API_KEY",
    api: "openai-completions",
    authHeader: true,
    models,
  });
}

async function fetchModels(apiKey: string): Promise<RefreshResult> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= FETCH_RETRIES + 1; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    try {
      const response = await fetch(SYNTHETIC_MODELS_URL, {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        signal: controller.signal,
      });

      const body = await response.text();

      if (!response.ok) {
        throw new Error(`Synthetic API returned ${response.status}: ${truncate(body)}`);
      }

      let parsed: SyntheticModelsResponse;
      try {
        parsed = JSON.parse(body) as SyntheticModelsResponse;
      } catch (error) {
        throw new Error(`Synthetic API returned invalid JSON: ${normalizeError(error).message}`);
      }

      const rawModels = Array.isArray(parsed.data) ? (parsed.data as SyntheticModel[]) : [];
      if (rawModels.length === 0) {
        throw new Error("Synthetic API returned no models");
      }

      const eligibleModels = rawModels.filter(isEligibleModel);
      const convertedModels: ProviderModelConfig[] = [];

      for (const model of eligibleModels) {
        const converted = convertToPiModel(model);
        if (converted) {
          convertedModels.push(converted);
        }
      }

      const models = sortAndDedupeModels(convertedModels);
      const skippedCount = eligibleModels.length - models.length;
      if (models.length === 0) {
        throw new Error("Synthetic API returned 0 usable models after validation");
      }

      return {
        models,
        rawCount: rawModels.length,
        eligibleCount: eligibleModels.length,
        skippedCount,
      };
    } catch (error) {
      lastError = normalizeError(error);

      const isAbort = lastError.name === "AbortError";
      if (isAbort) {
        lastError = new Error(`Synthetic API request timed out after ${FETCH_TIMEOUT_MS}ms`);
      }

      if (attempt <= FETCH_RETRIES) {
        await sleep(attempt * 500);
      }
    } finally {
      clearTimeout(timeout);
    }
  }

  throw lastError ?? new Error("Unknown error fetching Synthetic models");
}

// =============================================================================
// Default Models (used before cache/refresh is available)
// =============================================================================

const DEFAULT_MODELS: ProviderModelConfig[] = [
  {
    id: "hf:moonshotai/Kimi-K2.5",
    name: "moonshotai/Kimi-K2.5",
    reasoning: true,
    input: ["text", "image"],
    cost: { input: 0.45, output: 3.4, cacheRead: 0.45, cacheWrite: 0 },
    contextWindow: 262144,
    maxTokens: 65536,
    compat: {
      supportsReasoningEffort: true,
      reasoningEffortMap: SYNTHETIC_REASONING_EFFORT_MAP,
    },
  },
  {
    id: "hf:Qwen/Qwen3-Coder-480B-A35B-Instruct",
    name: "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    reasoning: false,
    input: ["text"],
    cost: { input: 2, output: 2, cacheRead: 2, cacheWrite: 0 },
    contextWindow: 262144,
    maxTokens: 16384,
  },
  {
    id: "hf:zai-org/GLM-4.7",
    name: "zai-org/GLM-4.7",
    reasoning: true,
    input: ["text"],
    cost: { input: 0.45, output: 2.19, cacheRead: 0.45, cacheWrite: 0 },
    contextWindow: 202752,
    maxTokens: 65536,
    compat: {
      supportsReasoningEffort: true,
      reasoningEffortMap: SYNTHETIC_REASONING_EFFORT_MAP,
    },
  },
  {
    id: "hf:zai-org/GLM-5",
    name: "zai-org/GLM-5",
    reasoning: true,
    input: ["text"],
    cost: { input: 1, output: 3, cacheRead: 1, cacheWrite: 0 },
    contextWindow: 196608,
    maxTokens: 65536,
    compat: {
      supportsReasoningEffort: true,
      reasoningEffortMap: SYNTHETIC_REASONING_EFFORT_MAP,
    },
  },
  {
    id: "hf:deepseek-ai/DeepSeek-V3.2",
    name: "deepseek-ai/DeepSeek-V3.2",
    reasoning: false,
    input: ["text"],
    cost: { input: 0.56, output: 1.68, cacheRead: 0.56, cacheWrite: 0 },
    contextWindow: 162816,
    maxTokens: 16384,
  },
];

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
  let currentCachedModels = loadCachedModels();
  const startupModels = currentCachedModels ?? DEFAULT_MODELS;
  let currentModels = startupModels;
  let refreshInFlight: Promise<RefreshResult> | null = null;

  registerSyntheticProvider(pi, startupModels);
  registerSyntheticWebSearchTool(pi);

  async function refreshAndRegister(apiKey: string): Promise<RefreshResult> {
    if (!refreshInFlight) {
      refreshInFlight = (async () => {
        const result = await fetchModels(apiKey);
        registerSyntheticProvider(pi, result.models);
        currentModels = result.models;
        currentCachedModels = result.models;

        try {
          await saveCachedModels(result.models);
        } catch (error) {
          result.cacheWarning = `unable to persist model cache: ${normalizeError(error).message}`;
        }

        return result;
      })().finally(() => {
        refreshInFlight = null;
      });
    }

    return refreshInFlight;
  }

  pi.on("session_start", async (_event, ctx) => {
    const apiKey = process.env.SYNTHETIC_API_KEY;
    if (!apiKey) {
      return;
    }

    const hadCachedModels = !!currentCachedModels;
    const cacheAgeMs = getCacheAgeMs();
    const shouldAutoRefresh = !hadCachedModels || cacheAgeMs === null || cacheAgeMs > AUTO_REFRESH_MAX_AGE_MS;
    if (!shouldAutoRefresh) {
      return;
    }

    try {
      const result = await refreshAndRegister(apiKey);

      if (!hadCachedModels) {
        ctx.ui.notify(`Synthetic models loaded: ${result.models.length} available`, "info");
      }
    } catch (error) {
      if (!hadCachedModels) {
        const message = normalizeError(error).message;
        ctx.ui.notify(`Synthetic refresh failed; using built-in defaults (${message})`, "warning");
      }
    }
  });

  pi.registerCommand("synthetic:quotas", {
    description: "Show Synthetic weekly token and rolling 5h usage quotas",
    handler: async (_args, ctx) => {
      await handleQuotasCommand(ctx);
    },
  });

  pi.registerCommand("synthetic:refresh", {
    description: "Refresh Synthetic.new models from the API and persist the last good catalog",
    handler: async (_args, ctx) => {
      const apiKey = process.env.SYNTHETIC_API_KEY;

      if (!apiKey) {
        ctx.ui.notify("Error: SYNTHETIC_API_KEY environment variable not set", "error");
        return;
      }

      const previousModels = currentModels;
      ctx.ui.notify(
        `Refreshing Synthetic model catalog... current set: ${previousModels.length} models`,
        "info"
      );

      try {
        const result = await refreshAndRegister(apiKey);
        const reasoningCount = result.models.filter(model => model.reasoning).length;
        const imageCount = result.models.filter(model => model.input.includes("image")).length;
        const skipped = result.skippedCount > 0 ? `, ${result.skippedCount} skipped` : "";
        const diff = diffModelSets(previousModels, result.models);

        const lines = [
          `✓ Synthetic refresh complete`,
          `Models: ${result.models.length} total (${reasoningCount} reasoning, ${imageCount} multimodal${skipped})`,
          `Source: ${result.eligibleCount}/${result.rawCount} usable from API`,
        ];

        if (diff.added.length === 0 && diff.removed.length === 0 && diff.changed.length === 0) {
          lines.push("Changes: none");
        } else {
          const changedSummary = diff.changed.length > 0 ? `, ${diff.changed.length} changed` : "";
          lines.push(`Changes: +${diff.added.length} added, -${diff.removed.length} removed${changedSummary}`);

          if (diff.added.length > 0) {
            lines.push(`Added: ${formatModelList(diff.added)}`);
          }
          if (diff.removed.length > 0) {
            lines.push(`Removed: ${formatModelList(diff.removed)}`);
          }
          if (diff.changed.length > 0) {
            const changedPreview = diff.changed
              .slice(0, 5)
              .map(change => `${formatModelLabel(change.after)} [${change.fields.join("/")}]`)
              .join(", ");
            const more = diff.changed.length > 5 ? ` +${diff.changed.length - 5} more` : "";
            lines.push(`Changed: ${changedPreview}${more}`);
          }
        }

        lines.push(`Top models: ${formatModelList(result.models)}`);
        lines.push(
          result.cacheWarning
            ? `Cache: warning - ${result.cacheWarning}`
            : `Cache: updated ${CACHE_FILE}`
        );

        ctx.ui.notify(lines.join("\n"), diff.removed.length > 0 || !!result.cacheWarning ? "warning" : "info");
      } catch (error) {
        ctx.ui.notify(`Failed to refresh models: ${normalizeError(error).message}`, "error");
      }
    },
  });
}
