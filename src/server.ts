import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { randomUUID } from "node:crypto";
import { fetch as undiciFetch, ProxyAgent } from "undici";
import type {
  Message,
  AgentResponse,
  AgentSearchResult,
  AgentToolOptions,
  AgentCallHooks,
  ApiKeyProvider,
  PerplexityServerOptions,
  SearchOptions,
  SearchResponse,
  SearchType,
  UndiciRequestOptions
} from "./types.js";
import { AgentResponseSchema, SearchResponseSchema } from "./validation.js";

export type { ApiKeyProvider, PerplexityServerOptions } from "./types.js";

// =====================================================================
// FORK ADDITION: in-memory async job store for start_research/check_research
// pattern (Patch E). Lets clients with short tool-call timeouts (Claude.ai
// web's 60s cap) run deep research without timing out.
//
// Trade-offs:
// - State is in-memory only. Container restart → all in-flight jobs lost.
//   Acceptable for our single-instance Railway deployment with infrequent
//   restarts; clients can detect via `status: not_found` and retry.
// - 30-minute retention ceiling for completed jobs. 5-minute hard timeout
//   for stuck jobs. Periodic GC every 60s.
// - Each job ~5-50KB. Memory bound is generous (~5MB at 100 concurrent).
// =====================================================================

type JobStatus = "pending" | "running" | "done" | "error";

interface ResearchJob {
  id: string;
  status: JobStatus;
  query_preview: string; // first 200 chars of the user message — for diagnostics
  startedAt: number;     // epoch ms
  finishedAt?: number;
  result?: string;
  error?: string;
}

const RESEARCH_JOBS = new Map<string, ResearchJob>();

const JOB_RETENTION_MS = 30 * 60_000; // 30 min for completed jobs
const JOB_HARD_TIMEOUT_MS = 5 * 60_000; // 5 min for stuck pending/running jobs
// Bounds memory and paid deep-research spend from start_research bursts.
export const MAX_INFLIGHT_RESEARCH_JOBS = 10;

function gcResearchJobs(): void {
  const now = Date.now();
  for (const [id, job] of RESEARCH_JOBS) {
    const age = now - job.startedAt;
    if (job.status === "done" || job.status === "error") {
      if (age > JOB_RETENTION_MS) RESEARCH_JOBS.delete(id);
    } else if (age > JOB_HARD_TIMEOUT_MS) {
      job.status = "error";
      job.error = `Job exceeded ${JOB_HARD_TIMEOUT_MS / 1000}s hard timeout`;
      job.finishedAt = now;
    }
  }
}

// One global GC interval. Keeps the process alive in dev, harmless in prod.
setInterval(gcResearchJobs, 60_000).unref();

const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;
const PERPLEXITY_BASE_URL = process.env.PERPLEXITY_BASE_URL || "https://api.perplexity.ai";
const VERSION = "1.4.0";

// Agent API presets backing each tool: https://docs.perplexity.ai/docs/agent-api/presets
export const ASK_PRESET = "fast";
export const REASON_PRESET = "medium";
export const RESEARCH_PRESET = "high";

export function getProxyUrl(): string | undefined {
  return process.env.PERPLEXITY_PROXY ||
         process.env.HTTPS_PROXY ||
         process.env.HTTP_PROXY ||
         undefined;
}

export async function proxyAwareFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const proxyUrl = getProxyUrl();

  if (proxyUrl) {
    const proxyAgent = new ProxyAgent(proxyUrl);
    const undiciOptions: UndiciRequestOptions = {
      ...options,
      dispatcher: proxyAgent,
    };
    const response = await undiciFetch(url, undiciOptions);
    return response as unknown as Response;
  }

  return fetch(url, options);
}

export function validateMessages(messages: unknown, toolName: string): asserts messages is Message[] {
  if (!Array.isArray(messages)) {
    throw new Error(`Invalid arguments for ${toolName}: 'messages' must be an array`);
  }

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg || typeof msg !== 'object') {
      throw new Error(`Invalid message at index ${i}: must be an object`);
    }
    if (!msg.role || typeof msg.role !== 'string') {
      throw new Error(`Invalid message at index ${i}: 'role' must be a string`);
    }
    if (msg.content === undefined || msg.content === null || typeof msg.content !== 'string') {
      throw new Error(`Invalid message at index ${i}: 'content' must be a string`);
    }
  }
}

async function makeApiRequest(
  endpoint: string,
  body: Record<string, unknown>,
  serviceOrigin: string | undefined,
  signal?: AbortSignal,
  apiKey?: ApiKeyProvider,
): Promise<Response> {
  // A configured provider fully replaces the env var: falling back would let
  // a multi-tenant misconfiguration silently bill the process-wide key.
  let resolvedApiKey: string | undefined;
  if (apiKey) {
    resolvedApiKey = apiKey();
    if (!resolvedApiKey) {
      throw new Error("API key provider returned no key");
    }
  } else {
    resolvedApiKey = PERPLEXITY_API_KEY;
    if (!resolvedApiKey) {
      throw new Error("PERPLEXITY_API_KEY environment variable is required");
    }
  }

  // Read timeout fresh each time to respect env var changes
  const TIMEOUT_MS = parseInt(process.env.PERPLEXITY_TIMEOUT_MS || "300000", 10);

  const url = new URL(`${PERPLEXITY_BASE_URL}/${endpoint}`);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
  // An abort from the caller's signal also tears down an in-flight body stream.
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", () => controller.abort(), { once: true });
    }
  }

  let response;
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${resolvedApiKey}`,
      "User-Agent": `perplexity-mcp/${VERSION}`,
      "X-Source": "pplx-mcp-server",
      "X-Pplx-Integration": `perplexity-mcp/${VERSION}`,
    };
    if (serviceOrigin) {
      headers["X-Service"] = serviceOrigin;
    }
    response = await proxyAwareFetch(url.toString(), {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === "AbortError") {
      if (signal?.aborted) {
        // The caller knows whether its signal meant cancellation or deadline.
        throw error;
      }
      throw new Error(`Request timeout: Perplexity API did not respond within ${TIMEOUT_MS}ms. Consider increasing PERPLEXITY_TIMEOUT_MS.`);
    }
    throw new Error(`Network error while calling Perplexity API: ${error}`);
  }
  clearTimeout(timeoutId);

  if (!response.ok) {
    let errorText;
    try {
      errorText = await response.text();
    } catch (parseError) {
      errorText = "Unable to parse error response";
    }
    throw new Error(
      `Perplexity API error: ${response.status} ${response.statusText}\n${errorText}`
    );
  }

  return response;
}

/** Best-effort cancellation of an agent run so an abandoned request stops billing. */
export async function cancelAgentResponse(responseId: string, serviceOrigin?: string, apiKey?: ApiKeyProvider): Promise<void> {
  try {
    await makeApiRequest(`v1/agent/${encodeURIComponent(responseId)}/cancel`, {}, serviceOrigin, undefined, apiKey);
  } catch {
    // The run may already be terminal; nothing actionable either way.
  }
}

/**
 * Consume an Agent API SSE stream and return the final response object.
 * Emits progress via hooks.onProgress as reasoning events arrive.
 */
export async function consumeAgentStream(
  response: Response,
  hooks?: AgentCallHooks,
  serviceOrigin?: string,
  deadlineSignal?: AbortSignal,
  apiKey?: ApiKeyProvider,
): Promise<AgentResponse> {
  const body = response.body;
  if (!body) {
    throw new Error("Response body is null");
  }

  const reader = (body as ReadableStream<Uint8Array>).getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let eventName: string | undefined;
  let responseId: string | undefined;
  let finalResponse: unknown;
  let streamError: string | undefined;

  const handleEvent = (type: string, parsed: Record<string, unknown>) => {
    const eventResponse = parsed.response as { id?: string } | undefined;
    if (eventResponse?.id) {
      responseId = eventResponse.id;
    }
    switch (type) {
      case "response.completed":
        finalResponse = parsed.response;
        break;
      case "response.failed": {
        const error = parsed.error as { message?: string } | undefined;
        streamError = error?.message || "Agent request failed";
        break;
      }
      case "response.cancelled":
        streamError = "Agent request was cancelled";
        break;
      case "error": {
        const error = parsed.error as { message?: string } | undefined;
        streamError = error?.message || (typeof parsed.message === "string" ? parsed.message : "Agent request failed");
        break;
      }
      case "response.reasoning.search_queries": {
        const queries = Array.isArray(parsed.queries) ? parsed.queries.filter((q) => typeof q === "string") : [];
        if (queries.length > 0) {
          hooks?.onProgress?.({ message: `Searching: ${queries.join(" | ")}` });
        }
        break;
      }
      case "response.reasoning.search_results": {
        const results = Array.isArray(parsed.results) ? parsed.results : [];
        if (results.length > 0) {
          hooks?.onProgress?.({ message: `Reading ${results.length} search results` });
        }
        break;
      }
      case "response.reasoning.fetch_url_queries":
        hooks?.onProgress?.({ message: "Fetching page content" });
        break;
      case "response.output_item.added": {
        const item = parsed.item as { type?: string } | undefined;
        if (item?.type === "message") {
          hooks?.onProgress?.({ message: "Writing answer" });
        }
        break;
      }
    }
  };

  // Stop reading as soon as a terminal event is in hand: waiting for the
  // server to close the socket would let a deadline abort in the tail window
  // discard an already-received (and billed) answer.
  const isTerminal = () => finalResponse !== undefined || streamError !== undefined;

  try {
    while (!isTerminal()) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      // Keep the last potentially incomplete line in the buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          eventName = undefined;
          continue;
        }
        if (trimmed.startsWith("event:")) {
          eventName = trimmed.slice("event:".length).trim();
          continue;
        }
        if (!trimmed.startsWith("data:")) continue;

        const data = trimmed.slice("data:".length).trim();
        if (data === "[DONE]") continue;

        try {
          const parsed = JSON.parse(data) as Record<string, unknown>;
          const type = typeof parsed.type === "string" ? parsed.type : eventName;
          if (type) {
            handleEvent(type, parsed);
          }
        } catch {
          // Skip malformed JSON chunks (e.g. keep-alive pings)
        }
        if (isTerminal()) break;
      }
    }
    if (isTerminal()) {
      void reader.cancel().catch(() => {});
    }
  } catch (error) {
    if (hooks?.signal?.aborted || deadlineSignal?.aborted) {
      if (responseId) {
        // Stop the server-side run so an abandoned request stops billing.
        void cancelAgentResponse(responseId, serviceOrigin, apiKey);
      }
      if (hooks?.signal?.aborted) {
        throw new Error("Request cancelled");
      }
      throw error;
    }
    throw new Error(`Network error while streaming from Perplexity API: ${error}`);
  }

  if (hooks?.signal?.aborted) {
    if (responseId) {
      void cancelAgentResponse(responseId, serviceOrigin, apiKey);
    }
    throw new Error("Request cancelled");
  }

  if (streamError) {
    throw new Error(`Perplexity API error: ${streamError}`);
  }

  if (!finalResponse) {
    throw new Error("Agent stream ended without a completed response");
  }

  try {
    return AgentResponseSchema.parse(finalResponse) as AgentResponse;
  } catch (error) {
    throw new Error(`Invalid response from Perplexity Agent API: ${error}`);
  }
}

export function extractAgentText(response: AgentResponse): string {
  return response.output
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content ?? [])
    .filter((part) => part.type === "output_text" && typeof part.text === "string")
    .map((part) => part.text as string)
    .join("");
}

/**
 * Append the "Citations:" block ("[n] url" lines) to the answer text, keyed
 * by the numeric search-result ids that inline references point at. The
 * answer body is never modified: bracketed tokens also appear in code, LaTeX,
 * and slice syntax, so any rewrite risks corrupting the answer. If ids are
 * missing or ambiguous, all unique source URLs are appended with positional
 * numbering instead.
 */
export function formatAgentResponseText(response: AgentResponse): string {
  const text = extractAgentText(response);

  const entries: Array<{ id?: number | null; url: string }> = [];
  const urlById = new Map<number, string>();
  let idsUsable = true;
  for (const item of response.output) {
    if (item.type !== "search_results" || !Array.isArray(item.results)) continue;
    for (const result of item.results as AgentSearchResult[]) {
      if (!result.url) continue;
      entries.push({ id: result.id, url: result.url });
      if (typeof result.id !== "number") {
        idsUsable = false;
        continue;
      }
      const existing = urlById.get(result.id);
      if (existing === undefined) {
        urlById.set(result.id, result.url);
      } else if (existing !== result.url) {
        idsUsable = false;
      }
    }
  }

  if (entries.length === 0) {
    return text;
  }

  let output = text + "\n\nCitations:\n";
  if (idsUsable) {
    const emitted = new Set<number>();
    for (const entry of entries) {
      const id = entry.id as number;
      if (emitted.has(id)) continue;
      emitted.add(id);
      output += `[${id}] ${entry.url}\n`;
    }
    return output;
  }

  const seen = new Set<string>();
  let position = 0;
  for (const entry of entries) {
    if (seen.has(entry.url)) continue;
    seen.add(entry.url);
    position += 1;
    output += `[${position}] ${entry.url}\n`;
  }
  return output;
}

function buildWebSearchTool(options?: AgentToolOptions): Record<string, unknown> | undefined {
  if (!options) return undefined;
  // Local compatibility policy, not an upstream equivalence to Sonar's sizes.
  // Explicit budgets override either legacy default independently.
  const legacyBudgets = options.search_context_size && {
    low: { max_tokens: 2000, max_tokens_per_page: 512 },
    medium: { max_tokens: 4000, max_tokens_per_page: 1024 },
    high: { max_tokens: 8000, max_tokens_per_page: 2048 },
  }[options.search_context_size];
  const maxTokens = options.max_tokens ?? (legacyBudgets && legacyBudgets.max_tokens);
  const maxTokensPerPage = options.max_tokens_per_page ?? (legacyBudgets && legacyBudgets.max_tokens_per_page);
  const filters: Record<string, unknown> = {
    ...(options.search_recency_filter && { search_recency_filter: options.search_recency_filter }),
    ...(options.search_domain_filter && { search_domain_filter: options.search_domain_filter }),
  };
  const tool: Record<string, unknown> = {
    type: "web_search",
    ...(Object.keys(filters).length > 0 && { filters }),
    ...(maxTokens !== undefined && { max_tokens: maxTokens }),
    ...(maxTokensPerPage !== undefined && { max_tokens_per_page: maxTokensPerPage }),
  };
  // No override means the preset's own web_search config stays in effect.
  return Object.keys(tool).length > 1 ? tool : undefined;
}

export async function performAgentResponse(
  messages: Message[],
  preset: string,
  serviceOrigin?: string,
  options?: AgentToolOptions,
  hooks?: AgentCallHooks,
  apiKey?: ApiKeyProvider,
): Promise<string> {
  const webSearchTool = buildWebSearchTool(options);

  const body: Record<string, unknown> = {
    preset,
    input: messages.map((message) => ({
      type: "message",
      role: message.role,
      content: message.content,
    })),
    // Always stream: long runs outlive intermediate proxy timeouts, and the
    // reasoning events double as progress reporting.
    stream: true,
    ...(webSearchTool && { tools: [webSearchTool] }),
  };

  // PERPLEXITY_TIMEOUT_MS bounds the whole call, not just time-to-headers:
  // streamed responses return headers immediately, so a headers-only timeout
  // would never fire.
  const TIMEOUT_MS = parseInt(process.env.PERPLEXITY_TIMEOUT_MS || "300000", 10);
  const deadline = new AbortController();
  const timeoutId = setTimeout(() => deadline.abort(), TIMEOUT_MS);
  const abortDeadline = () => deadline.abort();
  if (hooks?.signal) {
    if (hooks.signal.aborted) {
      deadline.abort();
    } else {
      hooks.signal.addEventListener("abort", abortDeadline, { once: true });
    }
  }

  try {
    const response = await makeApiRequest("v1/agent", body, serviceOrigin, deadline.signal, apiKey);
    const agentResponse = await consumeAgentStream(response, hooks, serviceOrigin, deadline.signal, apiKey);
    return formatAgentResponseText(agentResponse);
  } catch (error) {
    if (hooks?.signal?.aborted) {
      throw new Error("Request cancelled");
    }
    if (deadline.signal.aborted) {
      throw new Error(`Request timeout: Perplexity API did not respond within ${TIMEOUT_MS}ms. Consider increasing PERPLEXITY_TIMEOUT_MS.`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
    hooks?.signal?.removeEventListener("abort", abortDeadline);
  }
}

export function formatSearchResults(data: SearchResponse): string {
  if (!data.results || !Array.isArray(data.results)) {
    return "No search results found.";
  }

  let formattedResults = `Found ${data.results.length} search results:\n\n`;

  data.results.forEach((result, index) => {
    formattedResults += `${index + 1}. **${result.title}**\n`;
    formattedResults += `   URL: ${result.url}\n`;
    if (result.snippet) {
      formattedResults += `   ${result.snippet}\n`;
    }
    if (result.date) {
      formattedResults += `   Date: ${result.date}\n`;
    }
    if (result.last_updated) {
      formattedResults += `   Last updated: ${result.last_updated}\n`;
    }
    formattedResults += `\n`;
  });

  return formattedResults;
}

export async function performSearch(
  query: string,
  maxResults: number = 10,
  maxTokensPerPage: number = 1024,
  country?: string,
  filters?: SearchOptions,
  serviceOrigin?: string,
  apiKey?: ApiKeyProvider,
): Promise<string> {
  const body: Record<string, unknown> = {
    query: query,
    max_results: maxResults,
    max_tokens_per_page: maxTokensPerPage,
    ...(country && { country }),
    ...(filters?.search_recency_filter && { search_recency_filter: filters.search_recency_filter }),
    ...(filters?.search_domain_filter && { search_domain_filter: filters.search_domain_filter }),
    ...(filters?.search_type && { search_type: filters.search_type }),
  };

  const response = await makeApiRequest("search", body, serviceOrigin, undefined, apiKey);

  let data: SearchResponse;
  try {
    const json = await response.json();
    data = SearchResponseSchema.parse(json);
  } catch (error) {
    throw new Error(`Failed to parse JSON response from Perplexity Search API: ${error}`);
  }

  return formatSearchResults(data);
}

interface ToolExtra {
  signal?: AbortSignal;
  _meta?: { progressToken?: string | number };
  sendNotification?: (notification: {
    method: string;
    params: Record<string, unknown>;
  }) => Promise<void>;
}

function buildHooks(extra: ToolExtra | undefined): AgentCallHooks {
  const progressToken = extra?._meta?.progressToken;
  const sendNotification = extra?.sendNotification;
  let progress = 0;
  return {
    signal: extra?.signal,
    onProgress:
      progressToken !== undefined && sendNotification
        ? (update) => {
            void sendNotification({
              method: "notifications/progress",
              params: {
                progressToken,
                progress: ++progress,
                message: update.message,
              },
            }).catch(() => {});
          }
        : undefined,
  };
}

export function createPerplexityServer(serviceOrigin?: string, serverOptions?: PerplexityServerOptions) {
  const server = new McpServer(
    {
      name: "ai.perplexity/mcp-server",
      version: VERSION,
    },
    {
      instructions:
        "Perplexity AI server for web-grounded search, research, and reasoning, backed by the Perplexity Agent API. " +
        "\n\nTOOL TIMING (critical for Claude.ai web — has a hard 60s tool-call timeout):\n" +
        "- perplexity_search: 1-3s. Quick web search returning ranked results (title/URL/snippet/date). No AI synthesis.\n" +
        "- perplexity_ask: ~5-15s. Agent API " + ASK_PRESET + " preset, answer with citations. PREFERRED for time-sensitive queries.\n" +
        "- perplexity_reason: ~10-30s. Agent API " + REASON_PRESET + " preset, step-by-step reasoning. May approach 60s on complex prompts.\n" +
        "- perplexity_research (sync): ~60-240s. Agent API " + RESEARCH_PRESET + " preset. WILL TIME OUT in Claude.ai web — use ONLY in Claude Desktop / Claude Code / terminal MCP clients.\n" +
        "- start_research + check_research (async): the timeout-safe alternative for deep research from any client. " +
        "Call start_research → returns job_id immediately → poll check_research every 30s until status='done'. Bypasses the 60s client cap.\n" +
        "\nSelection rule: for multi-topic 'give me a summary of X, Y, Z' queries from Claude.ai web, prefer perplexity_ask. " +
        "Reach for sync perplexity_research only when you know your client tolerates >60s. " +
        "Use start_research when you need deep research AND the client has a short timeout. " +
        "All tools are read-only and access live web data.",
    }
  );

  const messageSchema = z.object({
    role: z.enum(["system", "user", "assistant"]).describe("Role of the message sender"),
    content: z.string().describe("The content of the message"),
  });

  const messagesField = z.array(messageSchema).describe("Array of conversation messages");

  const searchRecencyFilterField = z.enum(["hour", "day", "week", "month", "year"]).optional()
    .describe("Filter search results by recency. Use 'hour' for very recent news, 'day' for today's updates, 'week' for this week, etc.");

  const searchDomainFilterField = z.array(z.string()).optional()
    .describe("Restrict search results to specific domains (e.g., ['wikipedia.org', 'arxiv.org']). Use '-' prefix for exclusion (e.g., ['-reddit.com']).");

  const searchContextSizeField = z.enum(["low", "medium", "high"]).optional()
    .describe("Deprecated compatibility shorthand for search token budgets: low=2000/512, medium=4000/1024, high=8000/2048 (total/per-page). Explicit max_tokens and max_tokens_per_page override these defaults.");
  const MAX_SEARCH_TOKENS = 120_000;
  const MAX_SEARCH_TOKENS_PER_PAGE = 20_000;
  const maxTokensField = z.number().int().positive().max(MAX_SEARCH_TOKENS).optional()
    .describe("Maximum total search-context tokens, not generated answer tokens. Overrides the legacy search_context_size total budget.");
  const maxTokensPerPageField = z.number().int().positive().max(MAX_SEARCH_TOKENS_PER_PAGE).optional()
    .describe("Maximum search-context tokens extracted per result page. Overrides the legacy search_context_size per-page budget.");

  const responseOutputSchema = {
    response: z.string().describe("AI-generated text response with numbered citation references"),
  };

  // Input schemas
  const askAndReasonInputSchema = {
    messages: messagesField,
    search_recency_filter: searchRecencyFilterField,
    search_domain_filter: searchDomainFilterField,
    search_context_size: searchContextSizeField,
    max_tokens: maxTokensField,
    max_tokens_per_page: maxTokensPerPageField,
  };
  const researchInputSchema = {
    messages: messagesField,
  };

  server.registerTool(
    "perplexity_ask",
    {
      title: "Ask Perplexity",
      description: "Answer a question using web-grounded AI (Perplexity Agent API, " + ASK_PRESET + " preset). " +
        "Latency: ~5-15s typical. SAFE for Claude.ai web (60s client cap). " +
        "Best for: quick factual questions, summaries, explanations, general Q&A, and " +
        "MULTI-TOPIC SUMMARIES that would otherwise need slow research. " +
        "Returns a text response with numbered citations. Fastest and cheapest option. " +
        "Supports filtering by recency (hour/day/week/month/year), domain restrictions, and search token budgets. " +
        "For deep multi-source research from a timeout-tolerant client, use perplexity_research. " +
        "For deep research from Claude.ai web (60s cap), use start_research + check_research instead. " +
        "For step-by-step reasoning, use perplexity_reason.",
      inputSchema: askAndReasonInputSchema as any,
      outputSchema: responseOutputSchema as any,
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
        idempotentHint: false,
        destructiveHint: false,
      },
    },
    async (args: any, extra: any) => {
      const { messages, search_recency_filter, search_domain_filter, search_context_size, max_tokens, max_tokens_per_page } = args as {
        messages: Message[];
        search_recency_filter?: "hour" | "day" | "week" | "month" | "year";
        search_domain_filter?: string[];
        search_context_size?: "low" | "medium" | "high";
        max_tokens?: number;
        max_tokens_per_page?: number;
      };
      validateMessages(messages, "perplexity_ask");
      const options = {
        ...(search_recency_filter && { search_recency_filter }),
        ...(search_domain_filter && { search_domain_filter }),
        ...(search_context_size && { search_context_size }),
        ...(max_tokens !== undefined && { max_tokens }),
        ...(max_tokens_per_page !== undefined && { max_tokens_per_page }),
      };
      const result = await performAgentResponse(
        messages,
        ASK_PRESET,
        serviceOrigin,
        Object.keys(options).length > 0 ? options : undefined,
        buildHooks(extra),
        serverOptions?.apiKey,
      );
      return {
        content: [{ type: "text" as const, text: result }],
        structuredContent: { response: result },
      };
    }
  );

  server.registerTool(
    "perplexity_research",
    {
      title: "Deep Research (sync — may time out)",
      description: "Conduct deep, multi-source research on a topic (Perplexity Agent API, " + RESEARCH_PRESET + " preset). " +
        "Latency: ~60-240 seconds. WILL TIME OUT in Claude.ai web (60s hard client cap). " +
        "Use ONLY in Claude Desktop, Claude Code, terminal MCP clients, or other " +
        "clients with long tool-call tolerance. " +
        "From Claude.ai web: use start_research + check_research instead (async pattern, no cap). " +
        "Best for: literature reviews, comprehensive overviews, investigative queries needing " +
        "many sources. Returns a detailed response with numbered citations. " +
        "For quick factual questions, use perplexity_ask. " +
        "For logical analysis and reasoning, use perplexity_reason.",
      inputSchema: researchInputSchema as any,
      outputSchema: responseOutputSchema as any,
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
        idempotentHint: false,
        destructiveHint: false,
      },
    },
    async (args: any, extra: any) => {
      const { messages } = args as { messages: Message[] };
      validateMessages(messages, "perplexity_research");
      const result = await performAgentResponse(
        messages,
        RESEARCH_PRESET,
        serviceOrigin,
        undefined,
        buildHooks(extra),
        serverOptions?.apiKey,
      );
      return {
        content: [{ type: "text" as const, text: result }],
        structuredContent: { response: result },
      };
    }
  );

  server.registerTool(
    "perplexity_reason",
    {
      title: "Advanced Reasoning",
      description: "Analyze a question using step-by-step reasoning with web grounding (Perplexity Agent API, " + REASON_PRESET + " preset). " +
        "Latency: ~10-30s typical, can approach 60s on complex prompts (risk of Claude.ai web timeout). " +
        "Best for: math, logic, comparisons, complex arguments, and tasks requiring chain-of-thought. " +
        "Returns a reasoned response with numbered citations. " +
        "Supports filtering by recency (hour/day/week/month/year), domain restrictions, and search token budgets. " +
        "For quick factual questions, use perplexity_ask. " +
        "For comprehensive multi-source research, use perplexity_research (sync) or start_research (async).",
      inputSchema: askAndReasonInputSchema as any,
      outputSchema: responseOutputSchema as any,
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
        idempotentHint: false,
        destructiveHint: false,
      },
    },
    async (args: any, extra: any) => {
      const { messages, search_recency_filter, search_domain_filter, search_context_size, max_tokens, max_tokens_per_page } = args as {
        messages: Message[];
        search_recency_filter?: "hour" | "day" | "week" | "month" | "year";
        search_domain_filter?: string[];
        search_context_size?: "low" | "medium" | "high";
        max_tokens?: number;
        max_tokens_per_page?: number;
      };
      validateMessages(messages, "perplexity_reason");
      const options = {
        ...(search_recency_filter && { search_recency_filter }),
        ...(search_domain_filter && { search_domain_filter }),
        ...(search_context_size && { search_context_size }),
        ...(max_tokens !== undefined && { max_tokens }),
        ...(max_tokens_per_page !== undefined && { max_tokens_per_page }),
      };
      const result = await performAgentResponse(
        messages,
        REASON_PRESET,
        serviceOrigin,
        Object.keys(options).length > 0 ? options : undefined,
        buildHooks(extra),
        serverOptions?.apiKey,
      );
      return {
        content: [{ type: "text" as const, text: result }],
        structuredContent: { response: result },
      };
    }
  );

  const searchInputSchema = {
    query: z.string().describe("Search query string"),
    max_results: z.number().min(1).max(20).optional()
      .describe("Maximum number of results to return (1-20, default: 10)"),
    max_tokens_per_page: z.number().min(256).max(2048).optional()
      .describe("Maximum tokens to extract per webpage (default: 1024)"),
    country: z.string().optional()
      .describe("ISO 3166-1 alpha-2 country code for regional results (e.g., 'US', 'GB')"),
    search_recency_filter: searchRecencyFilterField,
    search_domain_filter: searchDomainFilterField,
    search_type: z.enum(["web", "fast"]).optional()
      .describe("Search backend. 'web' (default) is the standard web search. 'fast' is lower latency and lower cost; prefer it for routine lookups inside agent loops, and keep 'web' for rare or ambiguous questions."),
  };

  const searchOutputSchema = {
    results: z.string().describe("Formatted search results, each with title, URL, snippet, and date"),
  };

  server.registerTool(
    "perplexity_search",
    {
      title: "Search the Web",
      description: "Search the web and return a ranked list of results with titles, URLs, snippets, and dates. " +
        "Latency: 1-3s. FASTEST tool — always safe within Claude.ai web 60s cap. " +
        "Best for: finding specific URLs, checking recent news, verifying facts, discovering sources. " +
        "Returns formatted results (title, URL, snippet, date) with no AI synthesis. " +
        "Supports recency filters, domain restrictions, and a lower-latency fast search mode. " +
        "For AI-generated answers with citations, use perplexity_ask instead.",
      inputSchema: searchInputSchema as any,
      outputSchema: searchOutputSchema as any,
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
        idempotentHint: false,
        destructiveHint: false,
      },
    },
    async (args: any) => {
      const { query, max_results, max_tokens_per_page, country, search_recency_filter, search_domain_filter, search_type } = args as {
        query: string;
        max_results?: number;
        max_tokens_per_page?: number;
        country?: string;
        search_recency_filter?: "hour" | "day" | "week" | "month" | "year";
        search_domain_filter?: string[];
        search_type?: SearchType;
      };
      const maxResults = typeof max_results === "number" ? max_results : 10;
      const maxTokensPerPage = typeof max_tokens_per_page === "number" ? max_tokens_per_page : 1024;
      const countryCode = typeof country === "string" ? country : undefined;
      const filters: SearchOptions = {
        ...(search_recency_filter && { search_recency_filter }),
        ...(search_domain_filter && { search_domain_filter }),
        ...(search_type && { search_type }),
      };

      const result = await performSearch(query, maxResults, maxTokensPerPage, countryCode, filters, serviceOrigin, serverOptions?.apiKey);
      return {
        content: [{ type: "text" as const, text: result }],
        structuredContent: { results: result },
      };
    }
  );

  // =====================================================================
  // FORK ADDITION: async polling pattern for deep research (RESEARCH_PRESET).
  // Bypasses Claude.ai web's 60s hard tool-call timeout.
  // =====================================================================

  const startResearchInputSchema = {
    messages: messagesField,
  };
  const startResearchOutputSchema = {
    job_id: z.string().describe("Opaque job identifier — pass to check_research to poll status"),
    status: z.string().describe("Initial status (always 'pending' on success)"),
    poll_after_seconds: z.number().describe("Recommended seconds to wait before first check_research call"),
  };

  server.registerTool(
    "start_research",
    {
      title: "Start Deep Research (async)",
      description: "Kick off a deep research job (Agent API " + RESEARCH_PRESET + " preset) and return immediately with a job_id (~1s). " +
        "Use this from Claude.ai web (or any client with <60s tool-call timeout) to do deep " +
        "research without hitting the cap. After calling: poll check_research every 30s with " +
        "the returned job_id until status='done'. Typical completion: 30-180s. " +
        "For sync deep research (when client tolerates >180s), use perplexity_research instead.",
      inputSchema: startResearchInputSchema as any,
      outputSchema: startResearchOutputSchema as any,
      annotations: {
        readOnlyHint: false, // creates server-side state (the job)
        openWorldHint: true,
        idempotentHint: false,
        destructiveHint: false,
      },
    },
    async (args: any) => {
      const { messages } = args as { messages: Message[] };
      validateMessages(messages, "start_research");

      let inflight = 0;
      for (const j of RESEARCH_JOBS.values()) {
        if (j.status === "pending" || j.status === "running") inflight++;
      }
      if (inflight >= MAX_INFLIGHT_RESEARCH_JOBS) {
        return {
          content: [
            {
              type: "text" as const,
              text: `Too many research jobs in flight (${inflight}/${MAX_INFLIGHT_RESEARCH_JOBS}). Retry after a running job finishes.`,
            },
          ],
          isError: true,
        };
      }

      const id = randomUUID();
      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
      const job: ResearchJob = {
        id,
        status: "pending",
        query_preview: (lastUserMsg?.content ?? "").slice(0, 200),
        startedAt: Date.now(),
      };
      RESEARCH_JOBS.set(id, job);

      // Fire-and-forget background work. Errors handled inside the closure;
      // never escape to the start_research caller.
      (async () => {
        job.status = "running";
        try {
          const result = await performAgentResponse(
            messages,
            RESEARCH_PRESET,
            serviceOrigin,
            undefined,
            undefined,
            serverOptions?.apiKey,
          );
          job.status = "done";
          job.result = result;
          job.finishedAt = Date.now();
        } catch (e) {
          job.status = "error";
          job.error = e instanceof Error ? e.message : String(e);
          job.finishedAt = Date.now();
        }
      })();

      return {
        content: [
          {
            type: "text" as const,
            text:
              `Research job started.\n` +
              `job_id: ${id}\n` +
              `status: pending\n` +
              `Recommended next step: call check_research with this job_id after 30s.`,
          },
        ],
        structuredContent: { job_id: id, status: "pending", poll_after_seconds: 30 },
      };
    },
  );

  const checkResearchInputSchema = {
    job_id: z.string().describe("Job ID returned by start_research"),
  };
  const checkResearchOutputSchema = {
    status: z.string().describe("One of: pending, running, done, error, not_found"),
    elapsed_seconds: z.number().describe("How long the job has been running (or took, if finished)"),
    response: z.string().optional().describe("Final research response — only present when status='done'"),
    error: z.string().optional().describe("Error message — only present when status='error'"),
  };

  server.registerTool(
    "check_research",
    {
      title: "Check Research Status (async)",
      description: "Poll the status of a research job started by start_research. " +
        "Returns within ~1s — safe within any client's tool-call timeout. " +
        "Returns status='pending'/'running' (keep polling), 'done' (response field has the result), " +
        "'error' (error field has the failure reason), or 'not_found' (job ID expired or never existed). " +
        "Recommended poll interval: 30s. Job retention: 30 minutes after completion.",
      inputSchema: checkResearchInputSchema as any,
      outputSchema: checkResearchOutputSchema as any,
      annotations: {
        readOnlyHint: true,
        openWorldHint: false,
        idempotentHint: true,
        destructiveHint: false,
      },
    },
    async (args: any) => {
      const { job_id } = args as { job_id: string };
      const job = RESEARCH_JOBS.get(job_id);
      if (!job) {
        return {
          content: [
            {
              type: "text" as const,
              text:
                `Job ${job_id} not found. Possible reasons:\n` +
                `- Never existed (typo in job_id)\n` +
                `- Expired (jobs are kept for 30 minutes after completion)\n` +
                `- Server restarted (in-memory state was lost — start a new job)`,
            },
          ],
          structuredContent: { status: "not_found", elapsed_seconds: 0 },
        };
      }
      const elapsed = Math.round(((job.finishedAt ?? Date.now()) - job.startedAt) / 1000);

      if (job.status === "done") {
        return {
          content: [{ type: "text" as const, text: job.result ?? "" }],
          structuredContent: {
            status: "done",
            elapsed_seconds: elapsed,
            response: job.result,
          },
        };
      }
      if (job.status === "error") {
        return {
          content: [
            {
              type: "text" as const,
              text: `Research job ${job_id} failed after ${elapsed}s.\nError: ${job.error}`,
            },
          ],
          structuredContent: {
            status: "error",
            elapsed_seconds: elapsed,
            error: job.error,
          },
          isError: true,
        };
      }
      // pending or running
      return {
        content: [
          {
            type: "text" as const,
            text:
              `Research job ${job_id} is still ${job.status} (elapsed ${elapsed}s). ` +
              `Poll again in 30s. Typical completion: 30-180s.`,
          },
        ],
        structuredContent: { status: job.status, elapsed_seconds: elapsed },
      };
    },
  );

  advertiseJsonSchema202012(server.server);

  return server.server;
}

// The MCP TypeScript SDK stamps draft-07 on tools/list schemas
// (modelcontextprotocol/typescript-sdk#2084); 2020-12-only clients reject
// them. The schemas are valid under both dialects, so only the declared
// dialect needs rewriting. Remove once typescript-sdk#2085 ships.
const JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema";

function advertiseJsonSchema202012(server: McpServer["server"]) {
  type ListToolsHandler = (request: unknown, extra: unknown) => Promise<{
    tools: Array<{
      inputSchema?: { $schema?: string };
      outputSchema?: { $schema?: string };
    }>;
  }>;
  // _requestHandlers is private SDK state; goes away with this shim.
  const handlers = (server as unknown as {
    _requestHandlers: Map<string, ListToolsHandler>;
  })._requestHandlers;
  const listTools = handlers.get("tools/list");
  if (!listTools) {
    return;
  }
  handlers.set("tools/list", async (request, extra) => {
    const result = await listTools(request, extra);
    for (const tool of result.tools ?? []) {
      if (tool.inputSchema?.$schema) {
        tool.inputSchema.$schema = JSON_SCHEMA_2020_12;
      }
      if (tool.outputSchema?.$schema) {
        tool.outputSchema.$schema = JSON_SCHEMA_2020_12;
      }
    }
    return result;
  });
}
