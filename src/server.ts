import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { randomUUID } from "node:crypto";
import { fetch as undiciFetch, ProxyAgent } from "undici";
import type {
  Message,
  ChatCompletionResponse,
  ChatCompletionOptions,
  SearchResponse,
  SearchRequestBody,
  UndiciRequestOptions
} from "./types.js";
import { ChatCompletionResponseSchema, SearchResponseSchema } from "./validation.js";

// =====================================================================
// FORK ADDITION: in-memory async job store for start_research/check_research
// pattern (Patch E). Lets clients with short tool-call timeouts (Claude.ai
// web's 60s cap) run sonar-deep-research without timing out.
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
const VERSION = "0.9.0";

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

export function stripThinkingTokens(content: string): string {
  return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

async function makeApiRequest(
  endpoint: string,
  body: Record<string, unknown>,
  serviceOrigin: string | undefined,
): Promise<Response> {
  if (!PERPLEXITY_API_KEY) {
    throw new Error("PERPLEXITY_API_KEY environment variable is required");
  }

  // Read timeout fresh each time to respect env var changes
  const TIMEOUT_MS = parseInt(process.env.PERPLEXITY_TIMEOUT_MS || "300000", 10);

  const url = new URL(`${PERPLEXITY_BASE_URL}/${endpoint}`);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let response;
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
      "User-Agent": `perplexity-mcp/${VERSION}`,
      "X-Source": "pplx-mcp-server",
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

export async function consumeSSEStream(response: Response): Promise<ChatCompletionResponse> {
  const body = response.body;
  if (!body) {
    throw new Error("Response body is null");
  }

  const reader = (body as ReadableStream<Uint8Array>).getReader();
  const decoder = new TextDecoder();

  let contentParts: string[] = [];
  let citations: string[] | undefined;
  let usage: ChatCompletionResponse["usage"] | undefined;
  let id: string | undefined;
  let model: string | undefined;
  let created: number | undefined;
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    // Keep the last potentially incomplete line in the buffer
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data:")) continue;

      const data = trimmed.slice("data:".length).trim();
      if (data === "[DONE]") continue;

      try {
        const parsed = JSON.parse(data);

        if (parsed.id) id = parsed.id;
        if (parsed.model) model = parsed.model;
        if (parsed.created) created = parsed.created;
        if (parsed.citations) citations = parsed.citations;
        if (parsed.usage) usage = parsed.usage;

        const delta = parsed.choices?.[0]?.delta;
        if (delta?.content) {
          contentParts.push(delta.content);
        }
      } catch {
        // Skip malformed JSON chunks (e.g. keep-alive pings)
      }
    }
  }

  const assembled: ChatCompletionResponse = {
    choices: [
      {
        message: { content: contentParts.join("") },
        finish_reason: "stop",
        index: 0,
      },
    ],
    ...(citations && { citations }),
    ...(usage && { usage }),
    ...(id && { id }),
    ...(model && { model }),
    ...(created && { created }),
  };

  return ChatCompletionResponseSchema.parse(assembled);
}

export async function performChatCompletion(
  messages: Message[],
  model: string = "sonar-pro",
  stripThinking: boolean = false,
  serviceOrigin?: string,
  options?: ChatCompletionOptions
): Promise<string> {
  const useStreaming = model === "sonar-deep-research";

  const body: Record<string, unknown> = {
    model: model,
    messages: messages,
    ...(useStreaming && { stream: true }),
    ...(options?.search_recency_filter && { search_recency_filter: options.search_recency_filter }),
    ...(options?.search_domain_filter && { search_domain_filter: options.search_domain_filter }),
    ...(options?.search_context_size && { web_search_options: { search_context_size: options.search_context_size } }),
    ...(options?.reasoning_effort && { reasoning_effort: options.reasoning_effort }),
  };

  const response = await makeApiRequest("chat/completions", body, serviceOrigin);

  let data: ChatCompletionResponse;
  try {
    if (useStreaming) {
      data = await consumeSSEStream(response);
    } else {
      const json = await response.json();
      data = ChatCompletionResponseSchema.parse(json);
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      const issues = error.issues;
      if (issues.some(i => i.path.includes('message') || i.path.includes('content'))) {
        throw new Error("Invalid API response: missing message content");
      }
      if (issues.some(i => i.path.includes('choices'))) {
        throw new Error("Invalid API response: missing or empty choices array");
      }
    }
    throw new Error(`Failed to parse JSON response from Perplexity API: ${error}`);
  }

  const firstChoice = data.choices[0];

  let messageContent = firstChoice.message.content;

  if (stripThinking) {
    messageContent = stripThinkingTokens(messageContent);
  }

  if (data.citations && Array.isArray(data.citations) && data.citations.length > 0) {
    messageContent += "\n\nCitations:\n";
    data.citations.forEach((citation, index) => {
      messageContent += `[${index + 1}] ${citation}\n`;
    });
  }

  return messageContent;
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
    formattedResults += `\n`;
  });

  return formattedResults;
}

export async function performSearch(
  query: string,
  maxResults: number = 10,
  maxTokensPerPage: number = 1024,
  country?: string,
  serviceOrigin?: string
): Promise<string> {
  const body: Record<string, unknown> = {
    query: query,
    max_results: maxResults,
    max_tokens_per_page: maxTokensPerPage,
    ...(country && { country }),
  };

  const response = await makeApiRequest("search", body, serviceOrigin);

  let data: SearchResponse;
  try {
    const json = await response.json();
    data = SearchResponseSchema.parse(json);
  } catch (error) {
    throw new Error(`Failed to parse JSON response from Perplexity Search API: ${error}`);
  }

  return formatSearchResults(data);
}

export function createPerplexityServer(serviceOrigin?: string) {
  const server = new McpServer(
    {
      name: "ai.perplexity/mcp-server",
      version: VERSION,
    },
    {
      instructions:
        "Perplexity AI server for web-grounded search, research, and reasoning. " +
        "\n\nTOOL TIMING (critical for Claude.ai web — has a hard 60s tool-call timeout):\n" +
        "- perplexity_search: 1-3s. Quick web search returning ranked results (title/URL/snippet/date). No AI synthesis.\n" +
        "- perplexity_ask: 5-15s. Sonar Pro conversational answer with citations. PREFERRED for time-sensitive queries.\n" +
        "- perplexity_reason: 10-30s. Sonar Reasoning Pro with chain-of-thought. May approach 60s on complex prompts.\n" +
        "- perplexity_research (sync): 30-180s. Sonar Deep Research. WILL TIME OUT in Claude.ai web — use ONLY in Claude Desktop / Claude Code / terminal MCP clients.\n" +
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
  
  const stripThinkingField = z.boolean().optional()
    .describe("If true, removes <think>...</think> tags and their content from the response to save context tokens. Default is false.");
  
  const searchRecencyFilterField = z.enum(["hour", "day", "week", "month", "year"]).optional()
    .describe("Filter search results by recency. Use 'hour' for very recent news, 'day' for today's updates, 'week' for this week, etc.");
  
  const searchDomainFilterField = z.array(z.string()).optional()
    .describe("Restrict search results to specific domains (e.g., ['wikipedia.org', 'arxiv.org']). Use '-' prefix for exclusion (e.g., ['-reddit.com']).");
  
  const searchContextSizeField = z.enum(["low", "medium", "high"]).optional()
    .describe("Controls how much web context is retrieved. 'low' (default) is fastest, 'high' provides more comprehensive results.");
  
  const reasoningEffortField = z.enum(["minimal", "low", "medium", "high"]).optional()
    .describe("Controls depth of deep research reasoning. Higher values produce more thorough analysis.");
  
  const responseOutputSchema = {
    response: z.string().describe("AI-generated text response with numbered citation references"),
  };

  // Input schemas
  const messagesOnlyInputSchema = { 
    messages: messagesField,
    search_recency_filter: searchRecencyFilterField,
    search_domain_filter: searchDomainFilterField,
    search_context_size: searchContextSizeField,
  };
  const messagesWithStripThinkingInputSchema = { 
    messages: messagesField, 
    strip_thinking: stripThinkingField,
    search_recency_filter: searchRecencyFilterField,
    search_domain_filter: searchDomainFilterField,
    search_context_size: searchContextSizeField,
  };
  const researchInputSchema = {
    messages: messagesField,
    strip_thinking: stripThinkingField,
    reasoning_effort: reasoningEffortField,
  };

  server.registerTool(
    "perplexity_ask",
    {
      title: "Ask Perplexity",
      description: "Answer a question using web-grounded AI (Sonar Pro model). " +
        "Latency: 5-15s typical. SAFE for Claude.ai web (60s client cap). " +
        "Best for: quick factual questions, summaries, explanations, general Q&A, and " +
        "MULTI-TOPIC SUMMARIES that would otherwise need slow research. " +
        "Returns a text response with numbered citations. Fast and cheap. " +
        "Supports filtering by recency (hour/day/week/month/year), domain restrictions, and search context size. " +
        "For deep multi-source research from a timeout-tolerant client, use perplexity_research. " +
        "For deep research from Claude.ai web (60s cap), use start_research + check_research instead. " +
        "For step-by-step reasoning, use perplexity_reason.",
      inputSchema: messagesOnlyInputSchema as any,
      outputSchema: responseOutputSchema as any,
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
        idempotentHint: false,
        destructiveHint: false,
      },
    },
    async (args: any) => {
      const { messages, search_recency_filter, search_domain_filter, search_context_size } = args as { 
        messages: Message[];
        search_recency_filter?: "hour" | "day" | "week" | "month" | "year";
        search_domain_filter?: string[];
        search_context_size?: "low" | "medium" | "high";
      };
      validateMessages(messages, "perplexity_ask");
      const options = {
        ...(search_recency_filter && { search_recency_filter }),
        ...(search_domain_filter && { search_domain_filter }),
        ...(search_context_size && { search_context_size }),
      };
      const result = await performChatCompletion(messages, "sonar-pro", false, serviceOrigin, Object.keys(options).length > 0 ? options : undefined);
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
      description: "Conduct deep, multi-source research on a topic (Sonar Deep Research model). " +
        "Latency: 30-180+ seconds. ⚠️ WILL TIME OUT in Claude.ai web (60s hard client cap). " +
        "Use ONLY in Claude Desktop, Claude Code, terminal MCP clients, or other " +
        "clients with >180s tool-call tolerance. " +
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
    async (args: any) => {
      const { messages, strip_thinking, reasoning_effort } = args as { 
        messages: Message[];
        strip_thinking?: boolean;
        reasoning_effort?: "minimal" | "low" | "medium" | "high";
      };
      validateMessages(messages, "perplexity_research");
      const stripThinking = typeof strip_thinking === "boolean" ? strip_thinking : false;
      const options = {
        ...(reasoning_effort && { reasoning_effort }),
      };
      const result = await performChatCompletion(messages, "sonar-deep-research", stripThinking, serviceOrigin, Object.keys(options).length > 0 ? options : undefined);
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
      description: "Analyze a question using step-by-step reasoning with web grounding (Sonar Reasoning Pro model). " +
        "Latency: 10-30s typical, can approach 60s on complex prompts (risk of Claude.ai web timeout). " +
        "Best for: math, logic, comparisons, complex arguments, and tasks requiring chain-of-thought. " +
        "Returns a reasoned response with numbered citations. " +
        "Supports filtering by recency (hour/day/week/month/year), domain restrictions, and search context size. " +
        "For quick factual questions, use perplexity_ask. " +
        "For comprehensive multi-source research, use perplexity_research (sync) or start_research (async).",
      inputSchema: messagesWithStripThinkingInputSchema as any,
      outputSchema: responseOutputSchema as any,
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
        idempotentHint: false,
        destructiveHint: false,
      },
    },
    async (args: any) => {
      const { messages, strip_thinking, search_recency_filter, search_domain_filter, search_context_size } = args as { 
        messages: Message[];
        strip_thinking?: boolean;
        search_recency_filter?: "hour" | "day" | "week" | "month" | "year";
        search_domain_filter?: string[];
        search_context_size?: "low" | "medium" | "high";
      };
      validateMessages(messages, "perplexity_reason");
      const stripThinking = typeof strip_thinking === "boolean" ? strip_thinking : false;
      const options = {
        ...(search_recency_filter && { search_recency_filter }),
        ...(search_domain_filter && { search_domain_filter }),
        ...(search_context_size && { search_context_size }),
      };
      const result = await performChatCompletion(messages, "sonar-reasoning-pro", stripThinking, serviceOrigin, Object.keys(options).length > 0 ? options : undefined);
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
        "Returns formatted results (title, URL, snippet, date) — no AI synthesis. " +
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
      const { query, max_results, max_tokens_per_page, country } = args as {
        query: string;
        max_results?: number;
        max_tokens_per_page?: number;
        country?: string;
      };
      const maxResults = typeof max_results === "number" ? max_results : 10;
      const maxTokensPerPage = typeof max_tokens_per_page === "number" ? max_tokens_per_page : 1024;
      const countryCode = typeof country === "string" ? country : undefined;
      
      const result = await performSearch(query, maxResults, maxTokensPerPage, countryCode, serviceOrigin);
      return {
        content: [{ type: "text" as const, text: result }],
        structuredContent: { results: result },
      };
    }
  );

  // =====================================================================
  // FORK ADDITION: async polling pattern for sonar-deep-research.
  // Bypasses Claude.ai web's 60s hard tool-call timeout.
  // =====================================================================

  const startResearchInputSchema = {
    messages: messagesField,
    strip_thinking: stripThinkingField,
    reasoning_effort: reasoningEffortField,
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
      description: "Kick off a sonar-deep-research job and return immediately with a job_id (~1s). " +
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
      const { messages, strip_thinking, reasoning_effort } = args as {
        messages: Message[];
        strip_thinking?: boolean;
        reasoning_effort?: "minimal" | "low" | "medium" | "high";
      };
      validateMessages(messages, "start_research");

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
          const stripThinking = typeof strip_thinking === "boolean" ? strip_thinking : false;
          const options = reasoning_effort ? { reasoning_effort } : undefined;
          const result = await performChatCompletion(
            messages,
            "sonar-deep-research",
            stripThinking,
            serviceOrigin,
            options as ChatCompletionOptions | undefined,
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

  return server.server;
}

