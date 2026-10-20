import { describe, it, expect, vi, afterEach } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createPerplexityServer, MAX_INFLIGHT_RESEARCH_JOBS } from "./server.js";

describe("start_research in-flight cap", () => {
  afterEach(() => vi.restoreAllMocks());

  it("refuses new jobs once the in-flight limit is reached, without calling the API", async () => {
    // Perplexity API never responds, so every started job stays in flight.
    const fetchMock = vi.fn(() => new Promise<Response>(() => {}));
    vi.stubGlobal("fetch", fetchMock);

    const server = createPerplexityServer();
    const client = new Client({ name: "test", version: "1.0.0" });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);

    const args = { messages: [{ role: "user", content: "q" }] };
    for (let i = 0; i < MAX_INFLIGHT_RESEARCH_JOBS; i++) {
      const r = await client.callTool({ name: "start_research", arguments: args });
      expect(r.isError).toBeFalsy();
    }
    const rejected = await client.callTool({ name: "start_research", arguments: args });
    expect(rejected.isError).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(MAX_INFLIGHT_RESEARCH_JOBS);

    await client.close();
    vi.unstubAllGlobals();
  });
});
