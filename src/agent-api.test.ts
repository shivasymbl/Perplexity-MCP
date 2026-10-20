import { describe, it, expect, vi, afterEach } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createPerplexityServer } from "./server.js";

describe("start_research backend", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("runs deep research on the Agent API research preset (Sonar is deprecated)", async () => {
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) => new Promise<Response>(() => {}));
    vi.stubGlobal("fetch", fetchMock);

    const server = createPerplexityServer();
    const client = new Client({ name: "test", version: "1.0.0" });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);

    await client.callTool({ name: "start_research", arguments: { messages: [{ role: "user", content: "q" }] } });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/v1\/agent$/);
    expect(JSON.parse(String(init?.body)).preset).toBe("high");

    await client.close();
  });
});
