import { describe, it, expect, afterEach } from "vitest";
import express from "express";
import { Server } from "http";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { requireBearerToken } from "./auth.js";

describe("requireBearerToken", () => {
  let httpServer: Server | undefined;

  afterEach(async () => {
    if (httpServer) await new Promise<void>((r) => httpServer!.close(() => r()));
    httpServer = undefined;
  });

  async function start(token: string | undefined): Promise<string> {
    const app = express();
    app.use("/mcp", requireBearerToken(token));
    app.all("/mcp", (_req, res) => { res.json({ ok: true }); });
    httpServer = app.listen(0);
    const addr = httpServer.address();
    return `http://localhost:${typeof addr === "object" && addr ? addr.port : 0}/mcp`;
  }

  it.each([undefined, "", "   "])("refuses to build without a token (%j)", (token) => {
    expect(() => requireBearerToken(token)).toThrow("MCP_AUTH_TOKEN is required");
  });

  it.each([undefined, "", "   "])("http server refuses to start without MCP_AUTH_TOKEN (%j)", (token) => {
    const env: NodeJS.ProcessEnv = { ...process.env, PERPLEXITY_API_KEY: "test-api-key", PORT: "0" };
    delete env.MCP_AUTH_TOKEN;
    if (token !== undefined) env.MCP_AUTH_TOKEN = token;
    const tsx = fileURLToPath(new URL("../node_modules/.bin/tsx", import.meta.url));
    const httpTs = fileURLToPath(new URL("./http.ts", import.meta.url));
    const r = spawnSync(tsx, [httpTs], { env, encoding: "utf8", timeout: 15000 });
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("MCP_AUTH_TOKEN environment variable is required");
  }, 20000);

  it("rejects requests without Authorization when a token is configured", async () => {
    const url = await start("s3cret");
    const res = await fetch(url, { method: "POST" });
    expect(res.status).toBe(401);
  });

  it("rejects requests with a wrong token", async () => {
    const url = await start("s3cret");
    const res = await fetch(url, { method: "POST", headers: { Authorization: "Bearer nope" } });
    expect(res.status).toBe(401);
  });

  it("accepts requests with the correct bearer token", async () => {
    const url = await start("s3cret");
    const res = await fetch(url, { method: "POST", headers: { Authorization: "Bearer s3cret" } });
    expect(res.status).toBe(200);
  });
});
