import { describe, it, expect, afterEach } from "vitest";
import express from "express";
import { Server } from "http";
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

  it("allows all requests when no token is configured", async () => {
    const url = await start(undefined);
    expect((await fetch(url, { method: "POST" })).status).toBe(200);
  });

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
