import { createHash, timingSafeEqual } from "node:crypto";
import type { RequestHandler } from "express";

const digest = (s: string) => createHash("sha256").update(s).digest();

// Fail-closed bearer auth: a missing/blank token is a configuration error, never "allow all".
export function requireBearerToken(token: string | undefined): RequestHandler {
  if (!token?.trim()) throw new Error("MCP_AUTH_TOKEN is required");
  const expected = digest(`Bearer ${token}`);
  return (req, res, next) => {
    if (timingSafeEqual(digest(req.headers.authorization ?? ""), expected)) return next();
    res.status(401).json({
      jsonrpc: "2.0",
      error: { code: -32001, message: "Unauthorized" },
      id: null,
    });
  };
}
