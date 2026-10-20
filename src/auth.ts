import { createHash, timingSafeEqual } from "node:crypto";
import type { RequestHandler } from "express";

const digest = (s: string) => createHash("sha256").update(s).digest();

// Opt-in bearer auth: when no token is configured, every request is allowed.
export function requireBearerToken(token: string | undefined): RequestHandler {
  if (!token) return (_req, _res, next) => next();
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
