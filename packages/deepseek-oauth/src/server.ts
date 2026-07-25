import { type IncomingMessage, type ServerResponse, createServer } from "node:http";
import { createDeepSeekTransport, deleteChatSession } from "@deepseek-oauth/core";
import { LoginRequired, deepSeekCredentials } from "@deepseek-oauth/local";
import { readBody, sendJson, sendText } from "./shared.js";

const DEBUG = !!process.env.DEBUG_DEEPSEEK;
const SHUTDOWN_TIMEOUT_MS = 30_000;
const SESSION_TTL_MS = 30 * 60 * 1000;

function debug(...args: unknown[]) {
  if (DEBUG) console.error("[deepseek-oauth-server]", ...args);
}

export interface ServerOptions {
  host: string;
  port: number;
  onShutdown?: () => Promise<void>;
}

export interface ServerInstance {
  closed(): Promise<void>;
  close(): void;
}

export async function startServer(options: ServerOptions): Promise<ServerInstance> {
  const credentials = deepSeekCredentials();
  const transport = createDeepSeekTransport(credentials);
  const sessions = new Map<string, string>();
  const sessionTimestamps = new Map<string, number>();
  let activeRequests = 0;
  let shuttingDown = false;

  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    if (shuttingDown) {
      sendText(res, 503, "Server is shutting down");
      return;
    }
    activeRequests++;
    try {
      await handleRequest(req, res, transport, sessions, sessionTimestamps);
    } catch (e) {
      if (e instanceof LoginRequired) {
        sendText(res, 401, "Not signed in to DeepSeek. Run `deepseek-oauth login` first.");
        return;
      }
      debug("Request error:", e instanceof Error ? e.message : String(e));
      sendText(res, 500, "Internal server error");
    } finally {
      activeRequests--;
    }
  });

  server.requestTimeout = 300_000;
  server.headersTimeout = 60_000;

  const sessionCleanupTimer = setInterval(() => {
    const now = Date.now();
    for (const [key, ts] of sessionTimestamps) {
      if (now - ts > SESSION_TTL_MS) {
        sessions.delete(key);
        sessionTimestamps.delete(key);
      }
    }
  }, 60_000).unref();

  await new Promise<void>((resolve, reject) => {
    server.on("error", reject);
    server.listen(options.port, options.host, () => {
      server.removeListener("error", reject);
      resolve();
    });
  });

  let closedResolve: () => void;
  let closedReject: (err: Error) => void;
  const closedPromise = new Promise<void>((resolve, reject) => {
    closedResolve = resolve;
    closedReject = reject;
  });

  server.on("error", (err: Error) => {
    closedReject(err);
  });

  server.on("close", () => {
    clearInterval(sessionCleanupTimer);
    closedResolve?.();
  });

  let closed = false;

  const doClose = () => {
    if (closed) return;
    closed = true;
    server.close();
  };

  const performShutdown = async () => {
    if (shuttingDown) {
      debug("Forced exit on second signal");
      process.exit(1);
    }
    shuttingDown = true;
    debug("Shutting down gracefully...");
    console.error("\nShutting down...");

    doClose();

    const start = Date.now();
    while (activeRequests > 0 && Date.now() - start < SHUTDOWN_TIMEOUT_MS) {
      debug(`Draining ${activeRequests} in-flight request(s)...`);
      await new Promise((r) => setTimeout(r, 250));
    }

    if (activeRequests > 0) {
      debug(`Force-closing with ${activeRequests} request(s) still in-flight`);
      if (
        typeof (server as unknown as Record<string, unknown>).closeAllConnections === "function"
      ) {
        (server as unknown as { closeAllConnections(): void }).closeAllConnections();
      }
    }

    try {
      const session = await credentials.getSession();
      for (const chatSessionId of sessions.values()) {
        try {
          await deleteChatSession(session, chatSessionId);
          debug(`Deleted chat session ${chatSessionId}`);
        } catch {
          debug(`Failed to delete chat session ${chatSessionId}`);
        }
      }
    } catch {
      debug("Could not delete chat sessions (credentials unavailable)");
    }

    if (options.onShutdown) {
      try {
        await options.onShutdown();
      } catch {
        // ignore
      }
    }

    debug("Shutdown complete.");
  };

  process.on("SIGINT", () => {
    performShutdown();
  });
  process.on("SIGTERM", () => {
    performShutdown();
  });

  return {
    async closed() {
      await closedPromise;
    },
    close() {
      performShutdown();
    },
  };
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  transport: ReturnType<typeof createDeepSeekTransport>,
  sessions: Map<string, string>,
  sessionTimestamps: Map<string, number>,
): Promise<void> {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const path = url.pathname;

  if (path === "/health" || path === "/v1/health") {
    sendJson(res, 200, { status: "ok" });
    return;
  }

  if (path === "/v1/models" || path === "/models") {
    const response = await transport.fetch(new Request(`http://localhost${path}`));
    const data = await response.json();
    sendJson(res, response.status, data);
    return;
  }

  if (path === "/v1/chat/completions" || path === "/chat/completions") {
    if (req.method !== "POST") {
      sendText(res, 405, "Method not allowed");
      return;
    }

    const body = await readBody(req);
    const sessionKey = req.socket.remoteAddress ?? "127.0.0.1";
    const existingSessionId =
      (req.headers["x-deepseek-chat-session-id"] as string) ||
      sessions.get(sessionKey) ||
      undefined;

    const requestHeaders: Record<string, string> = { "content-type": "application/json" };
    if (existingSessionId) {
      requestHeaders["x-deepseek-chat-session-id"] = existingSessionId;
    }

    const response = await transport.fetch(
      new Request(`http://localhost${path}`, {
        method: "POST",
        headers: requestHeaders,
        body,
      }),
    );

    const responseSessionId = response.headers.get("x-deepseek-chat-session-id");
    if (responseSessionId) {
      sessions.set(sessionKey, responseSessionId);
      sessionTimestamps.set(sessionKey, Date.now());
    }

    if (response.headers.get("content-type")?.includes("text/event-stream")) {
      const outHeaders: Record<string, string> = {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      };
      if (responseSessionId) {
        outHeaders["x-deepseek-chat-session-id"] = responseSessionId;
      }
      res.writeHead(response.status, outHeaders);

      if (!response.body) {
        sendText(res, 500, "No response body from DeepSeek");
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let totalChunks = 0;

      debug("piping SSE stream to client");

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value, { stream: true });
          const ok = res.write(text);
          totalChunks++;
          debug("piped chunk", totalChunks, "length:", text.length);
          if (!ok) {
            await new Promise<void>((resolve) => res.once("drain", resolve));
          }
        }
      } finally {
        debug("SSE stream complete, total chunks:", totalChunks);
        res.end();
      }
    } else {
      const jsonHeaders: Record<string, string> = { "content-type": "application/json" };
      if (responseSessionId) {
        jsonHeaders["x-deepseek-chat-session-id"] = responseSessionId;
      }
      const data = await response.json();
      res.writeHead(response.status, jsonHeaders);
      res.end(JSON.stringify(data));
    }
    return;
  }

  sendText(res, 404, `Not found: ${path}`);
}
