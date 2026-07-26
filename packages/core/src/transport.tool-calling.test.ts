import assert from "node:assert/strict";
import test from "node:test";
import { deepSeekHashV1 } from "./pow.js";
import { createDeepSeekTransport } from "./transport.js";
import type { DeepSeekCredentials, OpenAIChatRequest, OpenAIToolDefinition } from "./types.js";

const TOOL: OpenAIToolDefinition = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get weather for city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
};

const READ_TOOL: OpenAIToolDefinition = {
  type: "function",
  function: {
    name: "Read",
    description: "Read a file",
    parameters: {
      type: "object",
      properties: { file_path: { type: "string" } },
      required: ["file_path"],
    },
  },
};

function createSolvableChallenge() {
  const salt = "salt";
  const expire_at = Date.now() + 60_000;
  const answer = 0;
  const input = new TextEncoder().encode(`${salt}_${expire_at}_${answer}`);
  const challenge = Buffer.from(deepSeekHashV1(input)).toString("hex");
  return {
    algorithm: "DeepSeekHashV1",
    challenge,
    salt,
    expire_at,
    difficulty: answer,
    signature: "sig",
    target_path: "/api/v0/chat/completion",
  };
}

function jsonResponse(data: unknown): Response {
  return new Response(JSON.stringify(data), { headers: { "content-type": "application/json" } });
}

function sseResponse(content: string): Response {
  const events = [
    `data: ${JSON.stringify({ response_message_id: 7 })}`,
    `data: ${JSON.stringify({
      p: "response/fragments",
      o: "APPEND",
      v: [{ type: "RESPONSE", content }],
    })}`,
    `data: ${JSON.stringify({ p: "response/status", v: "FINISHED" })}`,
    "data: [DONE]",
    "",
  ].join("\n");
  return new Response(events, { headers: { "content-type": "text/event-stream" } });
}

function makeTransportWithMock(options: {
  completionContent: string;
  capturePrompt?: (prompt: string) => void;
  extractionContent?: string;
}): { transport: ReturnType<typeof createDeepSeekTransport>; apiCalls: () => number } {
  let apiCallCount = 0;
  let completionCallIndex = 0;

  const credentials: DeepSeekCredentials = {
    async getSession() {
      return {
        accessToken: "token",
        cookies: { user_token: "token" },
        userAgent: "test-agent",
        capturedAt: Date.now(),
      };
    },
  };

  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    apiCallCount++;
    const url = String(input);
    if (url.endsWith("/api/v0/chat/create_pow_challenge")) {
      return jsonResponse({
        code: 0,
        data: { biz_data: { challenge: createSolvableChallenge() } },
      });
    }
    if (url.endsWith("/api/v0/chat_session/create")) {
      return jsonResponse({ code: 0, data: { biz_data: { chat_session: { id: "session-1" } } } });
    }
    if (url.endsWith("/api/v0/chat/completion")) {
      completionCallIndex++;
      if (completionCallIndex === 2) {
        return sseResponse(options.extractionContent ?? '{"tool_calls":[]}');
      }
      const bodyText = typeof init?.body === "string" ? init.body : "";
      const payload = JSON.parse(bodyText) as { prompt: string };
      options.capturePrompt?.(payload.prompt);
      return sseResponse(options.completionContent);
    }
    throw new Error(`Unexpected fetch url: ${url}`);
  }) as typeof fetch;

  return {
    transport: createDeepSeekTransport(credentials),
    apiCalls: () => apiCallCount,
  };
}

async function requestCompletion(
  transport: ReturnType<typeof createDeepSeekTransport>,
  body: OpenAIChatRequest,
  headers: Record<string, string> = {},
): Promise<Response> {
  return transport.fetch(
    new Request("http://deepseek-oauth.local/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(body),
    }),
  );
}

test("invalid named tool choice returns OpenAI-style 400", { concurrency: false }, async () => {
  const { transport } = makeTransportWithMock({ completionContent: "unused" });
  const response = await requestCompletion(transport, {
    model: "deepseek-chat",
    stream: false,
    messages: [{ role: "user", content: "Hi" }],
    tools: [TOOL],
    tool_choice: { type: "function", function: { name: "missing_tool" } },
  });
  const data = (await response.json()) as { error: { type: string; param: string } };
  assert.equal(response.status, 400);
  assert.equal(data.error.type, "invalid_request_error");
  assert.equal(data.error.param, "tool_choice");
});

test(
  "non-streaming returns tool_calls and finish_reason tool_calls",
  { concurrency: false },
  async () => {
    const { transport, apiCalls } = makeTransportWithMock({
      completionContent:
        '<tool_calls>{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}</tool_calls>',
    });
    const response = await requestCompletion(transport, {
      model: "deepseek-chat",
      stream: false,
      messages: [{ role: "user", content: "Weather in Paris?" }],
      tools: [TOOL],
    });
    const data = (await response.json()) as {
      choices: Array<{
        finish_reason: string;
        message: { content: string | null; tool_calls?: Array<{ function: { name: string } }> };
      }>;
    };
    assert.equal(data.choices[0].finish_reason, "tool_calls");
    assert.equal(data.choices[0].message.content, null);
    assert.equal(data.choices[0].message.tool_calls?.[0].function.name, "get_weather");
    assert.equal(apiCalls(), 3); // session create, PoW, completion — no extraction
  },
);

test(
  "streaming emits delta.tool_calls and final tool_calls finish reason",
  { concurrency: false },
  async () => {
    const { transport, apiCalls } = makeTransportWithMock({
      completionContent:
        '<tool_calls>{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}</tool_calls>',
    });
    const response = await requestCompletion(transport, {
      model: "deepseek-chat",
      stream: true,
      messages: [{ role: "user", content: "Weather in Paris?" }],
      tools: [TOOL],
    });

    const text = await response.text();
    const lines = text.split("\n").filter((line) => line.startsWith("data: "));
    const payloads = lines
      .map((line) => line.slice(6))
      .filter((line) => line !== "[DONE]")
      .map(
        (line) =>
          JSON.parse(line) as {
            choices: Array<{ delta: { tool_calls?: unknown[] }; finish_reason: string | null }>;
          },
      );

    assert.ok(payloads.some((chunk) => Array.isArray(chunk.choices[0].delta.tool_calls)));
    assert.equal(payloads[payloads.length - 1].choices[0].finish_reason, "tool_calls");
    assert.equal(apiCalls(), 3); // session create, PoW, completion — no extraction
  },
);

test(
  "tool prompt preserves assistant tool calls and tool result messages in reused session",
  { concurrency: false },
  async () => {
    let capturedPrompt = "";
    const { transport, apiCalls } = makeTransportWithMock({
      completionContent: "normal text",
      capturePrompt: (prompt) => {
        capturedPrompt = prompt;
      },
    });
    await requestCompletion(
      transport,
      {
        model: "deepseek-chat",
        stream: false,
        messages: [
          { role: "user", content: "Find weather" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_123",
                type: "function",
                function: { name: "get_weather", arguments: '{"city":"Paris"}' },
              },
            ],
          },
          { role: "tool", tool_call_id: "call_123", content: '{"temp":22}' },
          { role: "user", content: "Now summarize" },
        ],
        tools: [TOOL],
      },
      { "x-deepseek-chat-session-id": "reuse-session" },
    );

    assert.match(capturedPrompt, /Assistant tool call \(id=call_123\)/);
    assert.match(capturedPrompt, /Tool result \(tool_call_id=call_123\)/);
    assert.match(capturedPrompt, /Available tools:/);
    assert.equal(apiCalls(), 3); // session reused, PoW, primary completion, extraction completion
  },
);

test(
  "ordinary text chat remains unchanged when tools are absent",
  { concurrency: false },
  async () => {
    let capturedPrompt = "";
    const { transport, apiCalls } = makeTransportWithMock({
      completionContent: "Hello back!",
      capturePrompt: (prompt) => {
        capturedPrompt = prompt;
      },
    });
    const response = await requestCompletion(
      transport,
      {
        model: "deepseek-chat",
        stream: false,
        messages: [
          { role: "user", content: "First message" },
          { role: "assistant", content: "Ack" },
          { role: "user", content: "Latest only?" },
        ],
      },
      { "x-deepseek-chat-session-id": "reuse-session" },
    );
    const data = (await response.json()) as {
      choices: Array<{ finish_reason: string; message: { content: string } }>;
    };

    assert.equal(capturedPrompt, "User: Latest only?");
    assert.equal(data.choices[0].finish_reason, "stop");
    assert.equal(data.choices[0].message.content, "Hello back!");
    assert.equal(apiCalls(), 2); // PoW, completion — session reused, no extraction
  },
);

test(
  "LLM extraction handles non-XML tool call format",
  { concurrency: false },
  async () => {
    const { transport, apiCalls } = makeTransportWithMock({
      completionContent: "Let me check.\n\n[Read file: /src/main.ts]",
      extractionContent:
        '{"tool_calls":[{"name":"Read","arguments":{"file_path":"/src/main.ts"}}]}',
    });
    const response = await requestCompletion(transport, {
      model: "deepseek-chat",
      stream: false,
      messages: [{ role: "user", content: "Read main.ts" }],
      tools: [READ_TOOL],
    });
    const data = (await response.json()) as {
      choices: Array<{
        finish_reason: string;
        message: { content: string | null; tool_calls?: Array<{ function: { name: string } }> };
      }>;
    };
    assert.equal(data.choices[0].finish_reason, "tool_calls");
    assert.equal(data.choices[0].message.content, null);
    assert.equal(data.choices[0].message.tool_calls?.[0].function.name, "Read");
    assert.equal(apiCalls(), 4); // session create, PoW, primary completion, extraction completion
  },
);
