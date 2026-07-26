import { encodePowResponse, solvePoW } from "./pow.js";
import { buildCookieHeader, buildHeaders, createChatSession } from "./session.js";
import { DeepSeekSSEParser } from "./sse.js";
import {
  buildToolPrompt,
  flattenMessages,
  hasToolHistory,
  normalizeCall,
  parseToolCalls,
  toolProtocolEnabled,
} from "./tool-calls.js";
import type {
  DeepSeekCredentials,
  DeepSeekSession,
  OpenAIChatChunk,
  OpenAIChatRequest,
  OpenAIMessage,
  OpenAIToolCall,
  OpenAIToolDefinition,
  PoWChallenge,
} from "./types.js";

const BASE_URL = "https://chat.deepseek.com";

interface ModelConfig {
  model_type: string;
  defaultThinking: boolean;
  defaultSearch: boolean;
}

const MODEL_MAP: Record<string, ModelConfig> = {
  "deepseek-chat": { model_type: "default", defaultThinking: true, defaultSearch: true },
  "deepseek-instant": { model_type: "default", defaultThinking: true, defaultSearch: true },
  "deepseek-v3": { model_type: "default", defaultThinking: true, defaultSearch: true },
  "deepseek-reasoner": { model_type: "expert", defaultThinking: true, defaultSearch: true },
  "deepseek-expert": { model_type: "expert", defaultThinking: true, defaultSearch: true },
  "deepseek-r1": { model_type: "expert", defaultThinking: true, defaultSearch: true },
  "deepseek-vision": { model_type: "vision", defaultThinking: true, defaultSearch: true },
};

function resolveModel(model: string): ModelConfig {
  return MODEL_MAP[model] ?? MODEL_MAP["deepseek-chat"];
}

function extractContent(content: string | { type: string; text?: string }[] | null): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .filter((p) => p.type === "text")
    .map((p) => p.text ?? "")
    .join("\n");
}

function lastUserMessage(messages: OpenAIMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      return extractContent(messages[i].content);
    }
  }
  return "";
}

export function createDeepSeekTransport(credentials: DeepSeekCredentials) {
  const messageIds = new Map<string, number>();

  return {
    baseURL: "https://deepseek-oauth.local/v1",
    async fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
      const request = new Request(input, init);
      const url = new URL(request.url);
      const path = url.pathname;

      if (path === "/v1/models" || path === "/models") {
        return handleModels();
      }

      if (path === "/v1/chat/completions" || path === "/chat/completions") {
        const body = JSON.parse(await request.text()) as OpenAIChatRequest;
        const existingSessionId = request.headers.get("x-deepseek-chat-session-id");
        return handleChatCompletions(body, credentials, existingSessionId, messageIds);
      }

      return new Response("Not Found", { status: 404 });
    },
  };
}

async function handleModels(): Promise<Response> {
  const ids = Object.keys(MODEL_MAP);
  const data = ids.map((id) => ({
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "deepseek",
  }));

  return new Response(JSON.stringify({ object: "list", data }), {
    headers: { "content-type": "application/json" },
  });
}

async function handleChatCompletions(
  body: OpenAIChatRequest,
  credentials: DeepSeekCredentials,
  existingSessionId?: string | null,
  messageIds?: Map<string, number>,
): Promise<Response> {
  const session = await credentials.getSession();
  const config = resolveModel(body.model);

  const raw = body as unknown as Record<string, unknown>;
  const extraBody = (raw.extra_body ?? raw.thinking_body ?? {}) as Record<string, unknown>;
  const tools = toolProtocolEnabled(body.tools, body.tool_choice) ? body.tools : undefined;
  const usesToolProtocol = tools !== undefined;

  if (tools && typeof body.tool_choice === "object" && body.tool_choice.type === "function") {
    const requestedName = body.tool_choice.function.name;
    const found = tools.some(
      (t) => t.function.name.toLowerCase() === requestedName.toLowerCase(),
    );
    if (!found) {
      return new Response(
        JSON.stringify({
          error: {
            message: `Invalid tool_choice: '${requestedName}' is not a valid tool name`,
            type: "invalid_request_error",
            param: "tool_choice",
            code: null,
          },
        }),
        { status: 400, headers: { "content-type": "application/json" } },
      );
    }
  }
  const needsFullContext = usesToolProtocol || hasToolHistory(body.messages);
  const thinking =
    extraBody.thinking !== undefined ? Boolean(extraBody.thinking) : config.defaultThinking;
  const search =
    extraBody.search !== undefined
      ? Boolean(extraBody.search)
      : usesToolProtocol
        ? false
        : config.defaultSearch;

  const isStream = body.stream !== false;

  let chatSessionId = needsFullContext ? "" : (existingSessionId ?? "");
  const isReuse = Boolean(existingSessionId) && !needsFullContext;

  const { images, hasImages } = extractImages(body.messages);
  const refFileIds: string[] = [];
  const textMessages = hasImages ? stripImageParts(body.messages) : body.messages;

  let effectiveModelType = config.model_type;

  if (hasImages) {
    effectiveModelType = "vision";
    for (const img of images) {
      const buffer = dataUriToBuffer(img.url);
      if (buffer) {
        const fileId = await uploadFile(session, buffer, "image.png", "vision");
        if (fileId) refFileIds.push(fileId);
      }
    }
  }

  let prompt: string;
  if (usesToolProtocol) {
    prompt = buildToolPrompt(textMessages, tools, body.tool_choice);
  } else if (needsFullContext) {
    prompt = flattenMessages(textMessages);
  } else if (isReuse) {
    prompt = `User: ${lastUserMessage(textMessages)}`;
  } else {
    prompt = flattenMessages(textMessages);
  }

  if (hasImages && !prompt.trim()) {
    prompt = "Describe this image.";
  }

  const [chatSession, challenge] = await Promise.all([
    isReuse ? Promise.resolve(null) : createChatSession(session),
    requestPoWChallenge(session),
  ]);

  if (chatSession) {
    chatSessionId = chatSession.id;
  }

  const powResponse = solvePoW(challenge);
  const powEncoded = encodePowResponse(powResponse);
  const extractionArgs = tools ? { session, chatSessionId, powEncoded } : undefined;

  const parentMessageId =
    isReuse && chatSessionId ? (messageIds?.get(chatSessionId) ?? null) : null;

  const maxTokens = body.max_tokens;

  const completionBody = {
    chat_session_id: chatSessionId,
    parent_message_id: parentMessageId,
    prompt,
    ref_file_ids: refFileIds,
    thinking_enabled: thinking,
    search_enabled: search,
    action: null,
    preempt: false,
    model_type: effectiveModelType,
    ...(maxTokens != null ? { max_tokens: maxTokens } : {}),
  };

  const headers = buildHeaders(session);
  headers.cookie = buildCookieHeader(session.cookies);
  headers["x-ds-pow-response"] = powEncoded;

  const response = await fetch(`${BASE_URL}/api/v0/chat/completion`, {
    method: "POST",
    headers,
    body: JSON.stringify(completionBody),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`DeepSeek completion failed: ${response.status} ${text}`);
  }

  if (!response.body) {
    throw new Error("No response body from DeepSeek");
  }

  let result: Response;
  if (isStream) {
    result = usesToolProtocol
      ? await handleToolStreamingResponse(response, body.model, chatSessionId, tools, messageIds, extractionArgs)
      : await handleStreamingResponse(response, body.model, chatSessionId, messageIds);
  } else {
    result = await handleNonStreamingResponse(
      response,
      body.model,
      chatSessionId,
      messageIds,
      prompt,
      tools,
      extractionArgs,
    );
  }

  result.headers.set("x-deepseek-chat-session-id", chatSessionId);
  return result;
}

interface ExtractedImage {
  url: string;
}

function extractImages(messages: OpenAIMessage[]): {
  images: ExtractedImage[];
  hasImages: boolean;
} {
  const images: ExtractedImage[] = [];
  for (const msg of messages) {
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === "image_url" && part.image_url?.url) {
          images.push({ url: part.image_url.url });
        }
      }
    }
  }
  return { images, hasImages: images.length > 0 };
}

function stripImageParts(messages: OpenAIMessage[]): OpenAIMessage[] {
  return messages.map((msg) => {
    if (Array.isArray(msg.content)) {
      const textParts = msg.content.filter((p) => p.type !== "image_url");
      if (textParts.length === 0) return msg;
      return { ...msg, content: textParts };
    }
    return msg;
  });
}

function dataUriToBuffer(dataUri: string): Uint8Array | null {
  if (dataUri.startsWith("data:")) {
    const commaPos = dataUri.indexOf(",");
    if (commaPos === -1) return null;
    const base64 = dataUri.slice(commaPos + 1);
    try {
      return new Uint8Array(Buffer.from(base64, "base64"));
    } catch {
      return null;
    }
  }
  return null;
}

async function uploadFile(
  session: DeepSeekSession,
  fileBuffer: Uint8Array,
  fileName: string,
  modelType: string,
): Promise<string | null> {
  const challenge = await requestPoWChallengeForTarget(session, "/api/v0/file/upload_file");
  const powResponse = solvePoW(challenge);
  const powEncoded = encodePowResponse(powResponse);

  const boundary = `--deepseek-upload-${Date.now()}`;
  const header = `\r\n--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${fileName}"\r\nContent-Type: image/png\r\n\r\n`;
  const footer = `\r\n--${boundary}--`;

  const headerBytes = new TextEncoder().encode(header);
  const footerBytes = new TextEncoder().encode(footer);
  const body = new Uint8Array(headerBytes.length + fileBuffer.length + footerBytes.length);
  body.set(headerBytes, 0);
  body.set(fileBuffer, headerBytes.length);
  body.set(footerBytes, headerBytes.length + fileBuffer.length);

  const headers = buildHeaders(session);
  headers.cookie = buildCookieHeader(session.cookies, session.accessToken);
  headers["content-type"] = `multipart/form-data; boundary=${boundary}`;
  headers["x-ds-pow-response"] = powEncoded;
  headers["x-thinking-enabled"] = "0";
  headers["x-model-type"] = modelType;
  headers["x-file-size"] = String(fileBuffer.length);

  try {
    const response = await fetch(`${BASE_URL}/api/v0/file/upload_file`, {
      method: "POST",
      headers,
      body,
    });
    if (!response.ok) return null;
    const data = (await response.json()) as {
      code: number;
      data: { biz_data: { id: string; status: string } };
    };
    if (data.code !== 0) return null;
    const fileId = data.data.biz_data.id;

    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 500));
      const pollHeaders = buildHeaders(session);
      pollHeaders.cookie = buildCookieHeader(session.cookies, session.accessToken);
      const pollRes = await fetch(`${BASE_URL}/api/v0/file/fetch_files?file_ids=${fileId}`, {
        headers: pollHeaders,
      });
      if (!pollRes.ok) continue;
      const pollData = (await pollRes.json()) as {
        code: number;
        data: { biz_data: { files: Array<{ id: string; status: string }> } };
      };
      if (pollData.code !== 0) continue;
      const file = pollData.data.biz_data.files[0];
      if (file && file.status !== "PENDING" && file.status !== "PARSING") {
        if (file.status === "SUCCESS") return fileId;
        return null;
      }
    }
    return null;
  } catch {
    return null;
  }
}

async function requestPoWChallengeForTarget(
  session: DeepSeekSession,
  targetPath: string,
): Promise<PoWChallenge> {
  const headers = buildHeaders(session);
  headers.cookie = buildCookieHeader(session.cookies, session.accessToken);

  const response = await fetch(`${BASE_URL}/api/v0/chat/create_pow_challenge`, {
    method: "POST",
    headers,
    body: JSON.stringify({ target_path: targetPath }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`PoW challenge request failed: ${response.status} ${text}`);
  }

  const data = (await response.json()) as {
    code: number;
    data: { biz_data: { challenge: PoWChallenge } };
  };

  if (data.code !== 0) {
    throw new Error(`PoW challenge failed: code ${data.code}`);
  }

  return data.data.biz_data.challenge;
}

async function requestPoWChallenge(session: DeepSeekSession): Promise<PoWChallenge> {
  return requestPoWChallengeForTarget(session, "/api/v0/chat/completion");
}

const DEBUG = !!process.env.DEBUG_DEEPSEEK;

function debug(...args: unknown[]) {
  if (DEBUG) console.error("[deepseek-oauth]", ...args);
}

async function handleStreamingResponse(
  deepseekResponse: Response,
  model: string,
  chatSessionId: string,
  messageIds?: Map<string, number>,
): Promise<Response> {
  if (!deepseekResponse.body) {
    throw new Error("No response body from DeepSeek");
  }
  const reader = deepseekResponse.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const id = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  debug("stream start, id:", id);

  const stream = new ReadableStream({
    async start(controller) {
      let streamStarted = false;
      let streamFinished = false;
      let totalBytes = 0;
      let contentBuffer = "";
      let reasoningBuffer = "";
      let lastFlushTime = Date.now();

      let streamClosed = false;

      const closeStream = () => {
        if (streamClosed) return;
        streamClosed = true;
        if (!streamStarted) {
          streamStarted = true;
          const chunk: OpenAIChatChunk = {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
          };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
        }
        const final: OpenAIChatChunk = {
          id,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(final)}\n\n`));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      };

      const parser = new DeepSeekSSEParser((content, reasoning, done, msgId) => {
        if (msgId != null && messageIds) {
          messageIds.set(chatSessionId, msgId);
        }

        if (!streamStarted) {
          streamStarted = true;
          const chunk: OpenAIChatChunk = {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
          };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
        }

        contentBuffer += content;
        reasoningBuffer += reasoning;

        const hasPending = contentBuffer.length > 0 || reasoningBuffer.length > 0;
        const shouldFlush =
          done ||
          (hasPending &&
            (contentBuffer.length > 20 ||
              reasoningBuffer.length > 20 ||
              Date.now() - lastFlushTime > 50));
        if (shouldFlush) {
          if (hasPending) {
            const delta: OpenAIChatChunk["choices"][0]["delta"] = {};
            if (contentBuffer) delta.content = contentBuffer;
            if (reasoningBuffer) delta.reasoning_content = reasoningBuffer;
            const chunk: OpenAIChatChunk = {
              id,
              object: "chat.completion.chunk",
              created,
              model,
              choices: [{ index: 0, delta, finish_reason: null }],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            contentBuffer = "";
            reasoningBuffer = "";
            lastFlushTime = Date.now();
          }
        }

        if (done) {
          streamFinished = true;
          debug("stream done by parser, id:", id);
          closeStream();
        }
      });

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          totalBytes += value.length;
          parser.feed(decoder.decode(value, { stream: true }));
        }
        debug("reader exhausted, total bytes:", totalBytes, "id:", id);
        parser.flush();
      } catch (e) {
        debug("stream error:", e, "id:", id);
        if (!streamClosed) {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                error: {
                  message: e instanceof Error ? e.message : "Stream error",
                  type: "server_error",
                },
              })}\n\n`,
            ),
          );
        }
      }

      if (!streamFinished) {
        debug("stream fallback close, id:", id);
        closeStream();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    },
  });
}
async function handleToolStreamingResponse(
  deepseekResponse: Response,
  model: string,
  chatSessionId: string,
  tools: OpenAIToolDefinition[],
  messageIds: Map<string, number> | undefined,
  extractionArgs:
    | { session: DeepSeekSession; chatSessionId: string; powEncoded: string }
    | undefined,
): Promise<Response> {
  if (!deepseekResponse.body) {
    throw new Error("No response body from DeepSeek");
  }

  const reader = deepseekResponse.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const id = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  const stream = new ReadableStream({
    async start(controller) {
      const enqueue = (chunk: OpenAIChatChunk) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      };
      enqueue({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
      });

      let fullContent = "";
      let fullReasoning = "";
      const parser = new DeepSeekSSEParser((content, reasoning, _done, msgId) => {
        fullContent += content;
        fullReasoning += reasoning;
        if (msgId != null && messageIds) messageIds.set(chatSessionId, msgId);
      });

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          parser.feed(decoder.decode(value, { stream: true }));
        }
        parser.flush();
      } catch (error) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              error: {
                message: error instanceof Error ? error.message : "Stream error",
                type: "server_error",
              },
            })}\n\n`,
          ),
        );
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        return;
      }

      let toolCalls = parseToolCalls(fullContent, tools) ?? parseToolCalls(fullReasoning, tools);
      if (!toolCalls && extractionArgs) {
        const allContent = [fullContent, fullReasoning].filter(Boolean).join("\n");
        toolCalls = await llmExtractToolCalls(
          allContent, tools,
          extractionArgs.session, extractionArgs.chatSessionId, extractionArgs.powEncoded,
        );
      }
      const delta: OpenAIChatChunk["choices"][0]["delta"] = {};
      if (fullReasoning) delta.reasoning_content = fullReasoning;
      if (toolCalls) {
        delta.tool_calls = toolCalls.map((call, index) => ({
          index,
          id: call.id,
          type: call.type,
          function: call.function,
        }));
      } else if (fullContent) {
        delta.content = fullContent;
      }

      if (Object.keys(delta).length > 0) {
        enqueue({
          id,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [{ index: 0, delta, finish_reason: null }],
        });
      }

      enqueue({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: toolCalls ? "tool_calls" : "stop" }],
      });
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    },
  });
}
async function handleNonStreamingResponse(
  deepseekResponse: Response,
  model: string,
  chatSessionId: string,
  messageIds: Map<string, number> | undefined,
  prompt = "",
  tools: OpenAIToolDefinition[] | undefined,
  extractionArgs:
    | { session: DeepSeekSession; chatSessionId: string; powEncoded: string }
    | undefined,
): Promise<Response> {
  if (!deepseekResponse.body) {
    throw new Error("No response body from DeepSeek");
  }
  const reader = deepseekResponse.body.getReader();
  const decoder = new TextDecoder();
  const id = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  let fullContent = "";
  let fullReasoning = "";

  const parser = new DeepSeekSSEParser((content, reasoning, _done, msgId) => {
    if (content) fullContent += content;
    if (reasoning) fullReasoning += reasoning;
    if (msgId != null && messageIds) {
      messageIds.set(chatSessionId, msgId);
    }
  });

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.feed(decoder.decode(value, { stream: true }));
  }
  parser.flush();

  let toolCalls = tools
    ? (parseToolCalls(fullContent, tools) ?? parseToolCalls(fullReasoning, tools))
    : null;
  if (!toolCalls && tools && extractionArgs) {
    const allContent = [fullContent, fullReasoning].filter(Boolean).join("\n");
    toolCalls = await llmExtractToolCalls(
      allContent, tools,
      extractionArgs.session, extractionArgs.chatSessionId, extractionArgs.powEncoded,
    );
  }
  const message: Record<string, unknown> = {
    role: "assistant",
    content: toolCalls ? null : fullContent,
  };
  if (toolCalls) message.tool_calls = toolCalls;
  if (fullReasoning) {
    message.reasoning_content = fullReasoning;
  }

  const responseBody = {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: toolCalls ? "tool_calls" : "stop",
      },
    ],
    usage: {
      prompt_tokens: Math.ceil(prompt.length / 3),
      completion_tokens: Math.ceil(fullContent.length / 3),
      total_tokens: Math.ceil((prompt.length + fullContent.length) / 3),
    },
  };

  return new Response(JSON.stringify(responseBody), {
    headers: { "content-type": "application/json" },
  });
}

const TOOL_EXTRACTION_TIMEOUT_MS = parseInt(
  process.env.DEEPSEEK_TOOL_EXTRACTION_TIMEOUT_MS ?? "10000",
  10,
);


async function llmExtractToolCalls(
  content: string,
  tools: OpenAIToolDefinition[],
  session: DeepSeekSession,
  chatSessionId: string,
  powEncoded: string,
): Promise<OpenAIToolCall[] | null> {
  if (!content.trim()) return null;

  const MAX_DESC_LENGTH = 200;
  const toolSummaries = tools.map((t) => ({
    name: t.function.name,
    parameters: t.function.parameters,
    ...(t.function.description
      ? { description: t.function.description.slice(0, MAX_DESC_LENGTH) }
      : {}),
  }));

  const extractionPrompt = [
    "You are a tool-call parser. Extract any tool invocations from the assistant response below.",
    "The response may use XML tags, markdown, JSON, YAML, or plain prose — extract tool calls regardless of format.",
    "Return ONLY a JSON object with a \"tool_calls\" array. No markdown, no explanation.",
    "",
    "Available tools:",
    JSON.stringify(toolSummaries),
    "",
    "Assistant response:",
    '"""',
    content,
    '"""',
    "",
    'Output: {"tool_calls":[{"name":"tool_name","arguments":{...}}]} or {"tool_calls":[]}',
  ].join("\n");

  const extractionBody = {
    chat_session_id: chatSessionId,
    prompt: extractionPrompt,
    thinking_enabled: false,
    search_enabled: false,
    max_tokens: 1000,
    model_type: "default",
    action: null,
    preempt: false,
    ref_file_ids: [],
  };

  const headers = buildHeaders(session);
  headers.cookie = buildCookieHeader(session.cookies);
  headers["x-ds-pow-response"] = powEncoded;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TOOL_EXTRACTION_TIMEOUT_MS);

  try {
    const response = await fetch(`${BASE_URL}/api/v0/chat/completion`, {
      method: "POST",
      headers,
      body: JSON.stringify(extractionBody),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) return null;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullContent = "";

    const parser = new DeepSeekSSEParser((ct, _reasoning, _done, _msgId) => {
      fullContent += ct;
    });

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.feed(decoder.decode(value, { stream: true }));
    }
    parser.flush();

    const jsonText = fullContent
      .trim()
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/i, "");

    const parsed = JSON.parse(jsonText) as {
      tool_calls?: Array<{ name?: string; arguments?: unknown }>;
    };

    if (!parsed?.tool_calls?.length) return null;

    const allowedNames = new Map(
      tools.map((tool) => [tool.function.name.toLowerCase(), tool.function.name]),
    );

    const calls: OpenAIToolCall[] = [];
    for (const candidate of parsed.tool_calls) {
      const normalized = normalizeCall(candidate, allowedNames);
      if (normalized) calls.push(normalized);
    }

    return calls.length > 0 ? calls : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
