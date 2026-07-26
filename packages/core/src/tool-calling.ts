import type {
  OpenAIMessage,
  OpenAIToolCall,
  OpenAIToolChoice,
  OpenAIToolDefinition,
} from "./types.js";

const TOOL_CALLS_TAG_OPEN = "<tool_calls>";
const TOOL_CALLS_TAG_CLOSE = "</tool_calls>";

export class ToolRequestError extends Error {
  constructor(
    message: string,
    public readonly param: "tools" | "tool_choice",
    public readonly code: "invalid_value" | "invalid_tool_choice",
  ) {
    super(message);
    this.name = "ToolRequestError";
  }
}

export interface ToolChoiceMode {
  mode: "auto" | "none" | "required" | "named";
  name?: string;
}

export interface ToolingConfig {
  tools: OpenAIToolDefinition[];
  choice: ToolChoiceMode;
  parseOutput: boolean;
  toolNames: Set<string>;
}

interface ParsedToolEnvelope {
  tool_calls: Array<{ name: string; arguments: unknown }>;
}

export function validateToolingConfig(
  tools: OpenAIToolDefinition[] | undefined,
  toolChoice: OpenAIToolChoice | undefined,
): ToolingConfig | null {
  const suppliedTools = tools ?? [];
  if (suppliedTools.length === 0) {
    if (toolChoice !== undefined) {
      throw new ToolRequestError(
        "`tool_choice` requires `tools` to be provided.",
        "tool_choice",
        "invalid_value",
      );
    }
    return null;
  }

  for (const tool of suppliedTools) {
    if (tool.type !== "function" || !tool.function?.name?.trim()) {
      throw new ToolRequestError(
        "Each tool must be a function with a valid name.",
        "tools",
        "invalid_value",
      );
    }
  }

  const names = new Set(suppliedTools.map((tool) => tool.function.name));
  if (names.size !== suppliedTools.length) {
    throw new ToolRequestError("Tool names must be unique.", "tools", "invalid_value");
  }

  let choice: ToolChoiceMode;
  if (toolChoice === undefined || toolChoice === "auto") {
    choice = { mode: "auto" };
  } else if (toolChoice === "none") {
    choice = { mode: "none" };
  } else if (toolChoice === "required") {
    choice = { mode: "required" };
  } else if (
    typeof toolChoice === "object" &&
    toolChoice?.type === "function" &&
    typeof toolChoice.function?.name === "string"
  ) {
    const name = toolChoice.function.name;
    if (!names.has(name)) {
      throw new ToolRequestError(
        `tool_choice.function.name "${name}" is not present in provided tools.`,
        "tool_choice",
        "invalid_tool_choice",
      );
    }
    choice = { mode: "named", name };
  } else {
    throw new ToolRequestError(
      "Unsupported `tool_choice`. Use auto, none, required, or {type:'function', function:{name}}.",
      "tool_choice",
      "invalid_value",
    );
  }

  return {
    tools: suppliedTools,
    choice,
    parseOutput: choice.mode !== "none",
    toolNames: names,
  };
}

export function hasToolConversation(messages: OpenAIMessage[]): boolean {
  return messages.some((message) => {
    if (message.role === "tool") return true;
    return message.role === "assistant" && Boolean(message.tool_calls?.length);
  });
}

export function extractContent(content: string | { type: string; text?: string }[] | null): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("\n");
}

function cleanMessages(messages: OpenAIMessage[]): OpenAIMessage[] {
  const deduped: OpenAIMessage[] = [];
  for (const message of messages) {
    if (message.role === "tool" || (message.role === "assistant" && message.tool_calls?.length)) {
      deduped.push(message);
      continue;
    }
    const normalized = message as OpenAIMessage;
    if (deduped.length > 0 && deduped[deduped.length - 1].role === normalized.role) {
      const prev = deduped[deduped.length - 1];
      const prevText = extractContent(prev.content);
      const curText = extractContent(normalized.content);
      if (curText) {
        prev.content = prevText ? `${prevText}\n\n${curText}` : curText;
      }
    } else {
      deduped.push(normalized);
    }
  }
  return deduped;
}

export function flattenMessages(messages: OpenAIMessage[]): string {
  const cleaned = cleanMessages(messages);
  return cleaned
    .map((message) => {
      const text = extractContent(message.content);
      if (message.role === "system") return text;
      if (message.role === "user") return `User: ${text}`;
      if (message.role === "assistant") {
        const blocks: string[] = [];
        if (text) blocks.push(`Assistant: ${text}`);
        if (message.tool_calls?.length) {
          for (const toolCall of message.tool_calls) {
            blocks.push(
              `Assistant tool call (id=${toolCall.id}): ${toolCall.function.name} arguments=${toolCall.function.arguments}`,
            );
          }
        }
        return blocks.join("\n");
      }
      if (message.role === "tool") {
        return `Tool result (tool_call_id=${message.tool_call_id ?? "unknown"}): ${text}`;
      }
      return text;
    })
    .filter(Boolean)
    .join("\n\n");
}

export function buildToolPrompt(messages: OpenAIMessage[], tooling: ToolingConfig): string {
  const lines: string[] = [];
  lines.push(flattenMessages(messages));
  lines.push("Available tools:");
  for (const tool of tooling.tools) {
    lines.push(`- name: ${tool.function.name}`);
    lines.push(`  description: ${tool.function.description ?? ""}`);
    lines.push(`  parameters_json_schema: ${JSON.stringify(tool.function.parameters ?? {})}`);
  }
  lines.push("Tool calling protocol:");
  lines.push("- If you call tool(s), output ONLY this envelope and nothing else:");
  lines.push(TOOL_CALLS_TAG_OPEN);
  lines.push('{"tool_calls":[{"name":"tool_name","arguments":{"key":"value"}}]}');
  lines.push(TOOL_CALLS_TAG_CLOSE);
  lines.push("- The `arguments` value must be valid JSON.");
  if (tooling.choice.mode === "none") {
    lines.push("- tool_choice is 'none': do not call tools.");
  } else if (tooling.choice.mode === "required") {
    lines.push("- tool_choice is 'required': return at least one tool call.");
  } else if (tooling.choice.mode === "named") {
    lines.push(`- tool_choice requires exactly this tool name: ${tooling.choice.name}.`);
  } else {
    lines.push("- tool_choice is 'auto': use a tool only when needed.");
  }
  return lines.filter(Boolean).join("\n");
}

export function parseToolCallsFromContent(
  content: string,
  toolNames: Set<string>,
): OpenAIToolCall[] | null {
  const start = content.indexOf(TOOL_CALLS_TAG_OPEN);
  const end = content.indexOf(TOOL_CALLS_TAG_CLOSE);
  if (start === -1 || end === -1 || end <= start) return null;

  const before = content.slice(0, start);
  const after = content.slice(end + TOOL_CALLS_TAG_CLOSE.length);
  if (before.trim() || after.trim()) return null;

  const payload = content.slice(start + TOOL_CALLS_TAG_OPEN.length, end).trim();
  if (!payload) return null;

  let parsed: ParsedToolEnvelope;
  try {
    parsed = JSON.parse(payload) as ParsedToolEnvelope;
  } catch {
    return null;
  }

  if (!Array.isArray(parsed.tool_calls) || parsed.tool_calls.length === 0) return null;

  const calls: OpenAIToolCall[] = [];
  for (const [index, call] of parsed.tool_calls.entries()) {
    if (!call || typeof call.name !== "string" || !toolNames.has(call.name)) return null;
    if (!("arguments" in call)) return null;
    const argumentsText = JSON.stringify(call.arguments);
    if (argumentsText === undefined) return null;
    calls.push({
      id: `call_${index}`,
      type: "function",
      function: { name: call.name, arguments: argumentsText },
    });
  }
  return calls;
}

export function createToolRequestErrorResponse(error: ToolRequestError): Response {
  return new Response(
    JSON.stringify({
      error: {
        message: error.message,
        type: "invalid_request_error",
        param: error.param,
        code: error.code,
      },
    }),
    {
      status: 400,
      headers: { "content-type": "application/json" },
    },
  );
}
