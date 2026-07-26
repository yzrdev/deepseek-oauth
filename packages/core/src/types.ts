export interface DeepSeekSession {
  accessToken: string;
  cookies: Record<string, string>;
  userAgent: string;
  capturedAt: number;
}

export interface PoWChallenge {
  algorithm: string;
  challenge: string;
  salt: string;
  expire_at: number;
  difficulty: number;
  signature: string;
  target_path: string;
}

export interface PoWResponse {
  algorithm: string;
  challenge: string;
  salt: string;
  answer: number;
  signature: string;
  target_path: string;
}

export interface DeepSeekChatSession {
  id: string;
}

export interface SSESnapshot {
  response: {
    fragments: Array<{
      type: string;
      content: unknown;
    }>;
  };
}

export interface SSEPatch {
  p?: string;
  o?: "APPEND" | "REPLACE";
  v?: unknown;
}

export interface DeepSeekCompletionBody {
  chat_session_id: string;
  parent_message_id: number | null;
  prompt: string;
  ref_file_ids: string[];
  thinking_enabled: boolean;
  search_enabled: boolean;
  action: null;
  preempt: boolean;
  model_type?: string;
}

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAIToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
}

export type OpenAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | OpenAIContentPart[] | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  name?: string;
}

export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  tools?: OpenAIToolDefinition[];
  tool_choice?: OpenAIToolChoice;
}

export type OpenAIToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

export type OpenAIFinishReason = "stop" | "length" | "content_filter" | "tool_calls";

export interface OpenAIChatChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: "function";
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: OpenAIFinishReason | null;
  }>;
}

export interface DeepSeekCredentials {
  getSession(): Promise<DeepSeekSession>;
}
