import type { DeepSeekChatSession, DeepSeekSession } from "./types.js";

const BASE_URL = "https://chat.deepseek.com";
const API_VERSION = "2.0.0";
const CLIENT_VERSION = "2.0.0";
const CLIENT_PLATFORM = "web";

export function buildHeaders(session: DeepSeekSession): Record<string, string> {
  return {
    authorization: `Bearer ${session.accessToken}`,
    accept: "*/*",
    "content-type": "application/json",
    "user-agent": session.userAgent,
    origin: BASE_URL,
    referer: `${BASE_URL}/`,
    "x-app-version": API_VERSION,
    "x-client-version": CLIENT_VERSION,
    "x-client-platform": CLIENT_PLATFORM,
    "x-client-locale": "en_US",
  };
}

export function buildCookieHeader(cookies: Record<string, string>, accessToken?: string): string {
  const all = { ...cookies };
  if (accessToken && !all.user_token) {
    all.user_token = accessToken;
  }
  return Object.entries(all)
    .map(([k, v]) => `${k}=${v}`)
    .join("; ");
}

export function buildCookieHeaderFromSession(session: {
  accessToken: string;
  cookies: Record<string, string>;
}): string {
  return buildCookieHeader(session.cookies, session.accessToken);
}

async function apiRequest(
  session: DeepSeekSession,
  path: string,
  body: unknown,
): Promise<Response> {
  const headers = buildHeaders(session);
  headers.cookie = buildCookieHeader(session.cookies, session.accessToken);

  const response = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`DeepSeek API error ${response.status}: ${text}`);
  }

  return response;
}

export async function createChatSession(session: DeepSeekSession): Promise<DeepSeekChatSession> {
  const response = await apiRequest(session, "/api/v0/chat_session/create", {});
  const data = (await response.json()) as {
    code: number;
    data: { biz_data: { chat_session: { id: string } } };
  };

  if (data.code !== 0) {
    throw new Error(`Failed to create chat session: code ${data.code}`);
  }

  return { id: data.data.biz_data.chat_session.id };
}

export async function deleteChatSession(
  session: DeepSeekSession,
  chatSessionId: string,
): Promise<void> {
  try {
    await apiRequest(session, "/api/v0/chat_session/delete", {
      chat_session_id: chatSessionId,
    });
  } catch {
    
  }
}
