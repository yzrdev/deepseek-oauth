# deepseek-oauth

Use DeepSeek's models through any OpenAI-compatible client. No API key needed.

## Setup

```sh
git clone https://github.com/Devlrxxh/deepseek-oauth.git
cd deepseek-oauth
npm run setup    # installs dependencies, builds, downloads Playwright Chromium
npm run link     # makes deepseek-oauth available globally
```

## Quick start

```sh
deepseek-oauth login   # open browser to sign in
deepseek-oauth serve   # start the proxy
```

Your session is stored in `~/.deepseek-oauth/auth.json` and refreshes automatically. Or set the `DEEPSEEK_TOKEN` environment variable with your token instead.

## Server flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Interface to bind to |
| `--port` | `10531` | Port to listen on |

## Endpoints

| Path | Method | Description |
|------|--------|-------------|
| `/v1/chat/completions` | POST | Chat (streaming and non-streaming) |
| `/v1/models` | GET | Available model list |
| `/health` | GET | Health check |

## Models


- `deepseek-chat`
- `deepseek-instant`
- `deepseek-v3`
- `deepseek-reasoner`
- `deepseek-expert`
- `deepseek-r1`
- `deepseek-vision`

### DeepThink reasoning and web search

Control `thinking` and `search` per request via `extra_body` (OpenAI-compatible):

```ts
const res = await openai.chat.completions.create({
  model: "deepseek-chat",
  messages: [{ role: "user", content: "What's new today?" }],
  extra_body: {
    thinking: true,   // enable DeepThink reasoning
    search: true,      // enable web search
  },
});
```

Reasoning models (`deepseek-reasoner`, etc.) have `thinking` enabled by default.

### Tool calling

The chat completions endpoint supports OpenAI-compatible `tools` and `tool_choice` in streaming and non-streaming requests. Send tool results in the next request to continue the conversation.

### Vision

Send images as `image_url` content parts. The proxy handles upload automatically:

```ts
const res = await openai.chat.completions.create({
  model: "deepseek-vision",
  messages: [{
    role: "user",
    content: [
      { type: "text", text: "What is in this image?" },
      {
        type: "image_url",
        image_url: { url: "data:image/jpeg;base64,..." }
      },
    ],
  }],
});
```

When images are present the proxy automatically switches to the vision model. Both `data:` URIs and `https://` URLs are supported.

## Using from code

```ts
import { createDeepSeekTransport } from "@deepseek-oauth/core";
import { deepSeekCredentials } from "@deepseek-oauth/local";

const transport = createDeepSeekTransport(deepSeekCredentials());

const res = await transport.fetch(
  new Request("http://deepseek-oauth.local/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "deepseek-chat",
      messages: [{ role: "user", content: "Hello!" }],
    }),
  })
);

const data = await res.json();
console.log(data.choices[0].message.content);
```

### With the OpenAI JS SDK

```ts
import OpenAI from "openai";
import { createDeepSeekTransport } from "@deepseek-oauth/core";
import { deepSeekCredentials } from "@deepseek-oauth/local";

const transport = createDeepSeekTransport(deepSeekCredentials());

const openai = new OpenAI({
  apiKey: "deepseek-oauth",
  baseURL: transport.baseURL,
  fetch: transport.fetch,
});
```

### OpenAI-style tool calling (prompt-based emulation)

`/v1/chat/completions` supports OpenAI `tools`/`tool_choice` by converting tool definitions and prior tool turns into prompt instructions for the DeepSeek web endpoint.

```ts
const result = await openai.chat.completions.create({
  model: "deepseek-chat",
  messages: [{ role: "user", content: "What's the weather in Paris?" }],
  tools: [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather for a city",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      },
    },
  ],
  tool_choice: "auto",
});

const call = result.choices[0].message.tool_calls?.[0];
// Execute tool client-side, then send a follow-up turn with role: "tool"
```

Limitations:
- This is not native DeepSeek function-calling; it is prompt-guided emulation over `chat.deepseek.com`.
- Structured tool output depends on model compliance, so malformed outputs can fall back to plain text.
- Never trust tool arguments blindly: validate and sanitize before executing tools.

## Packages

| Package | Description |
|---------|-------------|
| `deepseek-oauth` | CLI: `login` and `serve` commands |
| `@deepseek-oauth/core` | Transport, SSE parser, PoW solver, file upload, types |
| `@deepseek-oauth/local` | Browser auth (Playwright), credential storage |

---

Inspired by [openai-oauth](https://github.com/EvanZhouDev/openai-oauth).

deepseek-oauth is unofficial and not affiliated with DeepSeek. Treat your credentials like passwords. Provided as-is.
