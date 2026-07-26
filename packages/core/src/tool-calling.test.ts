import assert from "node:assert/strict";
import test from "node:test";
import {
  buildToolPrompt,
  parseToolCallsFromContent,
  validateToolingConfig,
} from "./tool-calling.js";
import type { OpenAIMessage, OpenAIToolDefinition } from "./types.js";

const WEATHER_TOOL: OpenAIToolDefinition = {
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
};

test("tool choice supports auto/none/required/specific", () => {
  const auto = validateToolingConfig([WEATHER_TOOL], undefined);
  const none = validateToolingConfig([WEATHER_TOOL], "none");
  const required = validateToolingConfig([WEATHER_TOOL], "required");
  const named = validateToolingConfig([WEATHER_TOOL], {
    type: "function",
    function: { name: "get_weather" },
  });

  assert.equal(auto?.choice.mode, "auto");
  assert.equal(none?.choice.mode, "none");
  assert.equal(required?.choice.mode, "required");
  assert.equal(named?.choice.mode, "named");
});

test("buildToolPrompt includes tool definitions and protocol", () => {
  const tooling = validateToolingConfig([WEATHER_TOOL], "auto");
  assert.ok(tooling);
  const prompt = buildToolPrompt(
    [
      { role: "user", content: "Need weather" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: '{"city":"Paris"}' },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: '{"temp":22}' },
    ] as OpenAIMessage[],
    tooling,
  );
  assert.match(prompt, /name: get_weather/);
  assert.match(prompt, /parameters_json_schema/);
  assert.match(prompt, /Assistant tool call/);
  assert.match(prompt, /Tool result \(tool_call_id=call_1\)/);
  assert.match(prompt, /<tool_calls>/);
});

test("parse one and multiple valid tool calls", () => {
  const names = new Set(["get_weather", "lookup_timezone"]);
  const single = parseToolCallsFromContent(
    '<tool_calls>{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}</tool_calls>',
    names,
  );
  const multiple = parseToolCallsFromContent(
    '<tool_calls>{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}},{"name":"lookup_timezone","arguments":{"city":"Paris"}}]}</tool_calls>',
    names,
  );

  assert.equal(single?.length, 1);
  assert.equal(single?.[0].function.arguments, '{"city":"Paris"}');
  assert.equal(multiple?.length, 2);
});

test("malformed envelope and unknown function fall back", () => {
  const names = new Set(["get_weather"]);
  const malformed = parseToolCallsFromContent("<tool_calls>not-json</tool_calls>", names);
  const unknown = parseToolCallsFromContent(
    '<tool_calls>{"tool_calls":[{"name":"not_allowed","arguments":{"x":1}}]}</tool_calls>',
    names,
  );
  const proseWrapped = parseToolCallsFromContent(
    'Here you go <tool_calls>{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}</tool_calls>',
    names,
  );

  assert.equal(malformed, null);
  assert.equal(unknown, null);
  assert.equal(proseWrapped, null);
});
