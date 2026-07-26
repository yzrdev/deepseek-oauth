import { parseToolCalls } from './packages/core/dist/tool-calls.js';
import { solvePoW, encodePowResponse } from './packages/core/dist/pow.js';
import { DeepSeekSSEParser } from './packages/core/dist/sse.js';
import { readFileSync } from 'node:fs';

const tools = [
  { type: 'function', function: { name: 'read_file', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } },
  { type: 'function', function: { name: 'write_file', description: 'Write a file', parameters: { type: 'object', properties: { path: { type: 'string' }, content: { type: 'string' } }, required: ['path','content'] } } },
];

let passed = 0;
let failed = 0;
function test(name, fn) {
  try { fn(); passed++; }
  catch (e) { failed++; console.error(`FAIL: ${name} — ${e.message}`); }
}

// ── Phase 1 ──
console.log('=== Phase 1: PoW & WASM ===');
test('WASM solver loads, throws on impossible challenge', () => {
  try { solvePoW({ salt:'a'.repeat(32), expire_at:9999999999, difficulty:1000, challenge:'0'.repeat(64), algorithm:'v1', signature:'x', target_path:'/api' }); throw new Error('should throw'); }
  catch (e) { if (!e.message.includes('no answer found')) throw e; }
});
test('encodePowResponse: base64 JSON round-trip', () => {
  const e = encodePowResponse({ algorithm:'v1', challenge:'ff', salt:'ss', answer:42, signature:'sg', target_path:'/api' });
  if (JSON.parse(Buffer.from(e,'base64').toString()).answer !== 42) throw new Error('wrong answer');
});
test('pow.js: deepSeekHashV1 NOT exported', async () => {
  if ('deepSeekHashV1' in await import('./packages/core/dist/pow.js')) throw new Error('still exported');
});

// ── Phase 3: parseToolCalls ──
console.log('\n=== Phase 3: parseToolCalls ===');
test('XML plural with closing tag', () => {
  const r = parseToolCalls(`<tool_calls>[{"name":"read_file","arguments":{"path":"x"}}]</tool_calls>`, tools);
  if (!r||r.length!==1||r[0].function.name!=='read_file') throw new Error(JSON.stringify(r));
});
test('XML plural WITHOUT closing tag (lenient)', () => {
  const r = parseToolCalls(`<tool_calls>[{"name":"read_file","arguments":{"path":"x"}}]`, tools);
  if (!r||r.length!==1||r[0].function.name!=='read_file') throw new Error(`lenient: ${JSON.stringify(r)}`);
});
test('XML singular with closing tag', () => {
  const r = parseToolCalls(`<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>`, tools);
  if (!r||r.length!==1) throw new Error(JSON.stringify(r));
});
test('XML singular WITHOUT closing tag (lenient)', () => {
  const r = parseToolCalls(`<tool_call>{"name":"read_file","arguments":{"path":"x"}}`, tools);
  if (!r||r.length!==1) throw new Error(`lenient singular: ${JSON.stringify(r)}`);
});
test('Code-fenced JSON array', () => {
  const r = parseToolCalls('```json\n[{"name":"read_file","arguments":{"path":"x"}}]\n```', tools);
  if (!r||r.length!==1) throw new Error(JSON.stringify(r));
});
test('Bare JSON array', () => {
  const r = parseToolCalls('[{"name":"read_file","arguments":{"path":"x"}}]', tools);
  if (!r||r.length!==1) throw new Error(JSON.stringify(r));
});
test('Object with tool_calls key', () => {
  const r = parseToolCalls('{"tool_calls":[{"name":"read_file","arguments":{"path":"x"}}]}', tools);
  if (!r||r.length!==1) throw new Error(JSON.stringify(r));
});
test('Multiple calls in one message', () => {
  const r = parseToolCalls(`<tool_calls>[{"name":"read_file","arguments":{"path":"a"}},{"name":"write_file","arguments":{"path":"b","content":"c"}}]</tool_calls>`, tools);
  if (!r||r.length!==2) throw new Error(`expected 2: ${JSON.stringify(r)}`);
});
test('Unknown tool name → null', () => {
  if (parseToolCalls('[{"name":"nope","arguments":{}}]', tools) !== null) throw new Error('expected null');
});
test('Natural language → null', () => {
  if (parseToolCalls('Hello! How can I help?', tools) !== null) throw new Error('expected null');
});
test('Empty/whitespace → null', () => {
  if (parseToolCalls('   ', tools) !== null) throw new Error('expected null');
});
test('Whitespace around XML tags', () => {
  const r = parseToolCalls('  \n<tool_calls>  [{"name":"read_file","arguments":{"path":"x"}}]  </tool_calls>  ', tools);
  if (!r||r.length!==1) throw new Error(JSON.stringify(r));
});
test('Stringified arguments JSON', () => {
  const r = parseToolCalls(`<tool_calls>[{"name":"read_file","arguments":"{\\"path\\":\\"x\\"}"}]</tool_calls>`, tools);
  if (!r||r.length!==1) throw new Error(JSON.stringify(r));
});

// ── Heuristic ──
console.log('\n=== Heuristic ===');
test('Blocks natural language', () => { if (/[\[\{<`]/.test('Hello!'.slice(0,200))) throw new Error('should block'); });
test('Passes for {', () => { if (!/[\[\{<`]/.test('{'.slice(0,200))) throw new Error('should pass'); });
test('Passes for [', () => { if (!/[\[\{<`]/.test('['.slice(0,200))) throw new Error('should pass'); });
test('Passes for <', () => { if (!/[\[\{<`]/.test('<tool'.slice(0,200))) throw new Error('should pass'); });
test('Passes for `', () => { if (!/[\[\{<`]/.test('`json'.slice(0,200))) throw new Error('should pass'); });

// ── SSE ──
console.log('\n=== SSE Parser ===');
test('Snapshot event parsing', () => {
  let c = ''; const p = new DeepSeekSSEParser(x => { c += x; });
  p.feed('data: {"v":{"response":{"fragments":[{"type":"RESPONSE","content":"Hello"}]}}}\n');
  p.flush();
  if (!c.includes('Hello')) throw new Error(`got: "${c}"`);
});
test('Patch event parsing', () => {
  let c = ''; const p = new DeepSeekSSEParser(x => { c += x; });
  p.feed('data: {"v":{"response":{"fragments":[{"type":"RESPONSE","content":"He"}]}}}\n');
  p.feed('data: {"p":"response/fragments/0/content","v":"llo"}\n');
  p.flush();
  if (c !== 'Hello') throw new Error(`got: "${c}"`);
});

// ── Phase 5: Code quality ──
console.log('\n=== Code quality ===');
test('transport.js: TextDecoder singleton (≤1 allocation)', () => {
  const c = (readFileSync('./packages/core/dist/transport.js','utf-8').match(/new TextDecoder\(\)/g)||[]).length;
  if (c > 1) throw new Error(`${c} allocations (expected 1)`);
});
test('transport.js: TextEncoder singleton (≤1 allocation)', () => {
  const c = (readFileSync('./packages/core/dist/transport.js','utf-8').match(/new TextEncoder\(\)/g)||[]).length;
  if (c > 1) throw new Error(`${c} allocations (expected 1)`);
});
test('transport.js: has encodeSSE', () => {
  if (!readFileSync('./packages/core/dist/transport.js','utf-8').includes('encodeSSE')) throw new Error('missing');
});
test('transport.js: has DATA_DONE', () => {
  if (!readFileSync('./packages/core/dist/transport.js','utf-8').includes('DATA_DONE')) throw new Error('missing');
});
test('pow.js: deepSeekHashV1 removed', () => {
  if (readFileSync('./packages/core/dist/pow.js','utf-8').includes('deepSeekHashV1')) throw new Error('present');
});
test('pow.js: solvePoWJS removed', () => {
  if (readFileSync('./packages/core/dist/pow.js','utf-8').includes('solvePoWJS')) throw new Error('present');
});
test('pow.js: WASM error message', () => {
  if (!readFileSync('./packages/core/dist/pow.js','utf-8').includes('WASM solver not found')) throw new Error('missing');
});
test('transport.js: timeout default 5000', () => {
  if (!readFileSync('./packages/core/dist/transport.js','utf-8').includes('"5000"')) throw new Error('missing');
});
test('transport.js: max_tokens not 1000', () => {
  if (readFileSync('./packages/core/dist/transport.js','utf-8').match(/max_tokens:\s*1000/)) throw new Error('still 1000');
});
test('transport.js: has _powCache', () => {
  if (!readFileSync('./packages/core/dist/transport.js','utf-8').includes('_powCache')) throw new Error('missing');
});
test('transport.js: flush threshold 80', () => {
  const s = readFileSync('./packages/core/dist/transport.js','utf-8');
  if (s.match(/\.length > 20\b/)) throw new Error('still 20');
  if (!s.match(/\.length > 80\b/)) throw new Error('missing 80');
});
test('tool-calls.js: lenient XML', () => {
  if (!readFileSync('./packages/core/dist/tool-calls.js','utf-8').includes('lenientPlural')) throw new Error('missing');
});
test('build-wasm.js: SIMD flags', () => {
  if (!readFileSync('./scripts/build-wasm.js','utf-8').includes('-msimd128')) throw new Error('missing');
});
test('build-wasm.js: bulk-memory flags', () => {
  if (!readFileSync('./scripts/build-wasm.js','utf-8').includes('-mbulk-memory')) throw new Error('missing');
});

console.log(`\n${'='.repeat(40)}`);
console.log(`RESULTS: ${passed} passed, ${failed} failed`);
console.log(`${'='.repeat(40)}`);
if (failed > 0) process.exit(1);
