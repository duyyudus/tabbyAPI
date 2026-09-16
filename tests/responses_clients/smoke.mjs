import assert from 'node:assert/strict';
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'fixture-key', baseURL: process.env.TABBY_TEST_URL || 'http://127.0.0.1:18081/v1',
  maxRetries: 0,
});
const base = {model: 'test-model', store: false};
const text = await client.responses.create({...base, input: 'hello'});
assert.equal(text.output_text, 'hello');
const stream = await client.responses.create({...base, input: 'hello', stream: true});
const events = [];
for await (const event of stream) events.push(event);
assert.equal(events.at(-1).type, 'response.completed');
assert.equal(events.filter(e => e.type === 'response.output_text.delta').map(e => e.delta).join(''), 'hello');
assert.deepEqual(events.map(e => e.sequence_number), events.map((_, i) => i));

for (const tool of [
  {type: 'function', name: 'read', parameters: {type: 'object', properties: {path: {type: 'string'}}, required: ['path'], additionalProperties: false}, strict: true},
  {type: 'custom', name: 'patch', format: {type: 'grammar', syntax: 'lark', definition: 'start: "*** Begin Patch" "\\n" "*** End Patch" "\\n"'}},
]) {
  const input = [{role: 'user', content: 'use tool'}];
  const first = await client.responses.create({...base, input, tools: [tool]});
  const call = first.output.find(i => i.type === (tool.type === 'function' ? 'function_call' : 'custom_tool_call'));
  assert.ok(call, JSON.stringify(first));
  if (tool.type === 'custom') assert.equal(call.input, '*** Begin Patch\n*** End Patch\n');
  const result = {type: tool.type === 'function' ? 'function_call_output' : 'custom_tool_call_output', call_id: call.call_id, output: 'ok'};
  const next = await client.responses.create({...base, input: [...input, ...first.output, result], tools: [tool]});
  assert.equal(next.output_text, 'tool result received');
}
console.log('JavaScript SDK: text, SSE, function and custom tool round trips passed.');
