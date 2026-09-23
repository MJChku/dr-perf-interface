'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const E = require('../media/expressions');
const M = require('../media/model');
const model = require('../demo/pipeline.drperf.json');
const value = (text, n = 7n) => E.evaluate(E.compile(text), () => n);

test('integer expression precedence, signs, exact arithmetic, and Python floor division', () => {
  assert.equal(value('2 + 3 * 4 ** 2'), 50n);
  assert.equal(value('-2 ** 2'), -4n);
  assert.equal(value('2 ** 3 ** 2'), 512n);
  assert.equal(value('-7 // 3'), -3n);
  assert.equal(value('-7 % 3'), 2n);
  assert.equal(value('9007199254740993 + 7'), 9007199254741000n);
});
test('conditional state relationships evaluate only the selected branch', () => {
  assert.equal(value('last("r", "n") < 10 ? 2 * last("r", "n") : 1 // 0'), 14n);
  assert.equal(value('0 && 1 // 0'), 0n);
  assert.equal(value('1 || 1 // 0'), 1n);
  assert.equal(value('1 ? 0 ? 7 : 8 : 9'), 8n);
});
test('expressions cannot execute code or grow unbounded', () => {
  for (const text of ['process.exit()', 'globalThis["x"]', 'last("r","n");1', 'eval("1")', '1 / 2'])
    assert.throws(() => value(text));
  assert.throws(() => value('2 ** 1000'), /exponent/);
  assert.throws(() => value('1 // 0'), /zero/);
  assert.throws(() => value('('.repeat(60) + '1' + ')'.repeat(60)), /deeply/);
});
test('proposed nonlinear equation is checked at every target call', () => {
  const changed = structuredClone(model);
  changed.relations = changed.relations.filter(
    (r) => r.target.region !== 'lookup' && !r.terms.some((t) => t.region === 'lookup')
  );
  const lookup = changed.regions.find((r) => r.id === 'lookup');
  lookup.states = ['pairs'];
  lookup.regimes = [];
  lookup.points = [];
  for (const event of changed.trace.events)
    if (event.region === 'lookup') event.values = { pairs: event.values.entries ** 2 };
  const target = { region: 'lookup', state: 'pairs' };
  const correct = M.proposeRelationship(changed, target, 'last("dequeue", "items") ** 2');
  assert.equal(correct.check.holds, true);
  assert.equal(correct.check.calls, 24);
  const incorrect = M.proposeRelationship(changed, target, '2 * last("dequeue", "items")');
  assert.equal(incorrect.check.holds, false);
  assert.equal(incorrect.check.mismatches, 24);
  const result = M.replay(
    changed,
    {
      edits: [{ region: 'dequeue', state: 'items', op: 'scale', value: 2 }],
      relations: [correct.relation.id],
      proposals: [{ target, expression: correct.relation.expression }]
    },
    { includeTrace: true }
  );
  const baseline = changed.trace.events.find((e) => e.region === 'lookup');
  assert.equal(
    result.predictedEvents.find((e) => e.region === 'lookup').values.pairs,
    4 * baseline.values.pairs
  );
});
test('a conditional relationship can encode branch-dependent PCVs', () => {
  const changed = structuredClone(model);
  const dispatch = changed.regions.find((r) => r.id === 'dispatch');
  dispatch.states = ['work_items'];
  dispatch.regimes = [];
  dispatch.points = [];
  for (const event of changed.trace.events)
    if (event.region === 'dispatch')
      event.values = {
        work_items: event.values.items < 70 ? 4 * event.values.items : event.values.items
      };
  const proposal = M.proposeRelationship(
    changed,
    { region: 'dispatch', state: 'work_items' },
    'last("dequeue", "items") < 70 ? 4 * last("dequeue", "items") : last("dequeue", "items")'
  );
  assert.equal(proposal.check.holds, true);
  assert.equal(proposal.check.calls, 24);
});
