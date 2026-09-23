'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const M = require('../media/model');
function fixture(rounds = 5, groups = 1) {
  const events = [];
  for (let group = 0; group < groups; ++group)
    for (let i = 1; i <= rounds; ++i) {
      events.push({
        group: String(group),
        thread: '1',
        seq: i * 4 - 3,
        end: i * 4 - 2,
        region: 'producer',
        values: { n: i }
      });
      events.push({
        group: String(group),
        thread: '1',
        seq: i * 4 - 1,
        end: i * 4,
        region: 'consumer',
        values: { n: 2 * i }
      });
    }
  return {
    schema: M.schema,
    id: `fixture-${rounds}-${groups}`,
    measurement: { unit: 'CPU instructions per call', scope: 'exclusive region work' },
    validity: { errors: [], traceErrors: [] },
    regions: ['producer', 'consumer'].map((id) => ({
      id,
      name: id,
      originalName: id,
      states: ['n'],
      calls: rounds * groups,
      sources: [],
      regimes: [],
      points: []
    })),
    trace: { complete: true, recordCount: events.length, events },
    relations: [
      {
        id: 'double',
        target: { region: 'consumer', state: 'n' },
        constant: 0,
        terms: [{ kind: 'last', region: 'producer', state: 'n', coefficient: 2 }]
      }
    ]
  };
}
test('relationship checking uses new history without requiring matching call counts', () => {
  const result = M.checkRelationships(fixture(), fixture(9, 2));
  assert.equal(result.summary.holds, 1);
  assert.equal(result.checks[0].baseline.calls, 5);
  assert.equal(result.checks[0].measured.calls, 18);
  assert.equal(Object.hasOwn(result, 'totalCost'), false);
});
test('count history resets at each measured process/run boundary', () => {
  const proposal = {
    target: { region: 'producer', state: 'n' },
    expression: 'count("producer") + 1'
  };
  const result = M.checkRelationships(fixture(), fixture(9, 2), { proposals: [proposal] });
  assert.equal(result.summary.holds, 2);
  assert.ok(result.checks.every((check) => check.measured.calls === 18));
});
test('schema variants map by original name and PCV names, including proposed expressions', () => {
  const reference = fixture(),
    measured = fixture(7);
  for (const region of measured.regions) region.id += '@variant';
  for (const event of measured.trace.events) event.region += '@variant';
  measured.relations = [];
  const proposal = {
    target: { region: 'consumer', state: 'n' },
    expression: '2 * last("producer", "n")'
  };
  const result = M.checkRelationships(reference, measured, { proposals: [proposal] });
  assert.equal(result.summary.holds, 2);
});
test('missing schemas, unexercised targets and failed baseline equations stay distinct', () => {
  const baseline = fixture(),
    missing = fixture();
  missing.regions[0].states.push('m');
  for (const event of missing.trace.events) if (event.region === 'producer') event.values.m = 1;
  assert.equal(M.checkRelationships(baseline, missing).checks[0].status, 'unavailable');
  const absent = fixture();
  absent.trace.events = absent.trace.events.filter((e) => e.region === 'producer');
  absent.trace.recordCount = absent.trace.events.length;
  absent.regions[1].calls = 0;
  assert.equal(M.checkRelationships(baseline, absent).checks[0].status, 'not-exercised');
  const wrong = fixture();
  wrong.relations[0].terms[0].coefficient = 3;
  assert.equal(M.checkRelationships(wrong, baseline).checks[0].status, 'baseline-failed');
});
test('measured counterexamples and changed source PCV definitions are retained', () => {
  const reference = fixture(),
    measured = fixture(7);
  const source = {
    path: 'example.py',
    line: 1,
    endLine: 2,
    sha256: 'a'.repeat(64),
    expressions: { n: 'count' }
  };
  reference.regions[1].sources = [source];
  measured.regions[1].sources = [{ ...source, expressions: { n: 'count + 1' } }];
  for (const event of measured.trace.events) if (event.region === 'consumer') ++event.values.n;
  const check = M.checkRelationships(reference, measured).checks[0];
  assert.equal(check.status, 'fails');
  assert.equal(check.measured.mismatches, 7);
  assert.equal(check.measured.examples[0].expected, '3');
  assert.equal(check.measured.examples[0].predicted, '2');
  assert.deepEqual(check.changedExpressions, [{ region: 'consumer', state: 'n' }]);
});
test('an independent measured expansion run keeps one copy explanation and rejects another', () => {
  const reference = require('../demo/refined.drperf.json'),
    measured = require('../demo/expansion-changed.drperf.json');
  const copy = M.checkRelationships(reference, measured).checks.filter(
    (c) => c.target.region === 'copy'
  );
  assert.equal(copy.length, 2);
  assert.equal(copy.find((c) => c.equation.includes('decode.tokens')).status, 'holds');
  assert.equal(copy.find((c) => c.equation.includes('dequeue.items')).status, 'fails');
});
